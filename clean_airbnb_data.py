import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path("data") / "airbnb_US_2020.csv"
STANDALONE_DB = Path("airbnb.db")
DEFAULT_WEKA_CSV = Path("data") / "airbnb_US_2020_clean_FINAL.csv"
RAW_TABLE = "Raw Data Set"
CLEANED_TABLE = "Cleaned Data Set"

OUTPUT_COLUMNS = [
    "latitude",
    "longitude",
    "room_type",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "city",
    "price_outlier_iqr",
    "price_capped",
    "log1p_price",
    "price_bin",
    "minimum_nights_bin",
    "number_of_reviews_bin",
    "reviews_per_month_bin",
    "calculated_host_listings_count_bin",
    "availability_365_bin",
]

BIN_COLUMNS = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

NUMERIC_BASE_COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    #Return True if a table with table_name already exists in SQLite.
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def clean_airbnb_data(df: pd.DataFrame) -> pd.DataFrame:
    #Clean the raw Airbnb data and return the selected 19 columns.

    required_columns = [
        "id", "latitude", "longitude", "room_type", "price",
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365", "city",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    for col in NUMERIC_BASE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    starting_rows = len(df)
    print(f"Loaded {starting_rows:,} rows and {df.shape[1]} columns.")

    # 1. Remove duplicate listings using the listing ID.
    before = len(df)
    df = df.drop_duplicates(subset="id", keep="first").copy()
    print(f"1. Duplicate listing IDs removed: {before - len(df):,}")

    # 2. Remove listings with invalid or non-positive prices.
    before = len(df)
    df = df[df["price"] > 0].copy()
    print(f"2. Listings with price <= 0 removed: {before - len(df):,}")

    # 3. Remove implausible minimum-night requirements.
    before = len(df)
    df = df[df["minimum_nights"] <= 365].copy()
    print(f"3. Listings with minimum_nights > 365 removed: {before - len(df):,}")

    # 4. Impute reviews_per_month for listings that have never received reviews.
    zero_reviews = df["number_of_reviews"] == 0
    missing_before = df["reviews_per_month"].isna().sum()
    df.loc[zero_reviews, "reviews_per_month"] = (
        df.loc[zero_reviews, "reviews_per_month"].fillna(0)
    )
    print(
        "4. Missing reviews_per_month values imputed for listings with "
        f"zero reviews: {missing_before - df['reviews_per_month'].isna().sum():,}"
    )

    # The target dataset has no missing values in its selected attributes.
    # Any remaining missing values indicate an unexpected data-quality issue.
    selected_base_columns = [
        "latitude", "longitude", "room_type", "price", "minimum_nights",
        "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365", "city",
    ]
    remaining_missing = df[selected_base_columns].isna().sum()
    if remaining_missing.any():
        raise ValueError(
            "Unexpected missing values remain in selected columns:\n"
            + remaining_missing[remaining_missing > 0].to_string()
        )

    # 5. Flag price outliers using the IQR rule. Do not remove them.
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    df["price_outlier_iqr"] = (df["price"] > upper_fence).astype(int)

    # Cap prices at the 99th percentile for outlier-sensitive analysis.
    cap_value = df["price"].quantile(0.99)
    df["price_capped"] = np.minimum(df["price"], cap_value).astype(int)

    print(
        f"5. Price outliers flagged: {df['price_outlier_iqr'].sum():,}; "
        f"99th-percentile cap: ${cap_value:,.0f}"
    )

    # 6. Add the log-transformed price.
    df["log1p_price"] = np.log1p(df["price"])
    print("6. log1p_price generated.")

    # 7. Create equal-frequency price categories (terciles).
    price_q1, price_q2 = df["price"].quantile([1 / 3, 2 / 3])
    df["price_bin"] = pd.cut(
        df["price"],
        bins=[-np.inf, price_q1, price_q2, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    print(
        f"7. price_bin generated using cut points "
        f"${price_q1:.0f} and ${price_q2:.0f}."
    )

    # 8. Create equal-frequency categorical bins for the other numeric factors.
    for col in BIN_COLUMNS:
        q1, q2 = df[col].quantile([1 / 3, 2 / 3])
        df[f"{col}_bin"] = pd.cut(
            df[col],
            bins=[-np.inf, q1, q2, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)

    print("8. Low/Medium/High bins generated for the five numeric attributes.")

    # 9. Keep only the required attributes and enforce the target column order.
    cleaned = df[OUTPUT_COLUMNS].copy()

    print(
        f"\nCleaning complete: {starting_rows:,} -> {len(cleaned):,} rows "
        f"({starting_rows - len(cleaned):,} removed)."
    )
    print(f"Output columns: {len(cleaned.columns)}")
    print("Missing values in output:", int(cleaned.isna().sum().sum()))

    return cleaned


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load airbnb_US_2020.csv into standalone airbnb.db, create Cleaned Data Set, and export Weka CSV."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"Path to raw CSV file (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_WEKA_CSV),
        help=f"Path to output Weka CSV file (default: {DEFAULT_WEKA_CSV})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the Weka CSV file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing SQLite tables if they already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    db_path = Path(STANDALONE_DB.name)  # Always standalone airbnb.db (no /data)

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {csv_path.resolve()}")

    db_existed = db_path.exists()
    raw_df = None
    cleaned_df = None

    with sqlite3.connect(db_path) as conn:
        raw_exists = table_exists(conn, RAW_TABLE)
        cleaned_exists = table_exists(conn, CLEANED_TABLE)

        if not raw_exists or args.force:
            print(f"Reading raw dataset: {csv_path}")
            raw_df = pd.read_csv(csv_path, low_memory=False)
            raw_df.to_sql(RAW_TABLE, conn, if_exists="replace", index=False)
            print(f'Wrote "{RAW_TABLE}" ({len(raw_df):,} rows) to {db_path}.')
        else:
            print(f'"{RAW_TABLE}" already exists in {db_path}; skipping raw insert.')

        if not cleaned_exists or args.force:
            if raw_df is None:
                print(f"Reading raw dataset: {csv_path}")
                raw_df = pd.read_csv(csv_path, low_memory=False)
            cleaned_df = clean_airbnb_data(raw_df)
            cleaned_df.to_sql(CLEANED_TABLE, conn, if_exists="replace", index=False)
            print(f'Wrote "{CLEANED_TABLE}" ({len(cleaned_df):,} rows) to {db_path}.')
        else:
            print(f'"{CLEANED_TABLE}" already exists in {db_path}; skipping cleaning insert.')

        conn.commit()

        if not args.no_csv:
            if cleaned_df is None:
                cleaned_df = pd.read_sql_query(f'SELECT * FROM "{CLEANED_TABLE}"', conn)
            weka_csv_path = Path(args.csv)
            if weka_csv_path.parent and str(weka_csv_path.parent) != ".":
                weka_csv_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_csv(weka_csv_path, index=False, float_format="%.10g")
            print(f"Weka CSV saved to: {weka_csv_path.resolve()}")

    status_msg = "Created and populated" if not db_existed else "Updated"
    print(f"\n{status_msg} standalone SQLite database at: {db_path.resolve()}")


if __name__ == "__main__":
    main()