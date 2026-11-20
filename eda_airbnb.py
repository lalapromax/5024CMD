
# argparse reads optional command-line settings such as the database path and output folder.
import argparse
import os
# sqlite3 provides the connection used to read the selected table from the SQLite database.
import sqlite3
from pathlib import Path

# Use a non-interactive plotting backend so figures can be saved on computers without a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIGURATION
# These constants keep column names, plot settings and repeatability choices
# in one place so the analysis functions use consistent settings.
# ----------------------------------------------------------------------------
PURPLE = "#9966CC"
ORANGE = "#E8590C"
GREY = "#6B7280"

# Numeric variables included in descriptive statistics and outlier analysis.
NUMERIC = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]
# Exclude the target variable price to obtain the numeric explanatory factors.
FACTORS = NUMERIC[1:]
# Geographic coordinates are analysed separately from the other numeric factors.
GEO = ["latitude", "longitude"]
SHORT = {
    "price": "price",
    "minimum_nights": "min_nights",
    "number_of_reviews": "n_reviews",
    "reviews_per_month": "reviews/month",
    "calculated_host_listings_count": "host_listings",
    "availability_365": "availability",
    "latitude": "latitude",
    "longitude": "longitude",
}
# Fixed order prevents price-tier tables and charts from changing category order.
BIN_ORDER = ["Low", "Medium", "High"]
SEED = 42
# Scatterplots draw a reproducible sample to keep the figures readable and efficient.
# The orange trend line is still calculated using the complete dataset.
SCATTER_SAMPLE = 20_000

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "axes.titlesize": 11})

usd = FuncFormatter(lambda v, _: f"${v:,.0f}")


# Save a completed figure, close it to release memory, and report the output filename.
def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


# Export a result table as a CSV file without adding an unnecessary row index.
def save_table(df, out, name):
    df.to_csv(os.path.join(out, name), index=False)
    print(f"  saved {name}")


# ----------------------------------------------------------------------------
# 3.1 SINGLE-VARIABLE ANALYSIS
# Examine each variable on its own using summary statistics, distributions,
# skewness and IQR-based outlier counts.
# ----------------------------------------------------------------------------
def table_summary(df):
    # Calculate common descriptive statistics for every numeric variable.
    # Quartiles describe the middle spread, while skewness and kurtosis describe shape.
    rows = []
    for c in NUMERIC:
        s = df[c]
        rows.append(
            {
                "variable": c,
                "count": s.count(),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "q1": s.quantile(0.25),
                "q3": s.quantile(0.75),
                "max": s.max(),
                "skewness": s.skew(),
                "excess_kurtosis": s.kurt(),
            }
        )
    return pd.DataFrame(rows).round(3)


def table_skew_log(df):
    # Compare skewness on the original scale and after log(1 + x).
    # log1p is suitable for non-negative values and also handles zero values.
    rows = []
    for c in NUMERIC:
        rows.append(
            {
                "variable": c,
                "skew_raw": df[c].skew(),
                "skew_log1p": np.log1p(df[c]).skew(),
            }
        )
    t = pd.DataFrame(rows).round(2)
    t["interpretation"] = t["skew_raw"].apply(
        lambda v: "highly skewed (|s|>1)" if abs(v) > 1 else
        "moderately skewed" if abs(v) > 0.5 else "approximately symmetric"
    )
    return t


def table_outliers(df):
    # Apply the 1.5 x IQR rule separately to each numeric variable.
    # Values outside the lower and upper fences are counted, not removed.
    rows = []
    for c in NUMERIC:
        s = df[c]
        # IQR is the middle 50% range; the 1.5 x IQR fences flag potential outliers.
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        # Observations outside these fences are counted as potential outliers.
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_hi, n_lo = int((s > hi).sum()), int((s < lo).sum())
        rows.append(
            {
                "variable": c,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lo,
                "upper_fence": hi,
                "n_below": n_lo,
                "n_above": n_hi,
                "pct_outliers": 100 * (n_hi + n_lo) / len(s),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows).round(2)


def fig_price_distribution(df, out):
    # Compare raw price, log-transformed price and a log-scale boxplot.
    # The raw histogram is limited to $1,000 for readability; all prices
    # are still used to calculate the displayed statistics.
    p = df["price"]
    cut = 1000
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].hist(p[p <= cut], bins=60, color=PURPLE)
    ax[0].set(title="Price, raw scale (shown up to $1,000)",
              xlabel="Nightly price (USD)", ylabel="Number of listings")
    ax[0].xaxis.set_major_formatter(usd)
    ax[0].text(0.97, 0.95,
               f"skewness = {p.skew():.1f}\nmean = ${p.mean():,.0f}\nmedian = ${p.median():,.0f}\n"
               f"{(p > cut).sum():,} listings above $1,000\nnot shown (max ${p.max():,})",
               transform=ax[0].transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))
    ax[0].axvline(p.median(), color=ORANGE, ls="--", lw=1.5, label="median")
    ax[0].axvline(p.mean(), color="black", ls=":", lw=1.5, label="mean")
    ax[0].legend(loc="center right", fontsize=9)

    lp = np.log1p(p)
    ax[1].hist(lp, bins=60, color=PURPLE)
    ax[1].set(title="Price after log(1 + price)", xlabel="log(1 + price)", ylabel="Number of listings")
    ax[1].text(0.97, 0.95, f"skewness = {lp.skew():.2f}", transform=ax[1].transAxes,
               ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))

    sns.boxplot(x=p, ax=ax[2], color="#BFD3F7", fliersize=1.5,
                flierprops=dict(markerfacecolor=PURPLE, alpha=0.3, markeredgecolor="none"))
    ax[2].set_xscale("log")
    ax[2].xaxis.set_major_formatter(usd)
    ax[2].set(title="Price boxplot (log axis; points = outliers by 1.5 x IQR)",
              xlabel="Nightly price (USD, log scale)")
    fig.suptitle("Distribution of nightly price", fontsize=13, y=1.02)
    save(fig, out, "fig_3_1a_price_distribution.png")


def fig_numeric_histograms(df, out):
    # Plot each explanatory numeric factor twice: raw values above and
    # log-transformed values below. The raw view is capped visually at p99
    # so extreme values do not hide the distribution of most listings.
    fig, axes = plt.subplots(2, len(FACTORS), figsize=(19, 6.5))
    for j, c in enumerate(FACTORS):
        s = df[c]
        p99 = s.quantile(0.99)
        raw = s[s <= p99]
        axes[0, j].hist(raw, bins=50, color=PURPLE)
        axes[0, j].set_title(f"{SHORT[c]}\nskew = {s.skew():.2f}", fontsize=10)
        axes[0, j].set_xlabel(f"raw (up to 99th pct = {p99:,.0f})", fontsize=8)
        ls = np.log1p(s)
        axes[1, j].hist(ls, bins=50, color="#7FA6EC")
        axes[1, j].set_title(f"log(1 + {SHORT[c]})\nskew = {ls.skew():.2f}", fontsize=10)
        axes[1, j].set_xlabel("log scale", fontsize=8)
    axes[0, 0].set_ylabel("Number of listings")
    axes[1, 0].set_ylabel("Number of listings")
    fig.suptitle("Distributions of the continuous factors: raw (top) vs log-transformed (bottom)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, out, "fig_3_1b_numeric_histograms.png")


def fig_numeric_boxplots(df, out, outliers):
    # Display all numeric variables with a symmetric-log axis, which can
    # show values near zero and large values on the same plot.
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for ax, c in zip(axes.flat, NUMERIC):
        sns.boxplot(x=df[c], ax=ax, color="#BFD3F7", fliersize=1.5,
                    flierprops=dict(markerfacecolor=PURPLE, alpha=0.3, markeredgecolor="none"))
        # Symmetric-log scaling keeps small values readable alongside extreme values.
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0, right=df[c].max() * 1.3)
        row = outliers.loc[outliers.variable == c].iloc[0]
        ax.set_title(f"{c}\n{int(row.n_above + row.n_below):,} outliers ({row.pct_outliers:.1f}%)", fontsize=10)
        ax.set_xlabel("value (symlog axis)")
    fig.suptitle("Boxplots of every numeric variable (outliers = beyond 1.5 x IQR)", fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig_3_1c_numeric_boxplots.png")


def fig_skew_before_after(skew, out):
    # Put raw and transformed skewness side by side to show whether
    # the log transformation makes each distribution more symmetric.
    t = skew.set_index("variable")[["skew_raw", "skew_log1p"]]
    t.index = [SHORT[i] for i in t.index]
    t.columns = ["raw", "log(1 + x)"]
    ax = t.plot.barh(figsize=(9, 5), color=[PURPLE, ORANGE], width=0.75)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8, label="_nolegend_")
    ax.set(title="Skewness before and after log(1 + x)", xlabel="Skewness (0 = symmetric)")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlim(left=-2.5)
    ax.legend(loc="lower right")
    save(ax.figure, out, "fig_3_1d_skewness_before_after.png")


def fig_categorical_counts(df, out):
    # Compare the number of listings in each room type and price tier.
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    rt = df["room_type"].value_counts()
    ax[0].bar(rt.index, rt.values, color=PURPLE)
    for i, v in enumerate(rt.values):
        ax[0].text(i, v, f"{v:,}\n({100 * v / len(df):.1f}%)", ha="center", va="bottom", fontsize=8)
    ax[0].set(title="Listings by room type", ylabel="Number of listings")
    ax[0].set_ylim(0, rt.max() * 1.15)
    pb = df["price_bin"].value_counts().reindex(BIN_ORDER)
    ax[1].bar(pb.index, pb.values, color=PURPLE)
    for i, v in enumerate(pb.values):
        ax[1].text(i, v, f"{v:,}\n({100 * v / len(df):.1f}%)", ha="center", va="bottom", fontsize=8)
    ax[1].set(title="Listings by price tier (equal-frequency bins)", ylabel="Number of listings")
    ax[1].set_ylim(0, pb.max() * 1.15)
    fig.tight_layout()
    save(fig, out, "fig_3_1e_categorical_counts.png")


def fig_listings_per_city(df, out):
    # Count listings in each city and sort the bars for easier comparison.
    c = df["city"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(c.index, c.values, color=PURPLE)
    for i, v in enumerate(c.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set(title="Number of listings per city", xlabel="Number of listings")
    ax.set_xlim(0, c.max() * 1.12)
    save(fig, out, "fig_3_1f_listings_per_city.png")


# ----------------------------------------------------------------------------
# 3.2 RELATIONSHIPS BETWEEN VARIABLES
# Measure how price varies with numeric factors, room type and city.
# ----------------------------------------------------------------------------
def table_correlations(df):
    # Pearson measures linear association; Spearman measures rank-based
    # monotonic association and is less sensitive to extreme price values.
    # P-values are included to report statistical evidence for each association.
    rows = []
    for c in FACTORS + GEO:
        # Each test compares one factor with nightly price.
        pr, pp = stats.pearsonr(df[c], df["price"])
        pl, _ = stats.pearsonr(df[c], df["log1p_price"])
        sr, sp = stats.spearmanr(df[c], df["price"])
        rows.append(
            {
                "factor": c,
                "pearson_vs_price": pr,
                "pearson_p": pp,
                "pearson_vs_log1p_price": pl,
                "spearman_vs_price": sr,
                "spearman_p": sp,
            }
        )
    t = pd.DataFrame(rows)
    t[["pearson_vs_price", "pearson_vs_log1p_price", "spearman_vs_price"]] = \
        t[["pearson_vs_price", "pearson_vs_log1p_price", "spearman_vs_price"]].round(3)
    return t


def table_group_tests(df):
    # Kruskal-Wallis tests whether price distributions differ across groups.
    # Chi-square tests whether room type or city is associated with price tier.
    # Effect sizes (epsilon-squared and Cramer's V) describe association strength.
    rows = []
    n = len(df)
    for c in ["room_type", "city"]:
        groups = [g["price"].values for _, g in df.groupby(c)]
        # Kruskal-Wallis is a non-parametric alternative to one-way ANOVA.
        h, p = stats.kruskal(*groups)
        rows.append({"test": "Kruskal-Wallis", "variables": f"price ~ {c}", "statistic": h,
                     "dof": len(groups) - 1, "p_value": p, "effect_size": h / (n - 1),
                     "effect_measure": "epsilon-squared"})
    for c in ["room_type", "city"]:
        ct = pd.crosstab(df[c], df["price_bin"])
        # The contingency table contains observed counts for each category combination.
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        # Cramers V standardises the chi-square statistic as an association effect size.
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        rows.append({"test": "Chi-square", "variables": f"{c} x price_bin", "statistic": chi2,
                     "dof": dof, "p_value": p, "effect_size": v, "effect_measure": "Cramer's V"})
    return pd.DataFrame(rows).round(4)


def table_room_type(df):
    # Summarise price by room type and calculate each room type's
    # percentage distribution across the three price tiers.
    g = df.groupby("room_type")["price"].agg(count="count", median="median", mean="mean").round(1)
    shares = (pd.crosstab(df["room_type"], df["price_bin"], normalize="index") * 100).round(1)
    shares = shares.reindex(columns=BIN_ORDER).add_prefix("pct_")
    return g.join(shares).sort_values("median", ascending=False).reset_index()


def table_city(df):
    # Compare listing counts and price quartiles/averages across cities.
    g = df.groupby("city")["price"].agg(
        count="count", median="median", mean="mean",
        q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)).round(1)
    return g.sort_values("median", ascending=False).reset_index()


def fig_corr_heatmaps(df, out):
    # Show Pearson and Spearman correlation matrices for numeric and
    # geographic variables. Values close to -1 or +1 indicate stronger
    # negative or positive associations; values near zero indicate weak ones.
    cols = NUMERIC + GEO
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    for a, method in zip(ax, ["pearson", "spearman"]):
        # Calculate the full pairwise correlation matrix using the selected method.
        m = df[cols].corr(method=method)
        m.index = m.columns = [SHORT[c] for c in cols]
        sns.heatmap(m, ax=a, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    center=0, square=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 8})
        a.set_title(f"{method.capitalize()} correlation")
        a.tick_params(axis="x", rotation=45)
    fig.suptitle("Correlation matrices (all numeric variables)", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig_3_2a_correlation_heatmaps.png")


def fig_scatter(df, out, corr):
    # Use a fixed random sample for the plotted points to reduce overplotting.
    # Quantile bins and their median prices are calculated from all rows,
    # giving a stable summary trend that is not limited to the plotted sample.
    # A fixed random_state makes the same listings appear each time the script runs.
    s = df.sample(SCATTER_SAMPLE, random_state=SEED)
    specs = [(c, "symlog") for c in FACTORS[:4]] + [("availability_365", "linear")] + \
            [(g, "linear") for g in GEO]
    rho = corr.set_index("factor")["spearman_vs_price"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ax, (c, scale) in zip(axes.flat, specs):
        ax.scatter(s[c], s["price"], s=4, alpha=0.12, color=PURPLE, linewidths=0, rasterized=True)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(usd)
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=1)
        # qcut groups values into up to 20 quantile-based bins with similar row counts.
        # duplicates="drop" handles variables that do not have enough unique values.
        bins = pd.qcut(df[c], q=20, duplicates="drop")
        trend = df.groupby(bins, observed=True).agg(x=(c, "median"), y=("price", "median"))
        ax.plot(trend["x"], trend["y"], color=ORANGE, lw=2.2)
        ax.set_title(f"{c}\nSpearman rho = {rho[c]:+.3f}", fontsize=10)
        ax.set_xlabel(c + (" (symlog axis)" if scale == "symlog" else ""), fontsize=9)
        ax.set_ylabel("Nightly price (log axis)" if c in (specs[0][0], specs[4][0]) else "")
    leg = axes.flat[-1]
    leg.axis("off")
    leg.text(0, 0.6,
             f"PURPLE points: random sample of {SCATTER_SAMPLE:,}\nlistings (of {len(df):,}).\n\n"
             "Orange line: median price within 20\nquantile bins of the x-variable, computed\non ALL listings.\n\n"
             "Price axis is logarithmic.", fontsize=10, va="top")
    fig.suptitle("Relationship between each numeric factor and nightly price", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig_3_2b_scatter_vs_price.png")


def fig_geo_scatter(df, out):
    # Represent each city by its median latitude and longitude.
    # Bubble size represents listing count, while colour represents median price.
    # Group listings by city so each city is represented by one summary bubble.
    g = df.groupby("city").agg(lat=("latitude", "median"), lon=("longitude", "median"),
                               n=("price", "size"), med=("price", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(13, 7.5))
    sc = ax.scatter(g["lon"], g["lat"], s=g["n"] / 25, c=g["med"], cmap="viridis",
                    edgecolor="black", linewidth=0.5, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Median nightly price (USD)")
    # Shift selected city labels so nearby bubbles and names are easier to distinguish.
    nudge = {"San Francisco": (-95, 22), "Oakland": (-60, 40), "San Mateo County": (-115, 4),
             "San Clara Country": (-115, -12), "Santa Cruz County": (-115, -28),
             "Pacific Grove": (-85, -44), "Boston": (14, 16), "Cambridge": (-62, 18),
             "Jersey City": (-90, -40), "New York City": (-30, -58), "Salem": (-45, -12)}
    for _, r in g.iterrows():
        dx, dy = nudge.get(r["city"], (6, 5))
        ax.annotate(r["city"], (r["lon"], r["lat"]), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="grey") if r["city"] in nudge else None)
    ax.set(title="Where the 28 cities are: bubble size = number of listings, colour = median price",
           xlabel="Longitude", ylabel="Latitude")
    save(fig, out, "fig_3_2c_geographic_scatter.png")


def fig_price_by_room_type(df, out):
    # Compare the price distributions of room types, ordered by median price.
    # A logarithmic price axis makes both low and high prices visible.
    order = df.groupby("room_type")["price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=df, x="room_type", y="price", order=order, ax=ax, color="#BFD3F7",
                fliersize=1.5, flierprops=dict(markerfacecolor=PURPLE, alpha=0.25, markeredgecolor="none"))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(usd)
    meds = df.groupby("room_type")["price"].median()
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{r}\nmedian ${meds[r]:,.0f}" for r in order])
    ax.set(title="Nightly price by room type", xlabel="", ylabel="Nightly price (USD, log scale)")
    save(fig, out, "fig_3_2d_price_by_room_type.png")


def fig_median_price_city(df, out):
    # Rank cities by their median nightly price rather than their mean,
    # reducing the influence of a small number of unusually expensive listings.
    med = df.groupby("city")["price"].median().sort_values()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(med.index, med.values, color=PURPLE)
    for i, v in enumerate(med.values):
        ax.text(v, i, f" ${v:,.0f}", va="center", fontsize=8)
    ax.set_title(f"Median price ranges from \\${med.min():,.0f} to \\${med.max():,.0f} across the {len(med)} cities")
    ax.set_xlabel("Median nightly price (USD), cleaned 2020 data")
    ax.set_xlim(0, med.max() * 1.1)
    save(fig, out, "fig_3_2e_median_price_by_city.png")


def fig_city_roomtype_heatmap(df, out, min_n=30):
    # Calculate median price for each city-room-type combination.
    # Hide combinations with fewer than min_n listings to avoid highlighting
    # unstable medians based on very small groups.
    pt = df.pivot_table(index="city", columns="room_type", values="price", aggfunc="median")
    cnt = df.pivot_table(index="city", columns="room_type", values="price", aggfunc="count")
    # Replace low-count combinations with missing values so they appear blank.
    pt = pt.where(cnt >= min_n)
    order = df.groupby("city")["price"].median().sort_values(ascending=False).index
    cols = df.groupby("room_type")["price"].median().sort_values(ascending=False).index
    pt = pt.loc[order, cols]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pt, annot=True, fmt=".0f", cmap="Purples", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Median nightly price (USD)"}, annot_kws={"size": 8})
    ax.set(title=f"Median price by city and room type\n(blank = fewer than {min_n} listings)",
           xlabel="", ylabel="")
    ax.grid(False)
    save(fig, out, "fig_3_2f_city_roomtype_heatmap.png")


def fig_roomtype_vs_bin(df, out):
    # Convert price-tier counts into percentages within each room type.
    # Each stacked bar therefore totals 100%, enabling fair comparison
    # between room types with very different numbers of listings.
    # normalize="index" calculates proportions separately within each room type.
    ct = pd.crosstab(df["room_type"], df["price_bin"], normalize="index")[BIN_ORDER] * 100
    ct = ct.loc[df.groupby("room_type")["price"].median().sort_values(ascending=False).index]
    ax = ct.plot.barh(stacked=True, figsize=(10, 4.5), color=["#D3D3E8", "#7B72B4", "#552890"], width=0.7)
    ax.invert_yaxis()
    for cont, colour in zip(ax.containers, ["black", "black", "white"]):
        ax.bar_label(cont, fmt="%.1f%%", label_type="center", fontsize=8, color=colour)
    ax.set_xlim(0, 100)
    ax.set(title="Share of each price tier within each room type", xlabel="Percent of listings", ylabel="")
    ax.legend(title="Price tier", bbox_to_anchor=(1.01, 1), loc="upper left")
    save(ax.figure, out, "fig_3_2g_room_type_vs_price_bin.png")


# ----------------------------------------------------------------------------
# MAIN PROGRAM
# Read the selected SQLite table, run all analyses, and save tables and figures.
# ----------------------------------------------------------------------------
def main():
    # Resolve default paths relative to this Python file, not the terminal's
    # current working directory. Command-line arguments can override these paths.
    base_dir = Path(__file__).resolve().parent
    default_db = base_dir / "data" / "airbnb.db"
    default_out = base_dir / "EDA" / "eda_output"

    # Allow the database file, source table and output folder to be customised.
    ap = argparse.ArgumentParser()
    ap.add_argument("db", nargs="?", default=str(default_db), help="Path to SQLite database")
    ap.add_argument("--table", default="Cleaned Data Set", help="SQLite table name to query")
    ap.add_argument("--out", default=str(default_out), help="Output directory")
    args = ap.parse_args()

    # Stop early with a clear error if the requested database does not exist.
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Could not find SQLite database at: {db_path.resolve()} (cwd: {Path.cwd()})"
        )

    out_dir = str(Path(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # Open the database and load the chosen table into a pandas DataFrame.
    # The connection context manager closes the connection when reading finishes.
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{args.table}"', conn)

    # Create the transformed target only when it is not already stored in SQLite.
    if "log1p_price" not in df.columns:
        df["log1p_price"] = np.log1p(df["price"])

    print(f'Loaded {len(df):,} rows x {df.shape[1]} columns from {db_path} (table: "{args.table}")\n')

    # Run Section 3.1: calculate descriptive tables and create distribution plots.
    print("3.1 Single-variable analysis")
    summary = table_summary(df)
    skew = table_skew_log(df)
    outl = table_outliers(df)
    save_table(summary, out_dir, "table_3_1a_summary_stats.csv")
    save_table(skew, out_dir, "table_3_1b_skewness_log.csv")
    save_table(outl, out_dir, "table_3_1c_outliers_iqr.csv")
    fig_price_distribution(df, out_dir)
    fig_numeric_histograms(df, out_dir)
    fig_numeric_boxplots(df, out_dir, outl)
    fig_skew_before_after(skew, out_dir)
    fig_categorical_counts(df, out_dir)
    fig_listings_per_city(df, out_dir)

    # Run Section 3.2: calculate associations, group comparisons and relationship plots.
    print("\n3.2 Relationships between variables")
    corr = table_correlations(df)
    tests = table_group_tests(df)
    rt = table_room_type(df)
    city = table_city(df)
    save_table(corr, out_dir, "table_3_2a_correlations.csv")
    save_table(tests, out_dir, "table_3_2b_group_tests.csv")
    save_table(rt, out_dir, "table_3_2c_room_type_summary.csv")
    save_table(city, out_dir, "table_3_2d_city_summary.csv")
    fig_corr_heatmaps(df, out_dir)
    fig_scatter(df, out_dir, corr)
    fig_geo_scatter(df, out_dir)
    fig_price_by_room_type(df, out_dir)
    fig_median_price_city(df, out_dir)
    fig_city_roomtype_heatmap(df, out_dir)
    fig_roomtype_vs_bin(df, out_dir)

    # Print key result tables in the terminal as a quick review after files are saved.
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for title, t in [
        ("Summary statistics", summary),
        ("Skewness raw vs log", skew),
        ("IQR outliers", outl),
        ("Correlations with price", corr),
        ("Group tests", tests),
        ("Room type", rt),
    ]:
        print(f"\n--- {title} ---\n{t.to_string(index=False)}")
    print(f"\nDone. All files are in {os.path.abspath(out_dir)}")


# Run main only when this file is executed directly, not when imported by another script.
if __name__ == "__main__":
    main()
