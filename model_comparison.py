import argparse
import gc
import sqlite3
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42

SKEWED = ["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]
NOT_SKEWED = ["availability_365"]
GEO = ["latitude", "longitude"]
CONT5 = SKEWED + NOT_SKEWED                      # the five factors that have *_bin versions
CONT_FEATS = GEO + CONT5                         # continuous predictors (as in the first version)
CAT_FEATS = ["room_type", "city"]
BIN_FEATS = [c + "_bin" for c in CONT5]
TIERS = ["Low", "Medium", "High"]

# columns derived from price: they must never be used as predictors
FORBIDDEN = {"price", "price_capped", "log1p_price", "price_bin", "price_outlier_iqr"}
for _cols in (CONT_FEATS, CAT_FEATS, BIN_FEATS):
    assert not FORBIDDEN & set(_cols), "price-derived column in the feature list (target leakage)"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def resolve_default_db() -> Path:
    """Resolve standalone airbnb.db next to the script or current working directory."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "airbnb.db", Path.cwd() / "airbnb.db"):
        if candidate.exists():
            return candidate
    return here / "airbnb.db"


def save_table_to_db(df: pd.DataFrame, db_path: Path, table_name: str, index: bool = False):
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=index)
        conn.commit()
    print(f'  saved SQLite table "{table_name}" ({len(df)} rows) -> {db_path.resolve()}')


def log_then_scale():
    return Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])


def linear_pre(log_skewed: bool) -> ColumnTransformer:
    #Preprocessing for linear models: scaled numerics + one-hot room_type/city.
    if log_skewed:
        return ColumnTransformer([
            ("skewed", log_then_scale(), SKEWED),
            ("num", StandardScaler(), GEO + NOT_SKEWED),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
        ])
    return ColumnTransformer([
        ("num", StandardScaler(), CONT_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])


def tree_pre(num_cols, bin_cols=()) -> ColumnTransformer:
    #Ordinal-coded room_type/city (+ optional ordinal Low/Medium/High bins) for random forests.
    parts = [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATS)]
    if len(num_cols):
        parts.append(("num", "passthrough", list(num_cols)))
    if len(bin_cols):
        parts.append(("bins", OrdinalEncoder(categories=[TIERS] * len(bin_cols)), list(bin_cols)))
    return ColumnTransformer(parts)


def hgb_frame(frame: pd.DataFrame, cat_levels: dict, cols) -> pd.DataFrame:
    #HistGradientBoosting reads room_type/city as native categoricals.
    out = frame[list(cols) + CAT_FEATS].copy()
    for c in CAT_FEATS:
        out[c] = pd.Categorical(out[c], categories=cat_levels[c])
    return out


def reg_metrics(price_true, pred_log) -> dict:
    pred_usd = np.clip(np.expm1(pred_log), 0, None)
    y_log = np.log1p(price_true)
    return {
        "MAE_usd": mean_absolute_error(price_true, pred_usd),
        "RMSE_usd": np.sqrt(mean_squared_error(price_true, pred_usd)),
        "MedAE_usd": median_absolute_error(price_true, pred_usd),
        "MAE_log": mean_absolute_error(y_log, pred_log),
        "R2_log": r2_score(y_log, pred_log),
        "R2_usd": r2_score(price_true, pred_usd),
    }


def clf_metrics(y_true, pred) -> dict:
    return {"accuracy": accuracy_score(y_true, pred), "macro_F1": f1_score(y_true, pred, average="macro")}


def timed(label, fn):
    t = time.time()
    out = fn()
    print(f"    [{label}: {time.time() - t:.0f}s]", flush=True)
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare regression and classification models for Airbnb price.")
    ap.add_argument("db", nargs="?", default=str(resolve_default_db()), help="path to standalone airbnb.db")
    ap.add_argument("--table", default="Cleaned Data Set", help="SQLite table name to query")
    ap.add_argument("--out", default="model_output", help="folder for figures")
    ap.add_argument("--cv", type=int, default=0, help="k for k-fold cross-validation on the training set (0 = off)")
    ap.add_argument("--tune", action="store_true", help="randomised hyper-parameter search for the HGB regressor")
    ap.add_argument("--sample", type=int, default=0, help="use a stratified random sample of N rows (quick test)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Could not find SQLite database at {db_path.resolve()} (cwd: {Path.cwd()})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Read the selected SQLite table into a DataFrame.
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{args.table}"', conn)

    if "log1p_price" not in df.columns:
        df["log1p_price"] = np.log1p(df["price"])

    if args.sample:
        if args.sample > len(df):
            raise ValueError(f"--sample ({args.sample:,}) exceeds the table size ({len(df):,}).")
        df, _ = train_test_split(df, train_size=args.sample, random_state=SEED, stratify=df["price_bin"])
        df = df.reset_index(drop=True)
    print(f'Loaded {len(df):,} rows from {db_path.resolve()} (table: "{args.table}")')

    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=df["price_bin"])
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    y_tr_reg, y_tr_clf = tr["log1p_price"], tr["price_bin"]
    y_te_clf, price_te = te["price_bin"], te["price"]
    print(f"Train {len(tr):,} / test {len(te):,}; test price SD = ${price_te.std():,.1f}")

    cat_levels = {c: tr[c].astype("category").cat.categories for c in CAT_FEATS}
    hgb_cat_idx = [len(CONT_FEATS) + i for i in range(len(CAT_FEATS))]
    Xtr_h, Xte_h = hgb_frame(tr, cat_levels, CONT_FEATS), hgb_frame(te, cat_levels, CONT_FEATS)

    # price_bin boundaries taken from the data, so regress-then-bucket matches the labels exactly
    cut_low = df.loc[df["price_bin"] == "Low", "price"].max()
    cut_mid = df.loc[df["price_bin"] == "Medium", "price"].max()
    print(f"price_bin boundaries: Low <= ${cut_low:,.0f}, Medium <= ${cut_mid:,.0f}, High above")

    # ======================= REGRESSION: target = log1p(price) =======================
    print("\n=== Regression (target log1p(price); metrics back-transformed to dollars) ===")
    X_cols = CONT_FEATS + CAT_FEATS
    reg_models = {
        "Baseline: predict median": DummyRegressor(strategy="median"),
        "Ridge (raw predictors)": Pipeline([("pre", linear_pre(False)), ("m", Ridge(alpha=1.0))]),
        "Ridge (log predictors)": Pipeline([("pre", linear_pre(True)), ("m", Ridge(alpha=1.0))]),
        "RandomForestRegressor": Pipeline([
            ("pre", tree_pre(CONT_FEATS)),
            ("m", RandomForestRegressor(n_estimators=150, max_depth=20, random_state=SEED, n_jobs=-1))]),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            categorical_features=hgb_cat_idx, random_state=SEED),
    }
    reg_rows, reg_pred_log = [], {}
    for name, model in list(reg_models.items()):
        Xa, Xb = (Xtr_h, Xte_h) if name.startswith("HistGradient") else (tr[X_cols], te[X_cols])
        timed(name, lambda: model.fit(Xa, y_tr_reg))
        reg_pred_log[name] = model.predict(Xb)
        reg_rows.append({"model": name, **reg_metrics(price_te, reg_pred_log[name])})

    if args.tune:
        print("  tuning HistGradientBoostingRegressor (randomised search, 3-fold CV on the training set)...")
        search = RandomizedSearchCV(
            HistGradientBoostingRegressor(categorical_features=hgb_cat_idx, random_state=SEED),
            {"learning_rate": [0.03, 0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63, 127],
             "l2_regularization": [0, 0.1, 1, 10], "min_samples_leaf": [20, 50, 100], "max_iter": [200, 400]},
            n_iter=10, cv=3, scoring="r2", random_state=SEED, n_jobs=1)
        timed("tuning", lambda: search.fit(Xtr_h, y_tr_reg))
        print("  best parameters:", search.best_params_)
        name = "HistGradientBoostingRegressor (tuned)"
        reg_pred_log[name] = search.best_estimator_.predict(Xte_h)
        reg_rows.append({"model": name, **reg_metrics(price_te, reg_pred_log[name])})
        reg_models[name] = search.best_estimator_

    reg_table = pd.DataFrame(reg_rows).set_index("model")
    print(reg_table.round(3).to_string())
    save_table_to_db(reg_table.round(4).reset_index(), db_path, "Regression Model")

    # ===================== CLASSIFICATION: target = price_bin =====================
    print("\n=== Classification (target price_bin), continuous + categorical features ===")
    clf_models = {
        "Baseline: majority class": (DummyClassifier(strategy="most_frequent"), "frame"),
        "LogisticRegression (raw predictors)": (
            Pipeline([("pre", linear_pre(False)), ("m", LogisticRegression(max_iter=2000))]), "frame"),
        "LogisticRegression (log predictors)": (
            Pipeline([("pre", linear_pre(True)), ("m", LogisticRegression(max_iter=2000))]), "frame"),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier(categorical_features=hgb_cat_idx, random_state=SEED), "hgb"),
        "RandomForest (continuous factors)": (Pipeline([
            ("pre", tree_pre(CONT_FEATS)),
            ("m", RandomForestClassifier(n_estimators=150, max_depth=20, random_state=SEED, n_jobs=-1))]), "frame"),
    }
    clf_rows, clf_pred = [], {}
    for name, (model, kind) in list(clf_models.items()):
        Xa, Xb = (Xtr_h, Xte_h) if kind == "hgb" else (tr[X_cols], te[X_cols])
        timed(name, lambda: model.fit(Xa, y_tr_clf))
        clf_pred[name] = model.predict(Xb)
        clf_rows.append({"model": name, **clf_metrics(y_te_clf, clf_pred[name])})
        if name.startswith("RandomForest"):
            clf_models[name] = (clone(model), kind)
            del model
            gc.collect()

    # regress-then-bucket: turn each regression model's predicted price into a tier
    bucket = lambda p: np.where(p <= cut_low, "Low", np.where(p <= cut_mid, "Medium", "High"))
    for name in ["Ridge (log predictors)", "RandomForestRegressor", "HistGradientBoostingRegressor"]:
        pred = bucket(np.expm1(reg_pred_log[name]))
        clf_rows.append({"model": f"{name} -> bucket", **clf_metrics(y_te_clf, pred)})
    clf_table = pd.DataFrame(clf_rows).set_index("model")
    print(clf_table.round(4).to_string())
    save_table_to_db(clf_table.round(4).reset_index(), db_path, "Classification Model")

    # ===== controlled test: continuous vs binned factors, with and without lat/lon =====
    print("\n=== Controlled 2 x 2 test: RandomForest, continuous vs binned factors, with/without lat/lon ===")
    grid = {}
    for factors, num_cols, bin_cols in [("continuous", CONT5, ()), ("binned", (), BIN_FEATS)]:
        for geo_label, geo_cols in [("with lat/lon", GEO), ("without lat/lon", [])]:
            if factors == "continuous" and geo_label == "with lat/lon":     # identical to the main forest above
                grid[(factors, geo_label)] = clf_metrics(y_te_clf, clf_pred["RandomForest (continuous factors)"])
                continue
            pipe = Pipeline([
                ("pre", tree_pre(list(geo_cols) + list(num_cols), bin_cols)),
                ("m", RandomForestClassifier(n_estimators=150, max_depth=20, random_state=SEED, n_jobs=-1))])
            cols = CAT_FEATS + list(num_cols) + list(geo_cols) + list(bin_cols)
            timed(f"{factors}, {geo_label}", lambda: pipe.fit(tr[cols], y_tr_clf))
            grid[(factors, geo_label)] = clf_metrics(y_te_clf, pipe.predict(te[cols]))
            del pipe
            gc.collect()
    two_by_two = pd.DataFrame({
        geo_label: {f: grid[(f, geo_label)]["accuracy"] for f in ["continuous", "binned"]}
        for geo_label in ["with lat/lon", "without lat/lon"]}).round(4)
    print(two_by_two.to_string())
    two_by_two.index.name = "five_factors_accuracy"
    save_table_to_db(two_by_two.reset_index(), db_path, "Binning_2x2 Model")
    for g in ["with lat/lon", "without lat/lon"]:
        d = grid[("continuous", g)]["accuracy"] - grid[("binned", g)]["accuracy"]
        print(f"  cost of binning the five factors, {g}: {100 * d:.1f} accuracy points")
    d = grid[("continuous", "with lat/lon")]["accuracy"] - grid[("continuous", "without lat/lon")]["accuracy"]
    print(f"  cost of dropping latitude/longitude (continuous factors): {100 * d:.1f} accuracy points")

    # ============================ optional cross-validation ============================
    if args.cv and args.cv > 1:
        print(f"\n=== {args.cv}-fold cross-validation on the training set (mean +/- sd) ===")
        kf = KFold(args.cv, shuffle=True, random_state=SEED)
        skf = StratifiedKFold(args.cv, shuffle=True, random_state=SEED)
        cv_rows = []
        for name in ["Ridge (log predictors)", "RandomForestRegressor", "HistGradientBoostingRegressor"]:
            Xa = Xtr_h if name.startswith("HistGradient") else tr[X_cols]
            s = cross_val_score(reg_models[name], Xa, y_tr_reg, cv=kf, scoring="r2")
            cv_rows.append({"model": name, "metric": "R2 (log scale)", "mean": s.mean(), "sd": s.std()})
        for name in ["LogisticRegression (log predictors)", "HistGradientBoostingClassifier",
                     "RandomForest (continuous factors)"]:
            model, kind = clf_models[name]
            Xa = Xtr_h if kind == "hgb" else tr[X_cols]
            s = cross_val_score(model, Xa, y_tr_clf, cv=skf, scoring="accuracy")
            cv_rows.append({"model": name, "metric": "accuracy", "mean": s.mean(), "sd": s.std()})
        cv_table = pd.DataFrame(cv_rows)
        print(cv_table.round(4).to_string(index=False))
        save_table_to_db(cv_table.round(4), db_path, "model_4_4_cross_validation")

    # ================================== figures ==================================
    best_reg = reg_table.drop(index="Baseline: predict median")["R2_log"].idxmax()
    print(f"\nBest regression model on the test set (R2, log scale): {best_reg}")
    price_ticks = [10, 25, 50, 100, 250, 500, 1000, 5000]
    fmt = FuncFormatter(lambda v, _: f"${np.expm1(v):,.0f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(te["log1p_price"], reg_pred_log[best_reg], gridsize=70, bins="log", cmap="Blues", mincnt=1,
                   extent=(np.log1p(5), np.log1p(5000), np.log1p(5), np.log1p(5000)))
    lim = [np.log1p(5), np.log1p(5000)]
    ax.plot(lim, lim, color="#E8590C", lw=1.5, label="perfect prediction")
    ax.set_xticks(np.log1p(price_ticks)); ax.set_yticks(np.log1p(price_ticks))
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.set(xlabel="Actual price (log scale)", ylabel="Predicted price (log scale)",
           title=f"{best_reg}: predicted vs actual (test set)")
    fig.colorbar(hb, label="listings (log colour scale)")
    ax.legend(loc="upper left")
    fig.savefig(out / "fig_4_1_pred_vs_actual.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nPermutation importance ({best_reg}; drop in R2 on the log scale; up to 15,000 test rows)...")
    best_model = reg_models[best_reg]
    X_imp = Xte_h if "HistGradient" in best_reg else te[X_cols]
    sub = X_imp.sample(min(15000, len(X_imp)), random_state=SEED)
    pi = permutation_importance(best_model, sub, te.loc[sub.index, "log1p_price"],
                                n_repeats=5, random_state=SEED, scoring="r2")
    imp = pd.DataFrame({"feature": sub.columns, "mean": pi.importances_mean, "sd": pi.importances_std}) \
        .sort_values("mean")
    imp_sorted = imp.sort_values("mean", ascending=False).round(4)
    save_table_to_db(imp_sorted, db_path, "Permutation Importance Model")
    print(imp_sorted.to_string(index=False))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(imp["feature"], imp["mean"], xerr=imp["sd"], color="#2F6FDE")
    ax.set(xlabel="Drop in R2 (log scale) when the feature is shuffled",
           title=f"Permutation importance: {best_reg}")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig_4_2_permutation_importance.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    name = "RandomForest (continuous factors)"
    cm = confusion_matrix(y_te_clf, clf_pred[name], labels=TIERS)
    rec = cm / cm.sum(axis=1, keepdims=True)
    print("\nPer-tier recall,", name, dict(zip(TIERS, rec.diagonal().round(3))))
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(rec, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({100 * rec[i, j]:.0f}%)", ha="center", va="center",
                    color="white" if rec[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(TIERS); ax.set_yticks(range(3)); ax.set_yticklabels(TIERS)
    ax.set(xlabel="Predicted tier", ylabel="Actual tier", title=f"{name}\nconfusion matrix (rows sum to 100%)")
    ax.grid(False)
    fig.colorbar(im, label="share of the actual tier")
    fig.savefig(out / "fig_4_3_confusion_matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with sqlite3.connect(db_path) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print(f"\nDone. Tables in {db_path.resolve()}: {tables}")
    print(f"Figures saved to {out.resolve()}")


if __name__ == "__main__":
    main()