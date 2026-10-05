"""Regression, classification and clustering on the cleaned Tokyo resale-condo data."""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score, roc_auc_score,
                             silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from names import EN

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
SEED = 0

NUM = ["area_m2", "station_min", "age_years", "rooms", "has_L", "has_D", "has_K", "has_S",
       "coverage_pct", "far_pct", "t", "ward23", "renovated"]
CAT = ["city_code", "structure", "zoning"]


def load():
    with sqlite3.connect(ROOT / "data" / "tokyo_condos.db") as con:
        df = pd.read_sql("SELECT * FROM transactions", con)
    df["zoning"] = df["zoning"].fillna("unknown")
    df["renovated"] = df["renovated"].fillna(-1)          # -1 = not reported
    return df


def linear_pre():
    num = make_pipeline(
        __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median"),
        StandardScaler())
    return ColumnTransformer([("num", num, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def tree_pre():
    return ColumnTransformer([("num", "passthrough", NUM),
                              ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)])


def regression(train, test):
    """Predict log(total price). Time split: fit on 2015-2021, evaluate on 2022."""
    ytr, yte = np.log(train["price_jpy"]), np.log(test["price_jpy"])
    models = {
        "ridge": make_pipeline(linear_pre(), Ridge(alpha=1.0)),
        "hist_gbm": make_pipeline(tree_pre(), HistGradientBoostingRegressor(
            categorical_features=list(range(len(NUM), len(NUM) + len(CAT))),
            max_iter=500, learning_rate=0.08, random_state=SEED)),
    }
    out = {}
    for name, m in models.items():
        m.fit(train[NUM + CAT], ytr)
        pred = m.predict(test[NUM + CAT])
        p, y = np.exp(pred), test["price_jpy"].to_numpy()
        out[name] = {"r2_log": round(r2_score(yte, pred), 3),
                     "mape_pct": round(100 * mean_absolute_percentage_error(y, p), 1),
                     "median_ape_pct": round(100 * float(np.median(np.abs(p - y) / y)), 1),
                     "mae_man_yen": round(mean_absolute_error(y, p) / 1e4, 0)}
        if name == "hist_gbm":
            plt.figure(figsize=(5, 5))
            plt.scatter(y / 1e6, p / 1e6, s=2, alpha=0.15)
            lim = [0, np.percentile(y, 99.5) / 1e6]
            plt.plot(lim, lim, "k--", lw=1)
            plt.xlim(lim); plt.ylim(lim)
            plt.xlabel("Actual price (million JPY)"); plt.ylabel("Predicted (million JPY)")
            plt.title("Gradient boosting, 2022 hold-out")
            plt.tight_layout(); plt.savefig(FIG / "regression_pred_vs_actual.png", dpi=130); plt.close()
    return out


def classification(train, test):
    """Is a unit priced above its ward's median price per m2 for that year?"""
    out = {}
    models = {
        "logistic": make_pipeline(linear_pre(), LogisticRegression(max_iter=2000)),
        "hist_gbm": make_pipeline(tree_pre(), HistGradientBoostingClassifier(
            categorical_features=list(range(len(NUM), len(NUM) + len(CAT))),
            max_iter=400, learning_rate=0.08, random_state=SEED)),
    }
    for name, m in models.items():
        m.fit(train[NUM + CAT], train["premium"])
        prob = m.predict_proba(test[NUM + CAT])[:, 1]
        pred = (prob >= 0.5).astype(int)
        out[name] = {"roc_auc": round(roc_auc_score(test["premium"], prob), 3),
                     "accuracy": round(accuracy_score(test["premium"], pred), 3),
                     "f1": round(f1_score(test["premium"], pred), 3)}
    out["base_rate_test"] = round(float(test["premium"].mean()), 3)
    return out


def clustering(df):
    """Group municipalities by their resale-market profile."""
    g = df.groupby("city_code")
    prof = pd.DataFrame({
        "n": g.size(),
        "ppm2_2022": df[df.year == 2022].groupby("city_code")["price_per_m2"].mean(),
        "ppm2_2015": df[df.year == 2015].groupby("city_code")["price_per_m2"].mean(),
        "median_age": g["age_years"].median(),
        "median_walk_min": g["station_min"].median(),
        "median_area": g["area_m2"].median(),
        "share_renovated": g["renovated"].apply(lambda s: (s == 1).sum() / max((s >= 0).sum(), 1)),
    })
    prof = prof[prof["n"] >= 300].dropna()
    prof["growth_pct"] = 100 * (prof["ppm2_2022"] / prof["ppm2_2015"] - 1)
    feats = ["ppm2_2022", "growth_pct", "median_age", "median_walk_min", "median_area", "share_renovated"]
    X = StandardScaler().fit_transform(prof[feats])
    sil = {k: silhouette_score(X, KMeans(k, n_init=20, random_state=SEED).fit_predict(X)) for k in range(2, 8)}
    k = max(sil, key=sil.get)
    prof["name"] = prof.index.map(EN)
    # k=4 is the runner-up by silhouette and splits the market more usefully; save it too.
    prof["cluster_k4"] = KMeans(4, n_init=20, random_state=SEED).fit_predict(X)
    k4 = prof.groupby("cluster_k4").agg(members=("name", lambda s: ", ".join(sorted(s))),
        ppm2_2022_man=("ppm2_2022", lambda s: round(s.mean() / 1e4, 1)),
        growth_pct=("growth_pct", lambda s: round(s.mean(), 1)),
        median_age=("median_age", "mean"), median_walk_min=("median_walk_min", "mean"),
        median_area=("median_area", "mean")).round(1)
    k4.to_csv(RES / "clusters_k4.csv")
    prof["cluster"] = KMeans(k, n_init=20, random_state=SEED).fit_predict(X)
    summary = prof.groupby("cluster").agg(
        members=("name", lambda s: ", ".join(sorted(s))),
        ppm2_2022_man=("ppm2_2022", lambda s: round(s.mean() / 1e4, 1)),
        growth_pct=("growth_pct", lambda s: round(s.mean(), 1)),
        median_age=("median_age", "mean"), median_walk_min=("median_walk_min", "mean"),
        median_area=("median_area", "mean")).round(1)
    prof.round(3).to_csv(RES / "municipality_profiles.csv")
    summary.to_csv(RES / "clusters.csv")
    xy = PCA(2, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(7, 5))
    for c in sorted(prof["cluster"].unique()):
        m = prof["cluster"].to_numpy() == c
        plt.scatter(xy[m, 0], xy[m, 1], label=f"cluster {c}", s=30)
    for (x, y), n in zip(xy, prof["name"]):
        plt.annotate(n, (x, y), fontsize=6, alpha=0.8)
    plt.legend(fontsize=8); plt.title(f"Tokyo municipalities, k-means (k={k}), PCA projection")
    plt.tight_layout(); plt.savefig(FIG / "clusters_pca.png", dpi=130); plt.close()
    return {"k": k, "silhouette_by_k": {kk: round(v, 3) for kk, v in sil.items()},
            "n_municipalities": len(prof), "clusters": summary.reset_index().to_dict("records"),
            "clusters_k4": k4.reset_index().to_dict("records")}


def main():
    df = load()
    med = df.groupby(["city_code", "year"])["price_per_m2"].transform("median")
    df["premium"] = (df["price_per_m2"] > med).astype(int)
    train, test = df[df.year <= 2021], df[df.year == 2022]
    results = {"rows": len(df), "train_rows": len(train), "test_rows_2022": len(test),
               "regression": regression(train, test),
               "classification": classification(train, test),
               "clustering": clustering(df)}
    (RES / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
