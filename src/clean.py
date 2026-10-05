"""Load the raw MLIT CSV (Shift-JIS), clean it, and write a SQLite database."""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "Tokyo_20151_20224.csv"
DB = ROOT / "data" / "tokyo_condos.db"

COLS = {
    "市区町村コード": "city_code", "市区町村名": "city", "地区名": "district",
    "最寄駅：名称": "station", "最寄駅：距離（分）": "station_min_raw",
    "取引価格（総額）": "price_jpy", "間取り": "layout_raw", "面積（㎡）": "area_raw",
    "建築年": "built_raw", "建物の構造": "structure_raw", "用途": "use",
    "今後の利用目的": "future_use", "都市計画": "zoning", "建ぺい率（％）": "coverage_pct",
    "容積率（％）": "far_pct", "取引時期": "period_raw", "改装": "renovated_raw",
    "取引の事情等": "special_raw",
}
# Station walking time is given in minutes, or as ranges once it exceeds 30 min.
RANGE_MIN = {"30分～60分": 45, "1H～1H30": 75, "1H30～2H": 105, "2H～": 120}
Z2H = str.maketrans("０１２３４５６７８９ＲＫＤＬＳ", "0123456789RKDLS")


def parse_layout(s):
    """'３ＬＤＫ＋Ｓ' -> rooms=3, has_L/D/K/S flags. Non-standard layouts -> NaN rooms."""
    if pd.isna(s):
        return pd.Series([np.nan, 0, 0, 0, 0])
    t = s.translate(Z2H)
    m = re.match(r"(\d+)", t)
    rooms = float(m.group(1)) if m else np.nan
    return pd.Series([rooms, int("L" in t), int("D" in t), int("K" in t), int("S" in t)])


def load_clean():
    df = pd.read_csv(RAW, encoding="cp932", dtype=str).rename(columns=COLS)[list(COLS.values())]
    n0 = len(df)

    df["price_jpy"] = df["price_jpy"].astype(float)
    df["area_m2"] = pd.to_numeric(df["area_raw"].str.replace(",", "").str.replace("㎡以上", ""), errors="coerce")
    df["area_capped"] = df["area_raw"].str.contains("以上", na=False).astype(int)
    df["station_min"] = pd.to_numeric(df["station_min_raw"], errors="coerce").fillna(df["station_min_raw"].map(RANGE_MIN))
    df["built_prewar"] = (df["built_raw"] == "戦前").astype(int)
    df["built_year"] = pd.to_numeric(df["built_raw"].str.replace("年", ""), errors="coerce")
    df.loc[df["built_prewar"] == 1, "built_year"] = 1945
    p = df["period_raw"].str.extract(r"(\d{4})年第(\d)四半期").astype(float)
    df["year"], df["quarter"] = p[0], p[1]
    df["t"] = df["year"] + (df["quarter"] - 1) / 4
    df["age_years"] = df["year"] - df["built_year"]
    df[["rooms", "has_L", "has_D", "has_K", "has_S"]] = df["layout_raw"].apply(parse_layout)
    df["structure"] = df["structure_raw"].fillna("unknown").map(
        lambda s: "SRC" if s.startswith("ＳＲＣ") else "RC" if s.startswith("ＲＣ") else "steel" if "鉄骨" in s else "unknown")
    df["renovated"] = df["renovated_raw"].map({"改装済み": 1, "未改装": 0})
    df["ward23"] = (df["city"].str.endswith("区")).astype(int)
    df["coverage_pct"] = pd.to_numeric(df["coverage_pct"], errors="coerce")
    df["far_pct"] = pd.to_numeric(df["far_pct"], errors="coerce")

    # Filters: residential units only, arm's-length sales only, sane sizes/prices.
    keep = (
        df["use"].fillna("住宅").eq("住宅")
        & df["special_raw"].isna()
        & df["area_capped"].eq(0)
        & df["area_m2"].between(10, 300)
        & df["built_year"].notna()
        & df["age_years"].ge(0)
        & df["station_min"].notna()
    )
    df = df[keep].copy()
    df["price_per_m2"] = df["price_jpy"] / df["area_m2"]
    lo, hi = df["price_per_m2"].quantile([0.001, 0.999])
    df = df[df["price_per_m2"].between(lo, hi)]
    report = {"raw_rows": n0, "clean_rows": len(df)}
    return df, report


def main():
    df, report = load_clean()
    out = df.drop(columns=[c for c in df.columns if c.endswith("_raw")])
    with sqlite3.connect(DB) as con:
        out.to_sql("transactions", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_city_year ON transactions(city, year)")
    print(report)


if __name__ == "__main__":
    main()
