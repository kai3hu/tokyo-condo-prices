# Tokyo Resale-Condo Prices, 2015–2022

This project analyzes 130,942 real resale-condo transactions in Tokyo. The data comes from Japan's Ministry of Land, Infrastructure, Transport and Tourism (MLIT). The analysis cleans the raw government CSV, loads it into SQLite, explores it with SQL, and then fits three kinds of scikit-learn models: regression, classification and clustering.

**Data source:** 国土交通省 不動産情報ライブラリ, 不動産取引価格情報. Settings used: Tokyo (東京都), 中古マンション等 (resale condominiums), transactions from 2015 Q1 to 2022 Q4.
Source: MLIT Real Estate Information Library, https://www.reinfolib.mlit.go.jp/

## Pipeline

| Step | File | What it does |
|---|---|---|
| Clean | `src/clean.py` | Reads the Shift-JIS CSV with pandas and parses the messy fields (below). It drops non-arm's-length and non-residential sales and trims price-per-m² outliers. The result goes into `data/tokyo_condos.db` (SQLite). |
| Explore | `sql/queries.sql`, `src/sql_report.py` | Runs named SQL queries: yearly price trend, 2015→2022 growth by ward, the station-distance premium, and the busiest stations. |
| Model | `src/models.py` | Regression on log price; classification of whether a unit is above its ward-year median; k-means clustering of municipalities. Results go to `results/metrics.json`. |

Run it with:

```bash
pip install pandas scikit-learn matplotlib
# Put Tokyo_20151_20224.csv from the MLIT download page into data/
python src/clean.py && python src/sql_report.py && python src/models.py
```

## Data cleaning

The raw file has 130,942 rows; 125,048 are kept after cleaning. The fields that needed work:

| Field | Problem | Handling |
|---|---|---|
| Walk to station | 946 values are ranges (`30分～60分`, `1H～1H30`, `2H～`); 98 are missing | Ranges are mapped to their midpoints; missing values are dropped |
| Floor area | Stored as text, with a top-coded `2,000㎡以上` | Parsed to a number; top-coded rows are dropped and the area is limited to 10–300 m² |
| Year built | 17 rows say `戦前` (pre-war); 3,353 are missing | Pre-war is set to 1945 and flagged; age is computed as transaction year minus build year |
| Layout | Full-width text such as `３ＬＤＫ＋Ｓ`; 882 rows are non-standard (studio, maisonette, open floor) | Converted to a room count plus L / D / K / S flags |
| Transaction period | `2022年第4四半期` | Parsed to year, quarter and a continuous time index |
| Special circumstances | 1,059 sales are auctions, related-party sales, etc. | Dropped, because they are not market prices |
| Use | 1,089 sales are offices or shops; 23,717 rows have no use listed | Non-residential rows are dropped; rows with no use listed are treated as residential |
| Renovated | 17,476 rows have no value | Coded as −1 (not reported) rather than imputed |

## Findings from SQL

- In the 23 wards, the mean price per m² rose every year, from **¥759k in 2015 to ¥1.00M in 2022 (+32%)**. In the Tama area it went from ¥403k to ¥529k (+31%), with most of that gain coming in 2021–2022.
- The cheapest wards in 2015 grew fastest: **Katsushika +52%, Arakawa +49%, Adachi +47%**. The wards that were already expensive grew about 26–35%.
- Price per m² falls with walking time to the station: **¥940k at 0–5 minutes, ¥888k at 6–10, ¥736k at 11–15, and ¥553k at 16 minutes or more** (23 wards).

## Models

All models are trained on 2015–2021 (108,296 sales) and tested on **2022 (16,752 sales)**. This is a time-based hold-out, so a model never sees the future.

**Regression: predicting the sale price** (features: ward, area, age, walk time, layout, structure, zoning, floor-area ratio, time)

| Model | R² (log price) | MAPE | Median abs. % error |
|---|---|---|---|
| Ridge regression (one-hot encoded wards) | 0.840 | 20.2% | 14.5% |
| Histogram gradient boosting | **0.891** | **16.0%** | **11.7%** |

**Classification: is a unit priced above its ward's median price per m² for that year?** (49% of test units are positive)

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Logistic regression | 0.851 | 0.771 | 0.772 |
| Histogram gradient boosting | **0.894** | **0.812** | **0.812** |

**Clustering: grouping Tokyo's municipalities by market profile.** Each municipality is described by six features: 2022 price per m², growth since 2015, median building age, median walk time, median unit size, and share renovated. The 44 municipalities with at least 300 sales are standardized and clustered with k-means. Silhouette scores for k = 2..7 pick **k = 2** (silhouette 0.418).

- **Central cluster (20 wards):** ¥1.06M per m², small units (median 39 m²), 6-minute walk.
- **Outer cluster (24 municipalities):** ¥543k per m², 63 m² units, 10.5-minute walk.

The useful finding: **Adachi, Katsushika and Edogawa are legally part of the 23 wards, but by market profile they cluster with the Tama suburbs.** Location on the map does not tell you which market a unit is in. The k = 4 solution is saved in `results/clusters_k4.csv`, but it mostly splits off small outlier municipalities.

## Limitations

- MLIT publishes these prices from buyer surveys. They are rounded and anonymized, and the location is coarse: district and nearest station, not an exact address.
- There is no floor number, view, or building-level identifier, which caps how accurate any model can be.
- The classification label is defined relative to the ward-year median, so it measures a unit's position within its own market, not its absolute price level.
