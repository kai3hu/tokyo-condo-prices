-- name: yearly_trend
-- Transactions and mean price per m2 by year, 23 wards vs. Tama area.
SELECT year,
       CASE WHEN ward23 = 1 THEN '23 wards' ELSE 'Tama' END AS area,
       COUNT(*)                         AS n,
       ROUND(AVG(price_per_m2) / 1e4, 1) AS mean_man_yen_per_m2
FROM transactions
GROUP BY year, area
ORDER BY area, year;

-- name: ward_growth
-- Mean price per m2 in 2015 vs. 2022 for each of the 23 wards, and the growth between them.
WITH by_year AS (
  SELECT city, year, AVG(price_per_m2) AS ppm2, COUNT(*) AS n
  FROM transactions
  WHERE ward23 = 1 AND year IN (2015, 2022)
  GROUP BY city, year
)
SELECT a.city,
       ROUND(a.ppm2 / 1e4, 1)              AS ppm2_2015_man,
       ROUND(b.ppm2 / 1e4, 1)              AS ppm2_2022_man,
       ROUND(100.0 * (b.ppm2 / a.ppm2 - 1), 1) AS growth_pct
FROM by_year a JOIN by_year b ON a.city = b.city AND a.year = 2015 AND b.year = 2022
ORDER BY growth_pct DESC;

-- name: station_walk
-- How price per m2 falls with walking time to the nearest station (23 wards).
SELECT CASE WHEN station_min <= 5  THEN '0-5'
            WHEN station_min <= 10 THEN '6-10'
            WHEN station_min <= 15 THEN '11-15'
            ELSE '16+' END              AS walk_min,
       COUNT(*)                          AS n,
       ROUND(AVG(price_per_m2) / 1e4, 1) AS mean_man_yen_per_m2
FROM transactions
WHERE ward23 = 1
GROUP BY walk_min
ORDER BY MIN(station_min);

-- name: top_stations
-- Busiest stations by number of resale transactions, with price level.
SELECT station, COUNT(*) AS n, ROUND(AVG(price_per_m2) / 1e4, 1) AS mean_man_yen_per_m2
FROM transactions
WHERE station IS NOT NULL
GROUP BY station
HAVING n >= 300
ORDER BY n DESC
LIMIT 10;
