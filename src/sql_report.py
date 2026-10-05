"""Run every named query in sql/queries.sql against the SQLite DB and save CSVs."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    text = (ROOT / "sql" / "queries.sql").read_text()
    blocks = re.split(r"^-- name: (\w+)\n", text, flags=re.M)[1:]
    with sqlite3.connect(ROOT / "data" / "tokyo_condos.db") as con:
        for name, sql in zip(blocks[::2], blocks[1::2]):
            df = pd.read_sql(sql, con)
            df.to_csv(ROOT / "results" / f"sql_{name}.csv", index=False)
            print(f"\n== {name}\n{df.to_string(index=False)}")

if __name__ == "__main__":
    main()
