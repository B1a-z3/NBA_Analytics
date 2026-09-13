"""
Runs each SQL analytics file in sql/analytics/ and prints a preview of the
result set. Handy for sanity-checking the warehouse and for generating the
real numbers to replace the "(example)" findings documented in each .sql file.

Run:
    python scripts/run_analytics.py
    python scripts/run_analytics.py --file 01_rolling_team_win_pct.sql
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

ANALYTICS_DIR = Path(__file__).resolve().parents[1] / "sql" / "analytics"


def run_file(engine, path: Path):
    sql = path.read_text()

    # Strip comment-only lines FIRST (every file here opens with a large
    # documentation header, so a naive "does this chunk start with --"
    # check -- the previous approach -- discards the whole file's SQL).
    # What's left is split on ';' to get each top-level statement (query 5
    # has two). Each statement is wrapped in sqlalchemy.text() before being
    # handed to pd.read_sql: passing a bare Python string straight through
    # hits a real pandas/SQLAlchemy/psycopg2 incompatibility whenever the
    # text contains a literal '%' (e.g. this project's queries alias columns
    # like "TS%") -- psycopg2's %-style parameter substitution misfires on
    # the stray '%' and raises "immutabledict is not a sequence", even
    # though the query has no bind parameters at all. text() sidesteps it.
    code_only = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    statements = [s.strip() for s in code_only.split(";") if s.strip()]

    print(f"\n{'=' * 80}\n{path.name}\n{'=' * 80}")
    for i, stmt in enumerate(statements):
        try:
            df = pd.read_sql(text(stmt), engine)
            print(f"\n--- statement {i + 1} ---")
            print(df.head(15).to_string(index=False))
        except Exception as exc:
            print(f"  [skipped statement {i + 1}: {exc}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Run only this file (e.g. 01_rolling_team_win_pct.sql)")
    args = parser.parse_args()

    engine = get_engine()
    files = sorted(ANALYTICS_DIR.glob("*.sql"))
    if args.file:
        files = [f for f in files if f.name == args.file]

    for f in files:
        run_file(engine, f)


if __name__ == "__main__":
    main()
