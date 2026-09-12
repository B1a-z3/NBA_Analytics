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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

ANALYTICS_DIR = Path(__file__).resolve().parents[1] / "sql" / "analytics"


def run_file(engine, path: Path):
    sql = path.read_text()
    # each file may contain more than one statement (e.g. query 5 has two);
    # split on blank-line-separated top-level statements naively by ';\n\n'
    statements = [s.strip() for s in sql.split(";\n\n") if s.strip() and not s.strip().startswith("--")]
    print(f"\n{'=' * 80}\n{path.name}\n{'=' * 80}")
    for i, stmt in enumerate(statements):
        clean = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--"))
        if not clean.strip():
            continue
        try:
            df = pd.read_sql(clean, engine)
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
