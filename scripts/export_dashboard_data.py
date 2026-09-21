"""
One-off export: snapshots everything the Streamlit dashboard needs out of a
local Postgres warehouse into small, committed files under dashboard/data/.
The deployed dashboard reads only these files -- no database required.

Needs a local Postgres populated by the pipeline (DATABASE_URL / .env).

Run:
    python scripts/export_dashboard_data.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from config.db import get_engine  # noqa: E402

OUT = ROOT / "dashboard" / "data"

QUERIES = {
    "team_features": "SELECT * FROM team_game_features",
    "player_features": """
        SELECT player_id, full_name, game_date, rolling_ts_pct,
               season_to_date_ts_pct, rolling_load_index, age_years
        FROM player_game_features
    """,
    "monitoring_log": """
        SELECT mv.model_name, mv.version_tag, mml.week_start, mml.n_predictions,
               mml.accuracy, mml.log_loss, mml.brier_score
        FROM model_monitoring_log mml
        JOIN model_versions mv ON mv.model_version_id = mml.model_version_id
        ORDER BY mml.week_start
    """,
    "teams": "SELECT team_id, abbreviation, name FROM teams",
    "home_away_edge": """
        WITH home_results AS (
            SELECT home_team_id AS team_id, COUNT(*) home_games,
                   COUNT(*) FILTER (WHERE home_score > away_score) home_wins
            FROM games WHERE home_score IS NOT NULL GROUP BY home_team_id
        ),
        away_results AS (
            SELECT away_team_id AS team_id, COUNT(*) away_games,
                   COUNT(*) FILTER (WHERE away_score > home_score) away_wins
            FROM games WHERE home_score IS NOT NULL GROUP BY away_team_id
        )
        SELECT t.abbreviation,
               ROUND((h.home_wins::numeric / h.home_games
                      - a.away_wins::numeric / a.away_games), 3)::float AS home_court_edge
        FROM teams t
        JOIN home_results h ON h.team_id = t.team_id
        JOIN away_results a ON a.team_id = t.team_id
        ORDER BY home_court_edge DESC
    """,
    "load_vs_performance": """
        WITH ordered AS (
            SELECT player_id, game_date, load_index, ts_pct,
                   LEAD(ts_pct) OVER (PARTITION BY player_id ORDER BY game_date) AS next_ts_pct,
                   NTILE(4) OVER (ORDER BY load_index) AS load_quartile
            FROM player_game_features
            WHERE load_index IS NOT NULL
        )
        SELECT load_quartile, COUNT(*) AS n,
               ROUND(AVG(next_ts_pct)::numeric, 3)::float AS avg_next_game_ts_pct
        FROM ordered
        WHERE next_ts_pct IS NOT NULL
        GROUP BY load_quartile
        ORDER BY load_quartile
    """,
    "back_to_back_fatigue": "SELECT * FROM dashboard_back_to_back_fatigue ORDER BY sort_key",
    "aging_curve": "SELECT * FROM dashboard_aging_curve ORDER BY age_at_season_start",
}


def export_feature_importances():
    path = ROOT / "models" / "artifacts" / "player_decline.joblib"
    if not path.exists():
        print("skip decline_feature_importances (no trained model found)")
        return
    bundle = joblib.load(path)
    df = pd.DataFrame({
        "feature": bundle["features"],
        "importance": bundle["model"].feature_importances_,
    })
    df.to_parquet(OUT / "decline_feature_importances.parquet", index=False)
    print(f"decline_feature_importances: {df.shape}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    for name, sql in QUERIES.items():
        df = pd.read_sql(sql, engine)
        df.to_parquet(OUT / f"{name}.parquet", index=False, compression="zstd")
        size_kb = (OUT / f"{name}.parquet").stat().st_size / 1024
        print(f"{name}: {df.shape}, {size_kb:,.0f} KB")
    export_feature_importances()


if __name__ == "__main__":
    main()
