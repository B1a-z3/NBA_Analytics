"""
Precomputes small, dashboard-sized summary tables from the full warehouse,
so the Streamlit dashboard's "League Analytics" tab doesn't have to query
the raw player_game_stats table (324MB+ and growing) live on every page
load. This also keeps the deployed/cloud copy of the database small: a
free-tier Postgres host (Neon, Supabase, etc. typically cap around 500MB)
can hold the dashboard's actual footprint -- teams, games, team_game_features,
player_game_features, monitoring log, and these two tiny summary tables --
without ever needing the full raw box-score table at all.

Produces:
    dashboard_back_to_back_fatigue  (3 rows)  -- sql/analytics/04, chart version
    dashboard_aging_curve           (~21 rows) -- sql/analytics/06, chart version

Run:
    python features/build_dashboard_summaries.py
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402


def build_back_to_back_fatigue(engine) -> pd.DataFrame:
    return pd.read_sql("""
        WITH team_shooting AS (
            SELECT game_id, team_id,
                   SUM(fgm)::numeric / NULLIF(SUM(fga), 0) AS team_fg_pct
            FROM player_game_stats
            GROUP BY game_id, team_id
        )
        SELECT
            CASE
                WHEN tgf.days_rest = 1 THEN 'Back-to-back (1 day rest)'
                WHEN tgf.days_rest = 2 THEN '2 days rest'
                ELSE '3+ days rest'
            END AS rest_bucket,
            MIN(tgf.days_rest) AS sort_key,
            COUNT(*) AS n_games,
            ROUND(AVG(ts.team_fg_pct)::numeric, 4) AS avg_team_fg_pct
        FROM team_game_features tgf
        JOIN team_shooting ts ON ts.game_id = tgf.game_id AND ts.team_id = tgf.team_id
        WHERE tgf.days_rest IS NOT NULL
        GROUP BY rest_bucket
        ORDER BY sort_key
    """, engine)


def build_aging_curve(engine) -> pd.DataFrame:
    return pd.read_sql("""
        WITH player_season_stats AS (
            SELECT pgs.player_id, g.season,
                   DATE_PART('year', AGE(
                       MAKE_DATE(SPLIT_PART(g.season, '-', 1)::int, 10, 1), p.birth_date
                   ))::int AS age_at_season_start,
                   AVG(pgs.points) AS avg_pts,
                   AVG(CASE WHEN (pgs.fga + 0.44 * pgs.fta) = 0 THEN NULL
                            ELSE pgs.points::numeric / (2 * (pgs.fga + 0.44 * pgs.fta)) END) AS avg_ts_pct
            FROM player_game_stats pgs
            JOIN games g ON g.game_id = pgs.game_id
            JOIN players p ON p.player_id = pgs.player_id
            WHERE p.birth_date IS NOT NULL
            GROUP BY pgs.player_id, g.season, p.birth_date
        )
        SELECT age_at_season_start,
               COUNT(DISTINCT player_id) AS n_players,
               ROUND(AVG(avg_pts)::numeric, 2) AS league_avg_pts_at_age,
               ROUND(AVG(avg_ts_pct)::numeric, 3) AS league_avg_ts_pct_at_age
        FROM player_season_stats
        WHERE age_at_season_start BETWEEN 20 AND 40
        GROUP BY age_at_season_start
        ORDER BY age_at_season_start
    """, engine)


def materialize(engine, name: str, df: pd.DataFrame):
    df.to_sql(name, engine, if_exists="replace", index=False)


def main():
    engine = get_engine()

    b2b_df = build_back_to_back_fatigue(engine)
    materialize(engine, "dashboard_back_to_back_fatigue", b2b_df)
    print(f"Materialized dashboard_back_to_back_fatigue: {b2b_df.shape}")

    aging_df = build_aging_curve(engine)
    materialize(engine, "dashboard_aging_curve", aging_df)
    print(f"Materialized dashboard_aging_curve: {aging_df.shape}")


if __name__ == "__main__":
    main()
