"""
Live/current-season puller using nba_api. By default pulls games from the
last N days (simulating a nightly cron during the season), plus per-player
box scores and tracking stats for those games, and upserts into Postgres.
An explicit --start-date/--end-date range can be given instead, to backfill
any past date range (e.g. June of a prior season) through this same path --
useful because player_tracking_stats (distance run, speed, touches, drives)
is ONLY available via nba_api, not in the Kaggle bulk CSVs, so this is how
you'd fill in tracking data for historical games already loaded by
load_kaggle_backfill.py.

Run:
    python ingestion/pull_nba_api_live.py --days 3
    python ingestion/pull_nba_api_live.py --start-date 2024-06-01 --end-date 2024-06-30

This mirrors the schema/validation used in load_kaggle_backfill.py so both
sources land in the same tables.
"""
import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402
import ingestion.nba_api_tls_patch  # noqa: E402,F401 -- must precede nba_api.stats.endpoints imports
from ingestion.validation import (  # noqa: E402
    validate_games,
    validate_player_game_stats,
    validate_tracking_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nba_api_live")

REQUEST_DELAY_SECONDS = 0.6  # be polite to the (unofficial) stats.nba.com endpoints


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_scoreboard(game_date: date):
    from nba_api.stats.endpoints import scoreboardv2
    return scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y")).get_data_frames()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_boxscore(game_id: str):
    from nba_api.stats.endpoints import boxscoretraditionalv2
    return boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_tracking(game_id: str):
    from nba_api.stats.endpoints import boxscoreplayertrackv2
    return boxscoreplayertrackv2.BoxScorePlayerTrackV2(game_id=game_id).get_data_frames()


def pull_games_for_range(start_date: date, end_date: date) -> pd.DataFrame:
    all_games = []
    d = start_date
    while d <= end_date:
        try:
            frames = _fetch_scoreboard(d)
            game_header = frames[0]
            line_score = frames[1]
            if not game_header.empty:
                all_games.append(_merge_scoreboard(game_header, line_score, d))
        except Exception as exc:
            log.warning("Scoreboard pull failed for %s: %s", d, exc)
        time.sleep(REQUEST_DELAY_SECONDS)
        d += timedelta(days=1)
    if not all_games:
        return pd.DataFrame()
    return pd.concat(all_games, ignore_index=True)


def _merge_scoreboard(game_header: pd.DataFrame, line_score: pd.DataFrame, game_date: date) -> pd.DataFrame:
    scores = line_score.set_index("TEAM_ID")["PTS"].to_dict()
    rows = []
    for _, g in game_header.iterrows():
        home_id, away_id = g["HOME_TEAM_ID"], g["VISITOR_TEAM_ID"]
        rows.append({
            "game_id": str(g["GAME_ID"]).zfill(10),
            "game_date": game_date,
            "season": _season_str(game_date),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_score": scores.get(home_id),
            "away_score": scores.get(away_id),
        })
    return pd.DataFrame(rows)


def _season_str(d: date) -> str:
    start_year = d.year if d.month >= 8 else d.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def pull_box_scores(game_ids: list[str]) -> pd.DataFrame:
    rows = []
    for gid in game_ids:
        try:
            frames = _fetch_boxscore(gid)
            player_stats = frames[0]
            for _, p in player_stats.iterrows():
                rows.append({
                    "player_id": p["PLAYER_ID"],
                    "game_id": gid,
                    "team_id": p["TEAM_ID"],
                    "minutes": _parse_min(p.get("MIN")),
                    "points": p.get("PTS", 0),
                    "rebounds": p.get("REB", 0),
                    "offensive_reb": p.get("OREB", 0),
                    "defensive_reb": p.get("DREB", 0),
                    "assists": p.get("AST", 0),
                    "steals": p.get("STL", 0),
                    "blocks": p.get("BLK", 0),
                    "turnovers": p.get("TO", 0),
                    "fouls": p.get("PF", 0),
                    "fgm": p.get("FGM", 0),
                    "fga": p.get("FGA", 0),
                    "fg3m": p.get("FG3M", 0),
                    "fg3a": p.get("FG3A", 0),
                    "ftm": p.get("FTM", 0),
                    "fta": p.get("FTA", 0),
                    "plus_minus": p.get("PLUS_MINUS", 0),
                    "started": bool(p.get("START_POSITION")),
                })
        except Exception as exc:
            log.warning("Box score pull failed for game %s: %s", gid, exc)
        time.sleep(REQUEST_DELAY_SECONDS)
    return pd.DataFrame(rows)


def pull_tracking_stats(game_ids: list[str]) -> pd.DataFrame:
    rows = []
    for gid in game_ids:
        try:
            frames = _fetch_tracking(gid)
            player_track = frames[0]
            for _, p in player_track.iterrows():
                rows.append({
                    "player_id": p["PLAYER_ID"],
                    "game_id": gid,
                    "distance_miles": p.get("DIST", 0),
                    "avg_speed_mph": p.get("AVG_SPEED", 0),
                    "touches": p.get("TCHS", 0),
                    "drives": p.get("DRIVES", 0),
                    "secondary_ast": p.get("SAST", 0),
                    "contested_shots": p.get("CFGA", 0),
                })
        except Exception as exc:
            log.warning("Tracking pull failed for game %s: %s", gid, exc)
        time.sleep(REQUEST_DELAY_SECONDS)
    return pd.DataFrame(rows)


def _parse_min(raw) -> float:
    if raw is None or pd.isna(raw):
        return 0.0
    s = str(raw)
    if ":" in s:
        m, sec = s.split(":")
        return round(int(m) + int(sec) / 60, 2)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _upsert(engine, df: pd.DataFrame, table: str, key_cols):
    from sqlalchemy import text
    if df.empty:
        return
    key_cols = [key_cols] if isinstance(key_cols, str) else key_cols
    staging = f"staging_{table}"
    with engine.begin() as conn:
        df.to_sql(staging, conn, if_exists="replace", index=False)
        cols = list(df.columns)
        update_cols = [c for c in cols if c not in key_cols]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols) or "created_at = now()"
        conn.execute(text(f"""
            INSERT INTO {table} ({', '.join(cols)})
            SELECT {', '.join(cols)} FROM {staging}
            ON CONFLICT ({', '.join(key_cols)}) DO UPDATE SET {set_clause}
        """))
        conn.execute(text(f"DROP TABLE {staging}"))


def main():
    parser = argparse.ArgumentParser(description="Pull recent (or historical) NBA data via nba_api")
    parser.add_argument("--days", type=int, default=3,
                         help="Pull the last N days of games, ending today (default: 3). "
                              "Ignored if --start-date/--end-date are given.")
    parser.add_argument("--start-date", type=str, default=None,
                         help="Explicit range start, YYYY-MM-DD (e.g. 2024-06-01). "
                              "Requires --end-date.")
    parser.add_argument("--end-date", type=str, default=None,
                         help="Explicit range end, YYYY-MM-DD (e.g. 2024-06-30). "
                              "Requires --start-date.")
    args = parser.parse_args()

    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be given together")

    if args.start_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
        if start_date > end_date:
            parser.error("--start-date must be on or before --end-date")
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)

    log.info("Pulling games from %s to %s", start_date, end_date)

    engine = get_engine()

    games_df = pull_games_for_range(start_date, end_date)
    if games_df.empty:
        log.info("No games found in range (off-season or no games played).")
        return
    games_df = validate_games(games_df)
    _upsert(engine, games_df, "games", "game_id")
    log.info("Upserted %d games", len(games_df))

    game_ids = games_df["game_id"].tolist()

    box_df = pull_box_scores(game_ids)
    box_df = validate_player_game_stats(box_df)
    _upsert(engine, box_df, "player_game_stats", ["player_id", "game_id"])
    log.info("Upserted %d player-game rows", len(box_df))

    track_df = pull_tracking_stats(game_ids)
    if not track_df.empty:
        track_df = validate_tracking_stats(track_df)
        _upsert(engine, track_df, "player_tracking_stats", ["player_id", "game_id"])
        log.info("Upserted %d tracking rows", len(track_df))


if __name__ == "__main__":
    main()
