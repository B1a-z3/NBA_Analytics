"""
Enriches the players table with birth_date, position, height_inches, and
weight_lbs, pulled from nba_api's commonplayerinfo endpoint.

This exists because the common Kaggle bulk export (nathanlauga/nba-games
players.csv) only has PLAYER_ID / PLAYER_NAME / TEAM_ID / SEASON -- no bio
data -- but birth_date is required for the aging-curve SQL analysis
(sql/analytics/06_aging_curve.sql) and the player-decline model's
age_years feature.

Only fetches players currently missing birth_date, so it's safe to re-run
(e.g. after a fresh Kaggle backfill wipes/reloads the players table).

Run:
    python ingestion/enrich_player_bio.py
    python ingestion/enrich_player_bio.py --limit 50   # smoke test a subset first
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402
import ingestion.nba_api_tls_patch  # noqa: E402,F401 -- must precede nba_api.stats.endpoints imports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("enrich_player_bio")

REQUEST_DELAY_SECONDS = 0.6
HEIGHT_RE_FEET_INCHES = "-"  # nba_api returns HEIGHT as "6-9" (feet-inches)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_player_info(player_id: int):
    from nba_api.stats.endpoints import commonplayerinfo
    return commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]


def _parse_height(raw) -> int | None:
    """nba_api HEIGHT field looks like '6-9' -> 81 inches."""
    if raw is None or pd.isna(raw) or "-" not in str(raw):
        return None
    feet, inches = str(raw).split("-")
    try:
        return int(feet) * 12 + int(inches)
    except ValueError:
        return None


def fetch_players_needing_enrichment(engine, limit: int | None) -> pd.DataFrame:
    query = "SELECT player_id, full_name FROM players WHERE birth_date IS NULL"
    if limit:
        query += f" LIMIT {int(limit)}"
    return pd.read_sql(query, engine)


def enrich(engine, players_df: pd.DataFrame):
    updated = 0
    failed = []
    with engine.begin() as conn:
        for _, row in players_df.iterrows():
            player_id = row["player_id"]
            try:
                info = _fetch_player_info(player_id)
                if info.empty:
                    failed.append(player_id)
                    continue
                r = info.iloc[0]
                conn.execute(text("""
                    UPDATE players
                    SET position = :position,
                        birth_date = :birth_date,
                        height_inches = :height_inches,
                        weight_lbs = :weight_lbs
                    WHERE player_id = :player_id
                """), {
                    "player_id": int(player_id),
                    "position": r.get("POSITION"),
                    "birth_date": pd.to_datetime(r.get("BIRTHDATE")).date() if r.get("BIRTHDATE") else None,
                    "height_inches": _parse_height(r.get("HEIGHT")),
                    "weight_lbs": int(r["WEIGHT"]) if pd.notna(r.get("WEIGHT")) and str(r.get("WEIGHT")).isdigit() else None,
                })
                updated += 1
            except Exception as exc:
                log.warning("Failed to enrich player_id=%s: %s", player_id, exc)
                failed.append(player_id)
            time.sleep(REQUEST_DELAY_SECONDS)
    return updated, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only enrich the first N players missing bio data")
    args = parser.parse_args()

    engine = get_engine()
    players_df = fetch_players_needing_enrichment(engine, args.limit)

    if players_df.empty:
        log.info("No players need bio enrichment.")
        return

    log.info("Enriching %d players via nba_api commonplayerinfo (this is slow -- "
              "~%.0f min at %.1fs/request)...",
              len(players_df), len(players_df) * REQUEST_DELAY_SECONDS / 60, REQUEST_DELAY_SECONDS)

    updated, failed = enrich(engine, players_df)
    log.info("Enriched %d players. %d failed/skipped.", updated, len(failed))
    if failed:
        log.info("Failed player_ids (re-run will retry these): %s", failed[:20])


if __name__ == "__main__":
    main()
