"""
Backfill loader: reads a Kaggle bulk NBA dataset (e.g. "nba-games" /
"nba-players-stats" style CSV exports covering 5-10 seasons of box scores)
from data/raw/kaggle/, validates it, and upserts into Postgres.

Expected input files (adjust KAGGLE_FILES if your Kaggle export uses
different names — this targets the common wyattowalsh/nba-database or
nathanlauga/nba-games Kaggle layout):
    games.csv               -> games table
    teams.csv                -> teams table
    players.csv               -> players table
    games_details.csv (or player_box_scores.csv) -> player_game_stats table

Run:
    python ingestion/load_kaggle_backfill.py
"""
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402
from ingestion.validation import (  # noqa: E402
    validate_games,
    validate_player_game_stats,
    validate_players,
    validate_teams,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kaggle_backfill")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "kaggle"

# Real nba.com person_ids top out around 1.6-1.7 million as of the mid-2020s
# (they're assigned sequentially and roughly track draft/debut order). This
# export's games_details.csv contains a handful of corrupted rows with
# 9-10 digit player_ids (e.g. 1962936250) paired with garbage/mismatched
# names (one literal row was named "Matt Matt") -- clearly not real nba.com
# IDs. Filtering them out here keeps both load_players' supplemental-player
# logic and load_player_game_stats' box-score rows from ever re-ingesting
# this junk on a future backfill re-run.
MAX_PLAUSIBLE_PLAYER_ID = 9_999_999


def _filter_implausible_player_ids(df: pd.DataFrame, id_col: str = "player_id") -> pd.DataFrame:
    bad = df[df[id_col] > MAX_PLAUSIBLE_PLAYER_ID]
    if len(bad):
        log.warning(
            "Dropping %d games_details.csv row(s) with implausible player_id "
            "(> %d, not a real nba.com id): %s",
            len(bad), MAX_PLAUSIBLE_PLAYER_ID, bad[id_col].unique().tolist(),
        )
    return df[df[id_col] <= MAX_PLAUSIBLE_PLAYER_ID].copy()


def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected Kaggle export at {path}. Download the bulk dataset "
            f"from Kaggle and place its CSVs in {DATA_DIR}."
        )
    return pd.read_csv(path, low_memory=False)


def load_teams(engine):
    """
    The common Kaggle bulk export (nathanlauga/nba-games teams.csv) does NOT
    include conference/division columns -- only ABBREVIATION, NICKNAME, CITY,
    ARENA, etc. We fill conference/division from a static reference table
    (ingestion/reference/team_conferences.csv) keyed on abbreviation, which
    also covers historically relocated/renamed franchises (SEA->OKC,
    NJN->BKN, VAN->MEM, etc.) so multi-season backfills don't lose the join.
    """
    df = _read_csv("teams.csv")
    df = df.rename(columns=str.lower)

    ref_path = Path(__file__).resolve().parent / "reference" / "team_conferences.csv"
    ref = pd.read_csv(ref_path)

    merged = df.merge(ref, on="abbreviation", how="left")
    missing = merged[merged["conference"].isna()]["abbreviation"].unique()
    if len(missing):
        log.warning(
            "No conference/division mapping for abbreviations: %s -- "
            "add them to ingestion/reference/team_conferences.csv",
            list(missing),
        )

    out = pd.DataFrame({
        "team_id": merged["team_id"],
        "abbreviation": merged["abbreviation"],
        "name": merged.get("nickname", merged.get("name")),
        "conference": merged["conference"],
        "division": merged["division"],
    })
    out = validate_teams(out)
    _upsert(engine, out, "teams", "team_id")
    log.info("Loaded %d teams", len(out))


def load_players(engine):
    """
    The common Kaggle bulk export's players.csv has one row per
    (player, season) with only PLAYER_NAME / PLAYER_ID / TEAM_ID / SEASON --
    no position, birth_date, height, or weight. We take the most recent
    team_id per player here; birth_date/position/height/weight are left NULL
    and filled in separately by ingestion/enrich_player_bio.py (nba_api),
    since aging-curve analysis and the decline model's age_years feature
    need birth_date.

    players.csv is also NOT a complete player roster -- games_details.csv
    (the box-score source) references some player_ids that never appear in
    players.csv at all (likely short call-ups / rare appearances the Kaggle
    export's players.csv snapshot missed). Loading player_game_stats would
    otherwise violate its FK on players, so we supplement with any
    (player_id, player_name) pairs found only in games_details.csv.

    Note: the `position` column is intentionally left OUT of the upsert
    entirely (not set to None) so re-running this backfill never clobbers
    values already filled in by enrich_player_bio.py.
    """
    df = _read_csv("players.csv")
    df = df.rename(columns=str.lower)
    if "season" in df.columns:
        df = df.sort_values("season").drop_duplicates(subset=["player_id"], keep="last")
    out = pd.DataFrame({
        "player_id": df["player_id"],
        "full_name": df.get("player_name", df.get("full_name")),
        "team_id": df.get("team_id"),
    })

    # supplement with player_ids that only appear in games_details.csv
    details = _read_csv("games_details.csv")
    details = details.rename(columns=str.lower)
    details = _filter_implausible_player_ids(details)
    supplemental = (
        details[["player_id", "player_name", "team_id"]]
        .rename(columns={"player_name": "full_name"})
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"])
    )
    missing = supplemental[~supplemental["player_id"].isin(out["player_id"])]
    if len(missing):
        log.info("Adding %d player_ids found in games_details.csv but missing from players.csv", len(missing))
        out = pd.concat([out, missing], ignore_index=True)

    out = validate_players(out)
    _upsert(engine, out, "players", "player_id")
    log.info("Loaded %d players (position/birth_date left NULL -- run "
              "ingestion/enrich_player_bio.py to fill them in)", len(out))


def load_games(engine):
    df = _read_csv("games.csv")
    df = df.rename(columns=str.lower)
    out = pd.DataFrame({
        "game_id": df["game_id"].astype(str).str.zfill(10),
        "game_date": pd.to_datetime(df["game_date_est"]).dt.date,
        "season": df["season"].apply(lambda s: f"{s}-{str(int(s) + 1)[-2:]}"),
        "home_team_id": df["home_team_id"],
        "away_team_id": df["visitor_team_id"],
        "home_score": df["pts_home"],
        "away_score": df["pts_away"],
    })
    out = validate_games(out)
    _upsert(engine, out, "games", "game_id")
    log.info("Loaded %d games", len(out))


def load_player_game_stats(engine):
    df = _read_csv("games_details.csv")
    df = df.rename(columns=str.lower)
    df = _filter_implausible_player_ids(df)
    out = pd.DataFrame({
        "player_id": df["player_id"],
        "game_id": df["game_id"].astype(str).str.zfill(10),
        "team_id": df["team_id"],
        "minutes": df["min"].apply(_parse_minutes),
        "points": df["pts"],
        "rebounds": df["reb"],
        "offensive_reb": df.get("oreb"),
        "defensive_reb": df.get("dreb"),
        "assists": df["ast"],
        "steals": df.get("stl"),
        "blocks": df.get("blk"),
        "turnovers": df.get("to"),
        "fouls": df.get("pf"),
        "fgm": df.get("fgm"),
        "fga": df.get("fga"),
        "fg3m": df.get("fg3m"),
        "fg3a": df.get("fg3a"),
        "ftm": df.get("ftm"),
        "fta": df.get("fta"),
        "plus_minus": df.get("plus_minus"),
        "started": df.get("start_position").notna() if "start_position" in df else False,
    })
    out = validate_player_game_stats(out)
    _upsert(engine, out, "player_game_stats", ["player_id", "game_id"])
    log.info("Loaded %d player-game rows", len(out))


_MIN_RE = re.compile(r"^(\d+(?:\.\d+)?)(?::(\d+))?$")


def _parse_minutes(raw) -> float:
    """
    Kaggle box scores store minutes in a few inconsistent formats:
      'MM:SS'            e.g. '18:06'
      'MM' (plain int)   e.g. '19'
      'MM.000000:SS'     e.g. '29.000000:24' -- a malformed variant seen in
                          this export where a float-formatted minute got
                          concatenated with ':SS' instead of cleanly replaced.
    The regex pulls the leading minutes (int or float) and an optional
    trailing seconds group, so all three shapes resolve correctly.
    """
    if pd.isna(raw):
        return 0.0
    s = str(raw).strip()
    match = _MIN_RE.match(s)
    if not match:
        return 0.0
    minutes = int(float(match.group(1)))
    seconds = int(match.group(2)) if match.group(2) else 0
    return round(minutes + seconds / 60, 2)


def _upsert(engine, df: pd.DataFrame, table: str, key_cols):
    """Simple upsert via a temp staging table + ON CONFLICT DO UPDATE.
    Fine at Kaggle-backfill volumes (millions of rows is still seconds-to-minutes)."""
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
    engine = get_engine()
    load_teams(engine)
    load_players(engine)
    load_games(engine)
    load_player_game_stats(engine)
    log.info("Kaggle backfill complete.")


if __name__ == "__main__":
    main()
