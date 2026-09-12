"""
Validation/cleaning layer sitting between raw landing data and the Postgres
warehouse. Each function takes a DataFrame shaped for a target table and
returns a cleaned DataFrame, raising on unrecoverable problems and logging
+ dropping rows for recoverable ones. This is intentionally lightweight
(no great_expectations dependency) but documents the checks explicitly so
the "validation" step in the architecture diagram is a real, visible thing.
"""
import logging

import pandas as pd

log = logging.getLogger("validation")


def _drop_and_log(df: pd.DataFrame, mask, reason: str) -> pd.DataFrame:
    bad = df[~mask]
    if len(bad):
        log.warning("Dropping %d rows: %s", len(bad), reason)
    return df[mask].copy()


def validate_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["team_id", "abbreviation", "name"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df = df.drop_duplicates(subset=["team_id"])
    return df


def validate_players(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["player_id", "full_name"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"] = df["team_id"].where(df["team_id"].notna(), None)
    df = df.drop_duplicates(subset=["player_id"])
    return df


def validate_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["game_id", "game_date", "home_team_id", "away_team_id"])
    df = _drop_and_log(df, df["home_team_id"] != df["away_team_id"], "home == away team")
    if "home_score" in df:
        df = _drop_and_log(
            df,
            df["home_score"].isna() | ((df["home_score"] >= 0) & (df["away_score"] >= 0)),
            "negative score",
        )
    df = df.drop_duplicates(subset=["game_id"])
    return df


def validate_player_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["player_id", "game_id", "team_id"])
    numeric_cols = [
        "minutes", "points", "rebounds", "assists", "steals", "blocks",
        "turnovers", "fouls", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # sanity: makes can't exceed attempts
    for made, att in [("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")]:
        if made in df.columns and att in df.columns:
            df = _drop_and_log(df, df[made] <= df[att], f"{made} > {att}")
    df = df.drop_duplicates(subset=["player_id", "game_id"])
    return df


def validate_tracking_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["player_id", "game_id"])
    if "distance_miles" in df.columns:
        df = _drop_and_log(df, df["distance_miles"].between(0, 10), "implausible distance")
    if "avg_speed_mph" in df.columns:
        df = _drop_and_log(df, df["avg_speed_mph"].between(0, 15), "implausible speed")
    df = df.drop_duplicates(subset=["player_id", "game_id"])
    return df
