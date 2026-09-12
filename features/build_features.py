"""
Feature engineering layer. Pulls from the Postgres warehouse and builds two
feature tables (returned as DataFrames, and optionally materialized back to
Postgres for reuse by the API at inference time):

  1. team_game_features  — one row per (team, game): rolling form, rest days,
     home/away, opponent strength -> feeds the game-outcome models.
  2. player_game_features — one row per (player, game): rolling performance,
     season baseline, load index, rest days, age -> feeds the decline model.

Run:
    python features/build_features.py               # build + materialize
    python features/build_features.py --no-write    # build only, for notebooks
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

ROLLING_WINDOW_TEAM = 10
ROLLING_WINDOW_PLAYER = 5


def _load_raw(engine):
    games = pd.read_sql("SELECT * FROM games WHERE home_score IS NOT NULL", engine)
    pgs = pd.read_sql("SELECT * FROM player_game_stats", engine)
    tracking = pd.read_sql("SELECT * FROM player_tracking_stats", engine)
    players = pd.read_sql("SELECT player_id, full_name, birth_date FROM players", engine)
    return games, pgs, tracking, players


def _team_long_format(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape games (1 row per game) into 2 rows per game (one per team)."""
    home = games.rename(columns={
        "home_team_id": "team_id", "away_team_id": "opponent_id",
        "home_score": "team_score", "away_score": "opponent_score",
    })
    home["is_home"] = True
    away = games.rename(columns={
        "away_team_id": "team_id", "home_team_id": "opponent_id",
        "away_score": "team_score", "home_score": "opponent_score",
    })
    away["is_home"] = False
    cols = ["game_id", "game_date", "season", "team_id", "opponent_id",
            "team_score", "opponent_score", "is_home"]
    long_df = pd.concat([home[cols], away[cols]], ignore_index=True)
    long_df["won"] = (long_df["team_score"] > long_df["opponent_score"]).astype(int)
    return long_df.sort_values(["team_id", "game_date"])


def build_team_game_features(games: pd.DataFrame) -> pd.DataFrame:
    long_df = _team_long_format(games)

    long_df["rolling_win_pct"] = (
        long_df.groupby("team_id")["won"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW_TEAM, min_periods=1).mean())
    )
    point_diff = long_df["team_score"] - long_df["opponent_score"]
    long_df["rolling_point_diff"] = (
        point_diff.groupby(long_df["team_id"])
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW_TEAM, min_periods=1).mean())
    )
    long_df["days_rest"] = (
        long_df.groupby("team_id")["game_date"]
        .transform(lambda s: s.diff().dt.days)
    )
    long_df["is_back_to_back"] = (long_df["days_rest"] <= 1).astype(int)

    # opponent's rolling win pct at time of game -> matchup context
    opp_form = long_df[["game_id", "team_id", "rolling_win_pct"]].rename(
        columns={"team_id": "opponent_id", "rolling_win_pct": "opponent_rolling_win_pct"}
    )
    long_df = long_df.merge(opp_form, on=["game_id", "opponent_id"], how="left")

    return long_df[[
        "game_id", "game_date", "season", "team_id", "opponent_id", "is_home",
        "won", "rolling_win_pct", "rolling_point_diff", "days_rest",
        "is_back_to_back", "opponent_rolling_win_pct",
    ]]


def build_player_game_features(pgs: pd.DataFrame, games: pd.DataFrame,
                                 tracking: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    df = pgs.merge(games[["game_id", "game_date", "season"]], on="game_id", how="left")
    df = df.merge(tracking, on=["player_id", "game_id"], how="left")
    df = df.merge(players, on="player_id", how="left")
    df = df.sort_values(["player_id", "game_date"])

    # true shooting %
    denom = 2 * (df["fga"] + 0.44 * df["fta"])
    df["ts_pct"] = np.where(denom > 0, df["points"] / denom, np.nan)

    df["rolling_ts_pct"] = (
        df.groupby("player_id")["ts_pct"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW_PLAYER, min_periods=2).mean())
    )
    df["season_to_date_ts_pct"] = (
        df.groupby(["player_id", "season"])["ts_pct"]
        .transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    )
    df["ts_pct_decline_gap"] = df["rolling_ts_pct"] - df["season_to_date_ts_pct"]

    df["rolling_minutes"] = (
        df.groupby("player_id")["minutes"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW_PLAYER, min_periods=1).mean())
    )
    df["distance_miles"] = df["distance_miles"].fillna(0)
    df["load_index"] = df["minutes"] + df["distance_miles"] * 5
    df["rolling_load_index"] = (
        df.groupby("player_id")["load_index"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW_PLAYER, min_periods=1).mean())
    )

    df["days_rest"] = df.groupby("player_id")["game_date"].transform(lambda s: s.diff().dt.days)
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

    df["age_years"] = (
        (pd.to_datetime(df["game_date"]) - pd.to_datetime(df["birth_date"])).dt.days / 365.25
    )

    feature_cols = [
        "player_id", "full_name", "game_id", "game_date", "season", "team_id",
        "minutes", "points", "ts_pct", "rolling_ts_pct", "season_to_date_ts_pct",
        "ts_pct_decline_gap", "rolling_minutes", "load_index", "rolling_load_index",
        "days_rest", "is_back_to_back", "age_years",
    ]
    return df[feature_cols]


def materialize(engine, team_features: pd.DataFrame, player_features: pd.DataFrame):
    team_features.to_sql("team_game_features", engine, if_exists="replace", index=False)
    player_features.to_sql("player_game_features", engine, if_exists="replace", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true", help="Build features but don't write to Postgres")
    args = parser.parse_args()

    engine = get_engine()
    games, pgs, tracking, players = _load_raw(engine)

    if games.empty:
        print("No completed games found in warehouse -- run ingestion first.")
        return

    team_features = build_team_game_features(games)
    player_features = build_player_game_features(pgs, games, tracking, players)

    print(f"Built team_game_features: {team_features.shape}")
    print(f"Built player_game_features: {player_features.shape}")

    if not args.no_write:
        materialize(engine, team_features, player_features)
        print("Materialized feature tables to Postgres (team_game_features, player_game_features).")


if __name__ == "__main__":
    main()
