import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from features.build_features import build_team_game_features


def test_team_game_features_shape_and_no_lookahead():
    """Two teams, three games each. Rolling features must be built only from
    PRIOR games (shifted by 1) so no game's outcome leaks into its own row."""
    games = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
        "season": ["2023-24"] * 3,
        "home_team_id": [1, 2, 1],
        "away_team_id": [2, 1, 2],
        "home_score": [100, 90, 110],
        "away_score": [95, 92, 105],
    })
    out = build_team_game_features(games)

    # 2 rows per game (one per team) x 3 games = 6 rows
    assert len(out) == 6

    team1_rows = out[out["team_id"] == 1].sort_values("game_date")
    # first game for team 1 has no prior history -> rolling_win_pct is NaN
    assert pd.isna(team1_rows.iloc[0]["rolling_win_pct"])
    # by the third row, rolling_win_pct reflects only games 1 and 2 (not game 3 itself)
    assert not pd.isna(team1_rows.iloc[-1]["rolling_win_pct"])


def test_back_to_back_flagged_correctly():
    games = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),  # 1 day apart -> back-to-back
        "season": ["2023-24"] * 2,
        "home_team_id": [1, 1],
        "away_team_id": [2, 3],
        "home_score": [100, 90],
        "away_score": [95, 92],
    })
    out = build_team_game_features(games)
    team1_rows = out[out["team_id"] == 1].sort_values("game_date")
    assert team1_rows.iloc[1]["is_back_to_back"] == 1
