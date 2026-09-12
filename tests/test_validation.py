import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ingestion.validation import (
    validate_games,
    validate_player_game_stats,
    validate_players,
    validate_teams,
)


def test_validate_teams_drops_nulls_and_dupes():
    df = pd.DataFrame({
        "team_id": [1, 1, None, 3],
        "abbreviation": ["A", "A", "B", "C"],
        "name": ["Team A", "Team A", "Team B", "Team C"],
        "conference": ["East", "East", "West", "West"],
        "division": ["Atlantic", "Atlantic", "Pacific", "Pacific"],
    })
    out = validate_teams(df)
    assert len(out) == 2
    assert set(out["team_id"]) == {1, 3}


def test_validate_players_nullable_team_id():
    df = pd.DataFrame({
        "player_id": [1, 2],
        "full_name": ["Player One", "Player Two"],
        "position": ["G", "F"],
        "team_id": [10, None],
    })
    out = validate_players(df)
    assert len(out) == 2


def test_validate_games_drops_self_matchup_and_negative_scores():
    df = pd.DataFrame({
        "game_id": ["1", "2", "3"],
        "game_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "season": ["2023-24"] * 3,
        "home_team_id": [1, 2, 3],
        "away_team_id": [2, 2, 4],   # row 2 is a self-matchup (2 vs 2)
        "home_score": [100, 100, -5],  # row 3 has negative score
        "away_score": [95, 90, 80],
    })
    out = validate_games(df)
    assert len(out) == 1
    assert out.iloc[0]["game_id"] == "1"


def test_validate_player_game_stats_fgm_exceeds_fga_dropped():
    df = pd.DataFrame({
        "player_id": [1, 2],
        "game_id": ["1", "1"],
        "team_id": [1, 1],
        "minutes": [30, 25],
        "points": [20, 15],
        "fgm": [8, 12],   # row 2: fgm > fga (invalid)
        "fga": [15, 10],
        "fg3m": [2, 1], "fg3a": [5, 3],
        "ftm": [2, 0], "fta": [2, 0],
    })
    out = validate_player_game_stats(df)
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == 1
