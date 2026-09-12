-- ============================================================================
-- Business Question 1: What is each team's rolling 10-game win % trend?
--
-- Why it matters: a team's season-long record can hide hot/cold streaks that
-- matter more for predicting the NEXT game than the full-season average does.
-- This is the core "team form" signal feeding the game-outcome model.
--
-- Technique: UNION each game into a per-team perspective (once as home, once
-- as away), then a window function (AVG ... OVER ROWS BETWEEN 9 PRECEDING)
-- computes the trailing 10-game win rate ordered by date.
-- ============================================================================

WITH team_game_results AS (
    SELECT
        game_id,
        game_date,
        home_team_id AS team_id,
        (home_score > away_score) AS won
    FROM games
    WHERE home_score IS NOT NULL

    UNION ALL

    SELECT
        game_id,
        game_date,
        away_team_id AS team_id,
        (away_score > home_score) AS won
    FROM games
    WHERE home_score IS NOT NULL
),
rolling AS (
    SELECT
        t.team_id,
        tm.abbreviation,
        t.game_date,
        t.game_id,
        t.won,
        AVG(t.won::int) OVER (
            PARTITION BY t.team_id
            ORDER BY t.game_date
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS rolling_10_win_pct,
        COUNT(*) OVER (
            PARTITION BY t.team_id
            ORDER BY t.game_date
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS games_in_window
    FROM team_game_results t
    JOIN teams tm ON tm.team_id = t.team_id
)
SELECT
    team_id,
    abbreviation,
    game_date,
    game_id,
    won,
    ROUND(rolling_10_win_pct, 3) AS rolling_10_win_pct,
    games_in_window
FROM rolling
ORDER BY team_id, game_date;

-- Finding (example, populate after running against real data):
--   Teams with a rolling 10-game win% above .700 in the last 2 weeks of a
--   season win their next game ~68% of the time, vs. ~50% for teams at .500.
-- Insight: recent form (last 10 games) is a stronger single predictor of the
-- next game's outcome than season-long win percentage.
