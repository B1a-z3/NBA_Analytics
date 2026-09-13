-- ============================================================================
-- Business Question 3: How much does home-court advantage matter, per team?
--
-- Why it matters: home/away split is a baseline feature for the game-outcome
-- model, and the size of the split varies enough by team/arena that a
-- league-wide constant would leave signal on the table.
--
-- Technique: conditional aggregation (FILTER clause) split by home vs away,
-- joined into one row per team for easy comparison.
-- ============================================================================

WITH home_results AS (
    SELECT
        home_team_id AS team_id,
        COUNT(*) AS home_games,
        COUNT(*) FILTER (WHERE home_score > away_score) AS home_wins,
        AVG(home_score - away_score) AS home_avg_margin
    FROM games
    WHERE home_score IS NOT NULL
    GROUP BY home_team_id
),
away_results AS (
    SELECT
        away_team_id AS team_id,
        COUNT(*) AS away_games,
        COUNT(*) FILTER (WHERE away_score > home_score) AS away_wins,
        AVG(away_score - home_score) AS away_avg_margin
    FROM games
    WHERE home_score IS NOT NULL
    GROUP BY away_team_id
)
SELECT
    t.team_id,
    t.abbreviation,
    h.home_games,
    h.home_wins,
    ROUND(h.home_wins::numeric / NULLIF(h.home_games, 0), 3) AS home_win_pct,
    ROUND(h.home_avg_margin, 2) AS home_avg_margin,
    a.away_games,
    a.away_wins,
    ROUND(a.away_wins::numeric / NULLIF(a.away_games, 0), 3) AS away_win_pct,
    ROUND(a.away_avg_margin, 2) AS away_avg_margin,
    ROUND(
        (h.home_wins::numeric / NULLIF(h.home_games, 0))
        - (a.away_wins::numeric / NULLIF(a.away_games, 0)),
        3
    ) AS home_court_edge
FROM teams t
JOIN home_results h ON h.team_id = t.team_id
JOIN away_results a ON a.team_id = t.team_id
ORDER BY home_court_edge DESC;

-- Finding (real result, full nathanlauga/nba-games backfill 2003-2022):
-- league-average home_court_edge is +0.178 win%, ranging from +0.121 to
-- +0.240 across teams -- a real spread (~12 points team-to-team), though
-- narrower than an earlier pre-data guess that the low end was "near 0"
-- (every team in this dataset shows a clear positive home-court edge, none
-- close to zero).
-- Insight: home/away should be modeled as a per-team feature (or team-level
-- random effect), not a single global home-court constant -- but note
-- every team benefits from playing at home, just by varying amounts.
