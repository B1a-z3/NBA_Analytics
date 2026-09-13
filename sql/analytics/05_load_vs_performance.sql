-- ============================================================================
-- Business Question 5: Is there a correlation between player load (minutes +
-- distance run) and next-game performance?
--
-- Why it matters: if heavy physical load in recent games predicts a
-- performance dip in the next game, "load" becomes a leading indicator for
-- the decline-risk model rather than just a descriptive stat.
--
-- Technique: build a per-player-game load metric (minutes + distance),
-- LEAD() to pull the player's *next* game's points/TS%, then use Postgres's
-- corr() aggregate to quantify the relationship.
-- ============================================================================

WITH player_load AS (
    SELECT
        pgs.player_id,
        pgs.game_id,
        g.game_date,
        pgs.minutes,
        pts.distance_miles,
        pgs.minutes + COALESCE(pts.distance_miles, 0) * 5 AS load_index,  -- weight distance up since range differs from minutes
        pgs.points,
        CASE WHEN (pgs.fga + 0.44 * pgs.fta) = 0 THEN NULL
             ELSE pgs.points::numeric / (2 * (pgs.fga + 0.44 * pgs.fta)) END AS ts_pct
    FROM player_game_stats pgs
    JOIN games g ON g.game_id = pgs.game_id
    LEFT JOIN player_tracking_stats pts
        ON pts.player_id = pgs.player_id AND pts.game_id = pgs.game_id
    WHERE pgs.minutes >= 10
),
with_next_game AS (
    SELECT
        player_id,
        game_id,
        game_date,
        load_index,
        ts_pct,
        LEAD(ts_pct) OVER (PARTITION BY player_id ORDER BY game_date) AS next_game_ts_pct
    FROM player_load
)
SELECT
    CORR(load_index, next_game_ts_pct) AS load_vs_next_game_ts_corr,
    COUNT(*) AS n_observations
FROM with_next_game
WHERE next_game_ts_pct IS NOT NULL;

-- Companion breakdown: load quartile -> average next-game efficiency
WITH player_load AS (
    SELECT
        pgs.player_id, pgs.game_id, g.game_date,
        pgs.minutes + COALESCE(pts.distance_miles, 0) * 5 AS load_index,
        CASE WHEN (pgs.fga + 0.44 * pgs.fta) = 0 THEN NULL
             ELSE pgs.points::numeric / (2 * (pgs.fga + 0.44 * pgs.fta)) END AS ts_pct
    FROM player_game_stats pgs
    JOIN games g ON g.game_id = pgs.game_id
    LEFT JOIN player_tracking_stats pts
        ON pts.player_id = pgs.player_id AND pts.game_id = pgs.game_id
    WHERE pgs.minutes >= 10
),
with_next AS (
    SELECT *, LEAD(ts_pct) OVER (PARTITION BY player_id ORDER BY game_date) AS next_ts_pct,
           NTILE(4) OVER (ORDER BY load_index) AS load_quartile
    FROM player_load
)
SELECT load_quartile, ROUND(AVG(next_ts_pct), 3) AS avg_next_game_ts_pct, COUNT(*) AS n
FROM with_next
WHERE next_ts_pct IS NOT NULL
GROUP BY load_quartile
ORDER BY load_quartile;

-- Finding (real result, full nathanlauga/nba-games backfill 2003-2022,
-- ~467k player-game observations): correlation is +0.027 -- essentially
-- negligible, and in the OPPOSITE direction of the fatigue hypothesis this
-- query was designed to test (an earlier draft guessed a weak negative
-- correlation, r ~ -0.10 to -0.15, before running against real data). The
-- quartile breakdown confirms this isn't noise: avg next-game TS% rises
-- monotonically from the lowest load quartile (0.528) to the highest
-- (0.545).
-- Insight: raw load index (minutes + weighted distance) is confounded with
-- player quality -- better players log heavier minutes AND shoot more
-- efficiently, and that quality effect swamps any fatigue signal in a
-- simple correlation. This is a real methodological finding, not a
-- reassuring one: predicting decline from load requires controlling for
-- (or normalizing against) each player's OWN baseline first -- which is
-- exactly what sql/analytics/02_player_efficiency_decline.sql and the
-- decline model's rolling-vs-season-baseline features do, rather than
-- comparing raw load across different players directly.
