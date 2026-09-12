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

-- Finding (example): weak-to-moderate negative correlation between load
-- index and next-game TS% (e.g. r ~ -0.10 to -0.15); top load quartile shows
-- a small but consistent efficiency dip the following game.
-- Insight: raw load alone is a modest predictor -- it strengthens combined
-- with rest days (see query 4), supporting a combined fatigue feature.
