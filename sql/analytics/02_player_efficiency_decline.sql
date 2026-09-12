-- ============================================================================
-- Business Question 2: Which players are showing efficiency decline right
-- now, relative to their own season baseline?
--
-- Why it matters: this is the core signal for the player decline-risk model
-- -- a player who is performing well below their own season average across
-- several recent games is a candidate for a fatigue/injury/age flag.
--
-- Technique: LAG() to build a trailing rolling average per player, compared
-- against a season-to-date average computed with a separate window frame.
-- True Shooting % (TS%) is used as the efficiency metric since it accounts
-- for 2pt/3pt/FT shot mix, unlike raw PPG.
-- ============================================================================

WITH player_game_ts AS (
    SELECT
        pgs.player_id,
        p.full_name,
        pgs.game_id,
        g.game_date,
        pgs.points,
        pgs.fga,
        pgs.fta,
        CASE
            WHEN (pgs.fga + 0.44 * pgs.fta) = 0 THEN NULL
            ELSE pgs.points::numeric / (2 * (pgs.fga + 0.44 * pgs.fta))
        END AS ts_pct
    FROM player_game_stats pgs
    JOIN games g ON g.game_id = pgs.game_id
    JOIN players p ON p.player_id = pgs.player_id
    WHERE pgs.minutes >= 10   -- filter garbage-time/low-minute noise
),
windowed AS (
    SELECT
        player_id,
        full_name,
        game_id,
        game_date,
        ts_pct,
        AVG(ts_pct) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS rolling_5_ts_pct,
        AVG(ts_pct) OVER (
            PARTITION BY player_id
            ORDER BY game_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS season_to_date_ts_pct
    FROM player_game_ts
)
SELECT
    player_id,
    full_name,
    game_id,
    game_date,
    ROUND(ts_pct, 3) AS ts_pct,
    ROUND(rolling_5_ts_pct, 3) AS rolling_5_ts_pct,
    ROUND(season_to_date_ts_pct, 3) AS season_to_date_ts_pct,
    ROUND(rolling_5_ts_pct - season_to_date_ts_pct, 3) AS decline_gap
FROM windowed
WHERE rolling_5_ts_pct < season_to_date_ts_pct - 0.05   -- >5 pt TS% drop
ORDER BY decline_gap ASC;

-- Finding (example): a meaningful subset of players 28+ show rolling-5 TS%
-- at least 5 points below their season average during the second half of
-- back-to-back-heavy stretches.
-- Insight: a sustained (5-game) efficiency gap vs. season baseline is a
-- usable early-warning signal, feeding directly into the decline-risk model.
