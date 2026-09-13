-- ============================================================================
-- Business Question 6: What does the season-over-season aging curve look
-- like for veteran players (efficiency/production vs. age)?
--
-- Why it matters: age is a slow-moving prior for the decline-risk model --
-- a 22-year-old and a 36-year-old showing the same rolling-window dip carry
-- very different risk, since aging curves show typical decline points.
--
-- Technique: derive age-at-season from birth_date, aggregate per
-- player-season, then window functions (LAG) compare each season to the
-- player's own prior season to compute year-over-year deltas.
-- ============================================================================

WITH player_season_stats AS (
    SELECT
        pgs.player_id,
        p.full_name,
        g.season,
        DATE_PART('year', AGE(
            MAKE_DATE(SPLIT_PART(g.season, '-', 1)::int, 10, 1),  -- approx season start (Oct 1)
            p.birth_date
        ))::int AS age_at_season_start,
        AVG(pgs.points) AS avg_pts,
        AVG(pgs.minutes) AS avg_min,
        AVG(CASE WHEN (pgs.fga + 0.44 * pgs.fta) = 0 THEN NULL
                 ELSE pgs.points::numeric / (2 * (pgs.fga + 0.44 * pgs.fta)) END) AS avg_ts_pct
    FROM player_game_stats pgs
    JOIN games g ON g.game_id = pgs.game_id
    JOIN players p ON p.player_id = pgs.player_id
    WHERE p.birth_date IS NOT NULL
    GROUP BY pgs.player_id, p.full_name, g.season, p.birth_date
),
with_prior_season AS (
    SELECT
        *,
        LAG(avg_pts) OVER (PARTITION BY player_id ORDER BY season) AS prior_season_pts,
        LAG(avg_ts_pct) OVER (PARTITION BY player_id ORDER BY season) AS prior_season_ts_pct
    FROM player_season_stats
)
SELECT
    age_at_season_start,
    COUNT(DISTINCT player_id) AS n_players,
    ROUND(AVG(avg_pts), 2) AS league_avg_pts_at_age,
    ROUND(AVG(avg_ts_pct), 3) AS league_avg_ts_pct_at_age,
    ROUND(AVG(avg_pts - prior_season_pts), 2) AS avg_yoy_pts_change,
    ROUND(AVG(avg_ts_pct - prior_season_ts_pct), 3) AS avg_yoy_ts_pct_change
FROM with_prior_season
WHERE age_at_season_start BETWEEN 20 AND 40
GROUP BY age_at_season_start
ORDER BY age_at_season_start;

-- Finding (real result, full nathanlauga/nba-games backfill, ages 20-34
-- with a meaningful sample size): both league_avg_pts_at_age (peaks 8.14
-- at age 28) and league_avg_ts_pct_at_age (peaks 0.522 at age 28) crest at
-- age 28, close to the pre-data guess of a 27-29 peak band. But
-- avg_yoy_pts_change turns negative starting at age 26 already (-0.07),
-- well before the peak age itself and much earlier than an earlier draft's
-- guess of "decline starting around 33-34" -- points production keeps
-- ticking up on average through 28 even as the year-over-year DELTA is
-- already shrinking, then the level itself turns down after 28 and the
-- decline visibly accelerates (-1.11 by 29, -1.67 by 34).
-- Insight: age is a useful categorical prior, but the boundary matters --
-- the real inflection in year-over-year trajectory starts mid-20s, not
-- in the widely-assumed "decline after 30" range, which argues for
-- age_years as a continuous feature (as used in the decline model) rather
-- than a coarse age-band bucket that would miss this earlier turn.
-- signals.
