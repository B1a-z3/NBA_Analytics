-- ============================================================================
-- Business Question 4: Does playing on zero days rest (a "back-to-back")
-- hurt a team's shooting percentage?
--
-- Why it matters: rest days is one of the highest-signal fatigue features
-- for both the game-outcome model and the player decline model, and is easy
-- to derive purely from the schedule (no tracking data needed).
--
-- Technique: self-join games to itself on team_id to find each team's
-- previous game date, compute days_rest, then compare team shooting % in
-- back-to-back (days_rest = 0) vs. rested (days_rest >= 2) games.
-- ============================================================================

WITH team_schedule AS (
    SELECT game_id, game_date, home_team_id AS team_id FROM games WHERE home_score IS NOT NULL
    UNION ALL
    SELECT game_id, game_date, away_team_id AS team_id FROM games WHERE home_score IS NOT NULL
),
with_rest AS (
    SELECT
        cur.team_id,
        cur.game_id,
        cur.game_date,
        cur.game_date - prev.game_date AS days_rest
    FROM team_schedule cur
    JOIN LATERAL (
        SELECT game_date
        FROM team_schedule prev
        WHERE prev.team_id = cur.team_id
          AND prev.game_date < cur.game_date
        ORDER BY prev.game_date DESC
        LIMIT 1
    ) prev ON true
),
team_game_shooting AS (
    SELECT
        pgs.game_id,
        pgs.team_id,
        SUM(pgs.fgm)::numeric / NULLIF(SUM(pgs.fga), 0) AS team_fg_pct
    FROM player_game_stats pgs
    GROUP BY pgs.game_id, pgs.team_id
)
SELECT
    CASE
        WHEN wr.days_rest <= 0 THEN 'back_to_back (0 days rest)'
        WHEN wr.days_rest = 1 THEN '1 day rest'
        ELSE '2+ days rest'
    END AS rest_bucket,
    COUNT(*) AS n_games,
    ROUND(AVG(tgs.team_fg_pct), 4) AS avg_team_fg_pct
FROM with_rest wr
JOIN team_game_shooting tgs
    ON tgs.game_id = wr.game_id AND tgs.team_id = wr.team_id
GROUP BY rest_bucket
ORDER BY MIN(wr.days_rest);

-- Finding (example): teams on a back-to-back shoot ~1.5-2.5 percentage
-- points worse from the field than teams with 2+ days of rest.
-- Insight: "days since last game" is a cheap, high-value fatigue feature --
-- worth engineering into both the game-outcome and player decline models.
