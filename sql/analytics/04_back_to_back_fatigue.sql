-- ============================================================================
-- Business Question 4: Does playing on zero days rest (a "back-to-back")
-- hurt a team's shooting percentage?
--
-- Why it matters: rest days is one of the highest-signal fatigue features
-- for both the game-outcome model and the player decline model, and is easy
-- to derive purely from the schedule (no tracking data needed).
--
-- Technique: self-join games to itself on team_id to find each team's
-- previous game date, compute days_rest (calendar days since that game),
-- then compare team shooting % in back-to-back vs. rested games. The join
-- key is each team's per-game sequence number (ROW_NUMBER), so "the
-- previous game" is found via a plain equi-join (cur.rn = prev.rn + 1)
-- rather than a correlated subquery -- a first version used a
-- per-row LATERAL subquery ("find the max prev.game_date < cur.game_date"),
-- which forces Postgres to re-scan per row and was measured taking multiple
-- minutes on the full ~53k-row backfill; this equi-join form runs as a
-- single hash/merge join and finishes in under a second on the same data.
--
-- Note on days_rest semantics: a "back-to-back" means games on consecutive
-- calendar days, i.e. days_rest == 1 (game_date minus the PREVIOUS game's
-- date) -- not days_rest == 0, since two games can't happen on the same
-- calendar day. An earlier version of this query bucketed on
-- `days_rest <= 0`, which can never be true, silently producing an empty
-- back-to-back bucket and folding every real back-to-back into "1 day
-- rest" unlabeled. Fixed to bucket on the correct threshold.
-- ============================================================================

WITH team_schedule AS (
    SELECT game_id, game_date, home_team_id AS team_id FROM games WHERE home_score IS NOT NULL
    UNION ALL
    SELECT game_id, game_date, away_team_id AS team_id FROM games WHERE home_score IS NOT NULL
),
numbered_schedule AS (
    SELECT
        game_id, game_date, team_id,
        ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY game_date) AS rn
    FROM team_schedule
),
with_rest AS (
    SELECT
        cur.team_id,
        cur.game_id,
        cur.game_date,
        cur.game_date - prev.game_date AS days_rest
    FROM numbered_schedule cur
    JOIN numbered_schedule prev
        ON prev.team_id = cur.team_id
       AND prev.rn = cur.rn - 1
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
        WHEN wr.days_rest = 1 THEN 'back_to_back (1 day rest)'
        WHEN wr.days_rest = 2 THEN '2 days rest'
        ELSE '3+ days rest'
    END AS rest_bucket,
    COUNT(*) AS n_games,
    ROUND(AVG(tgs.team_fg_pct), 4) AS avg_team_fg_pct
FROM with_rest wr
JOIN team_game_shooting tgs
    ON tgs.game_id = wr.game_id AND tgs.team_id = wr.team_id
GROUP BY rest_bucket
ORDER BY MIN(wr.days_rest);

-- Finding (real result, full nathanlauga/nba-games backfill 2003-2022,
-- ~53k team-game rows):
--   back_to_back (1 day rest):  0.4509 FG%  (n=10,733)
--   2 days rest:                0.4571 FG%  (n=29,336)
--   3+ days rest:                0.4548 FG%  (n=12,947)
-- Back-to-back teams shoot ~0.4-0.6 percentage points worse than teams
-- with 2+ days rest -- a real but modest gap, noticeably smaller than
-- initially guessed before running against actual data (an earlier draft
-- speculated ~1.5-2.5 points here). Interestingly, 2 days of rest slightly
-- outperforms 3+ days, suggesting rest has diminishing (or even slightly
-- reversing, perhaps due to rust) returns past a certain point rather than
-- a simple "more rest is always better" relationship.
-- Insight: back-to-back fatigue has a small, real, measurable effect on
-- shooting efficiency -- worth including as a feature, but on its own it's
-- a minor signal rather than a dominant one, consistent with the small
-- accuracy lift it contributed in models/train_game_outcome_xgb.py.
