-- ============================================================================
-- NBA Analytics Warehouse Schema
-- Postgres. Normalized star-ish schema: dimension tables (teams, players)
-- + fact tables (games, player_game_stats, player_tracking_stats)
-- + supporting tables (schedules/rest, injuries, model predictions, drift log)
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Dimension: teams
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_id         INTEGER PRIMARY KEY,        -- nba_api team id
    abbreviation    VARCHAR(5)  NOT NULL UNIQUE,
    name            VARCHAR(60) NOT NULL,
    conference      VARCHAR(10) CHECK (conference IN ('East', 'West')),
    division        VARCHAR(20),
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Dimension: players
-- team_id nullable + ON DELETE SET NULL: players get traded / teams dissolve
-- rosters; we don't want to lose player history if a team link goes stale.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    player_id       INTEGER PRIMARY KEY,        -- nba_api player id
    full_name       VARCHAR(100) NOT NULL,
    position        VARCHAR(10),
    birth_date      DATE,
    height_inches   SMALLINT,
    weight_lbs      SMALLINT,
    team_id         INTEGER REFERENCES teams(team_id) ON DELETE SET NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);

-- ---------------------------------------------------------------------------
-- Fact: games (one row per game)
-- CHECK constraints guard against bad backfill data (self-play, negative scores)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS games (
    game_id         VARCHAR(20) PRIMARY KEY,    -- nba_api game_id (string, zero-padded)
    game_date       DATE NOT NULL,
    season          VARCHAR(9)  NOT NULL,        -- e.g. '2023-24'
    season_type     VARCHAR(20) NOT NULL DEFAULT 'Regular Season',
    home_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    home_score      SMALLINT,
    away_score      SMALLINT,
    home_win        BOOLEAN GENERATED ALWAYS AS (home_score > away_score) STORED,
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    CONSTRAINT chk_teams_differ CHECK (home_team_id <> away_team_id),
    CONSTRAINT chk_scores_nonneg CHECK (home_score IS NULL OR (home_score >= 0 AND away_score >= 0))
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id, game_date);

-- ---------------------------------------------------------------------------
-- Fact: player_game_stats (one row per player per game — traditional box score)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS player_game_stats (
    id              BIGSERIAL PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    game_id         VARCHAR(20) NOT NULL REFERENCES games(game_id),
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    minutes         NUMERIC(5,2) DEFAULT 0,
    points          SMALLINT DEFAULT 0,
    rebounds        SMALLINT DEFAULT 0,
    offensive_reb   SMALLINT DEFAULT 0,
    defensive_reb   SMALLINT DEFAULT 0,
    assists         SMALLINT DEFAULT 0,
    steals          SMALLINT DEFAULT 0,
    blocks          SMALLINT DEFAULT 0,
    turnovers       SMALLINT DEFAULT 0,
    fouls           SMALLINT DEFAULT 0,
    fgm             SMALLINT DEFAULT 0,
    fga             SMALLINT DEFAULT 0,
    fg3m            SMALLINT DEFAULT 0,
    fg3a            SMALLINT DEFAULT 0,
    ftm             SMALLINT DEFAULT 0,
    fta             SMALLINT DEFAULT 0,
    plus_minus      SMALLINT,
    usage_rate      NUMERIC(5,2),
    started         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_pgs_player ON player_game_stats(player_id, game_id);
CREATE INDEX IF NOT EXISTS idx_pgs_game ON player_game_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_pgs_team ON player_game_stats(team_id);

-- ---------------------------------------------------------------------------
-- Fact: player_tracking_stats (SportVU/second-spectrum style tracking data)
-- One row per player per game — separate from box score since it's a
-- different source/cadence (available current season via nba_api only).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS player_tracking_stats (
    id              BIGSERIAL PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    game_id         VARCHAR(20) NOT NULL REFERENCES games(game_id),
    distance_miles  NUMERIC(5,2),
    avg_speed_mph   NUMERIC(5,2),
    touches         SMALLINT,
    drives          SMALLINT,
    secondary_ast   SMALLINT,
    contested_shots SMALLINT,
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_pts_player ON player_tracking_stats(player_id, game_id);

-- ---------------------------------------------------------------------------
-- Supporting: injury_reports (optional but referenced in modeling layer)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS injury_reports (
    id              BIGSERIAL PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    report_date     DATE NOT NULL,
    status          VARCHAR(20),   -- Out / Doubtful / Questionable / Probable
    description     TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (player_id, report_date)
);

-- ---------------------------------------------------------------------------
-- Model artifacts registry — which model version made which prediction
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id  SERIAL PRIMARY KEY,
    model_name        VARCHAR(50) NOT NULL,   -- 'game_outcome_logreg', 'game_outcome_xgb', 'player_decline'
    version_tag       VARCHAR(30) NOT NULL,
    trained_at        TIMESTAMP NOT NULL DEFAULT now(),
    metrics           JSONB,
    UNIQUE (model_name, version_tag)
);

-- ---------------------------------------------------------------------------
-- Predictions log (for drift monitoring)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS game_predictions (
    id                BIGSERIAL PRIMARY KEY,
    game_id           VARCHAR(20) NOT NULL REFERENCES games(game_id),
    model_version_id  INTEGER NOT NULL REFERENCES model_versions(model_version_id),
    predicted_home_win_prob NUMERIC(5,4) NOT NULL,
    predicted_label   BOOLEAN NOT NULL,
    actual_label      BOOLEAN,               -- filled in after game completes
    predicted_at      TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (game_id, model_version_id)
);

CREATE TABLE IF NOT EXISTS player_decline_predictions (
    id                BIGSERIAL PRIMARY KEY,
    player_id         INTEGER NOT NULL REFERENCES players(player_id),
    game_id           VARCHAR(20) NOT NULL REFERENCES games(game_id),
    model_version_id  INTEGER NOT NULL REFERENCES model_versions(model_version_id),
    decline_risk_score NUMERIC(5,4) NOT NULL,
    decline_flag      BOOLEAN NOT NULL,
    predicted_at      TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (player_id, game_id, model_version_id)
);

-- ---------------------------------------------------------------------------
-- Drift / accuracy monitoring log — one row per week per model
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_monitoring_log (
    id                BIGSERIAL PRIMARY KEY,
    model_version_id  INTEGER NOT NULL REFERENCES model_versions(model_version_id),
    week_start        DATE NOT NULL,
    n_predictions     INTEGER NOT NULL,
    accuracy          NUMERIC(5,4),
    log_loss          NUMERIC(7,4),
    brier_score       NUMERIC(7,4),
    notes             TEXT,
    logged_at         TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (model_version_id, week_start)
);
