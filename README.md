# NBA Player Performance & Game Outcome Prediction System

**Core question:** Can team form, player load/fatigue signals, and matchup
context predict game outcomes and flag player performance decline risk?

Two connected sub-problems: team-level game-outcome prediction, and
player-level decline/fatigue monitoring — sharing one data pipeline.

## Architecture

```
[nba_api + Kaggle bulk] -> [Raw storage] -> [Validation/cleaning] -> [Postgres warehouse]
        -> [SQL analytics layer] -> [Feature engineering] -> [Model training]
        -> [FastAPI serving] -> [Streamlit dashboard] -> [Monitoring/drift logging]
```

## Project layout

```
db/schema.sql                  Postgres DDL (5 core tables + supporting tables, FKs, constraints)
ingestion/
  load_kaggle_backfill.py      Backfill: 5-10 seasons of box scores from Kaggle CSV export
  pull_nba_api_live.py         Live/current season pull via nba_api (games, box scores, tracking)
  validation.py                Validation/cleaning layer shared by both ingestion sources
sql/analytics/                 6 documented business-question SQL queries (see sql/analytics/README.md)
features/build_features.py     Feature engineering -> team_game_features, player_game_features
models/
  train_game_outcome_baseline.py   Logistic regression baseline
  train_game_outcome_xgb.py        XGBoost iteration + documented pivot (rest-day features)
  train_player_decline.py          Player decline-risk classifier
api/main.py                    FastAPI: /predict/game-outcome, /predict/player-decline-risk
dashboard/app.py                Streamlit: team trends, player risk flags, model accuracy
monitoring/
  log_predictions.py            Logs predictions for later scoring
  update_drift_log.py           Weekly accuracy/log-loss/Brier scoring + drift alerting
scripts/
  nightly_refresh.sh            Simulated nightly cron (live pull -> features -> drift log)
  run_analytics.py               Runs all 6 SQL analytics files against the warehouse
tests/                          Unit tests for validation + feature logic (no DB required)
```

## Setup

1. **Start Postgres** (schema auto-applies via `db/schema.sql` on first boot):
   ```powershell
   # PowerShell (Windows)
   Copy-Item .env.example .env
   docker compose up -d postgres
   ```
   ```bash
   # macOS/Linux
   cp .env.example .env
   docker compose up -d postgres
   ```

2. **Install Python deps:**
   ```powershell
   # PowerShell (Windows)
   python -m venv venv
   venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   ```bash
   # macOS/Linux
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Backfill historical data** — download the
   [`nathanlauga/nba-games`](https://www.kaggle.com/datasets/nathanlauga/nba-games)
   Kaggle dataset (`games.csv`, `games_details.csv`, `players.csv`, `teams.csv`)
   into `data/raw/kaggle/`, then:
   ```bash
   python ingestion/load_kaggle_backfill.py
   ```
   Note: that export's `teams.csv` has no conference/division columns (filled in
   from the static [`ingestion/reference/team_conferences.csv`](ingestion/reference/team_conferences.csv)
   lookup) and its `players.csv` has no bio data, so `players.position`/`birth_date`
   land NULL from this step alone — run the enrichment step below to fill them in.

3b. **Enrich player bio data** (birth_date/position/height/weight, needed for
   the aging-curve query and the decline model's `age_years` feature):
   ```bash
   python ingestion/enrich_player_bio.py --limit 50   # smoke test first
   python ingestion/enrich_player_bio.py              # full run (~0.6s/player via nba_api)
   ```

4. **Pull recent/live data via nba_api:**
   ```bash
   python ingestion/pull_nba_api_live.py --days 3
   ```

5. **Run the SQL analytics layer** (sanity-check the warehouse + generate real findings):
   ```bash
   python scripts/run_analytics.py
   ```

6. **Build features:**
   ```bash
   python features/build_features.py
   ```

7. **Train models:**
   ```bash
   python models/train_game_outcome_baseline.py
   python models/train_game_outcome_xgb.py
   python models/train_player_decline.py
   ```

8. **Log predictions + monitoring** (simulates ongoing weekly tracking):
   ```bash
   python monitoring/log_predictions.py --model game_outcome_xgb --version v1
   python monitoring/update_drift_log.py
   ```

9. **Serve the API:**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

10. **Run the dashboard:**
    ```bash
    streamlit run dashboard/app.py
    ```

Or run everything containerized: `docker compose up --build` (after backfill +
training have been run at least once locally, since model artifacts and the
Kaggle bulk data aren't baked into the images).

## Tests

```bash
pytest tests/ -v
```
Covers validation rules (dedupe, FK-sanity checks, stat-consistency checks)
and feature-engineering correctness (no look-ahead leakage, back-to-back
detection) without requiring a live database.

## Modeling notes

- **Baseline:** logistic regression on team-form features only (rolling
  win %, point differential, home/away, opponent form).
- **Iteration + documented pivot:** the first XGBoost pass reused the
  baseline's features and barely beat logistic regression. Adding **rest-day
  asymmetry between the two teams** (`days_rest`, `is_back_to_back`,
  `rest_advantage`) — motivated directly by the SQL finding in
  [`sql/analytics/04_back_to_back_fatigue.sql`](sql/analytics/04_back_to_back_fatigue.sql)
  that back-to-back teams shoot meaningfully worse — produced the real
  accuracy lift. See the docstring in
  [`models/train_game_outcome_xgb.py`](models/train_game_outcome_xgb.py) for
  the full writeup and printed before/after metrics.
- **Player decline model:** XGBoost classifier flagging a >5-point TS% drop
  vs. season baseline, using rolling performance, load index, rest days, and
  age as features.

## Monitoring / drift

`monitoring/update_drift_log.py` backfills actual outcomes onto logged
predictions once games complete, computes weekly accuracy/log-loss/Brier
score per model version, and flags drift when the latest week's accuracy
drops more than 5 points below the trailing 4-week average. Both the
`/monitoring/accuracy` API endpoint and the Streamlit dashboard's "Model
Accuracy Over Time" tab read from this log.
