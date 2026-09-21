# NBA Player Performance & Game Outcome Prediction System

**Core question:** Can team form, player load/fatigue signals, and matchup
context predict game outcomes and flag player performance decline risk?

Two connected sub-problems: team-level game-outcome prediction, and
player-level decline/fatigue monitoring — sharing one data pipeline.

## Architecture

```
[Kaggle bulk] -> [Raw storage] -> [Validation/cleaning] -> [Postgres warehouse]
        -> [SQL analytics layer] -> [Feature engineering] -> [Model training]
        -> [Static parquet snapshot] -> [Streamlit dashboard]
```

## Project layout

```
db/schema.sql                  Postgres DDL (5 core tables + supporting tables, FKs, constraints)
ingestion/
  load_kaggle_backfill.py      Backfill: 5-10 seasons of box scores from Kaggle CSV export
  validation.py                Validation/cleaning layer for ingestion
sql/analytics/                 6 documented business-question SQL queries (see sql/analytics/README.md)
features/build_features.py     Feature engineering -> team_game_features, player_game_features
models/
  train_game_outcome_baseline.py   Logistic regression baseline
  train_game_outcome_xgb.py        XGBoost iteration + documented pivot (rest-day features)
  train_player_decline.py          Player decline-risk classifier
dashboard/app.py                Streamlit: team trends, player risk flags, model accuracy
dashboard/data/                Parquet snapshots the deployed dashboard reads (no database)
monitoring/
  log_predictions.py            Logs predictions for later scoring
  update_drift_log.py           Weekly accuracy/log-loss/Brier scoring + drift alerting
scripts/
  export_dashboard_data.py      Exports the dashboard's parquet snapshots from a local Postgres
  run_analytics.py               Runs all 6 SQL analytics files against the warehouse
tests/                          Unit tests for validation + feature logic (no DB required)
```

## Deploy (GitHub + Streamlit Community Cloud)

The dashboard needs no database: it reads the committed parquet snapshots in
`dashboard/data/`. Push the repo to GitHub, then on
[share.streamlit.io](https://share.streamlit.io) create an app with main file
`dashboard/app.py`. No secrets are required.

Run locally:
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Rebuilding the data (optional, local only)

The rest of the pipeline (backfill, features, models, monitoring, SQL
analytics) runs against a *local* Postgres in Docker and is only needed to
regenerate the snapshot. There is no live data ingestion.

1. `cp .env.example .env && docker compose up -d postgres`, then
   `pip install -r requirements.txt`
2. Put the [`nathanlauga/nba-games`](https://www.kaggle.com/datasets/nathanlauga/nba-games)
   CSVs in `data/raw/kaggle/` and run `python ingestion/load_kaggle_backfill.py`
   (team conferences come from `ingestion/reference/team_conferences.csv`;
   player bio fields such as birth date are not filled by this step)
3. `python scripts/run_analytics.py`
4. `python features/build_features.py`
5. `python models/train_game_outcome_baseline.py`, `train_game_outcome_xgb.py`, `train_player_decline.py`
6. `python monitoring/log_predictions.py --model game_outcome_xgb --version v1`, then `python monitoring/update_drift_log.py`
7. `python features/build_dashboard_summaries.py`
8. `python scripts/export_dashboard_data.py`, then commit the updated `dashboard/data/`

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
drops more than 5 points below the trailing 4-week average. The Streamlit dashboard's "Model
Accuracy Over Time" tab shows this log.
