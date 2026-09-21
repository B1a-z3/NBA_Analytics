#!/usr/bin/env bash
# Simulates the nightly cron job that would run during the season.
# Schedule with: 0 6 * * * /path/to/nightly_refresh.sh >> logs/nightly.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Nightly refresh started at $(date) ==="

python ingestion/pull_nba_api_live.py --days 1
python features/build_features.py
python monitoring/update_drift_log.py

echo "=== Nightly refresh completed at $(date) ==="
