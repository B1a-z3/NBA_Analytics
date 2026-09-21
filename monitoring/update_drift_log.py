"""
Weekly drift/accuracy monitoring job. Compares logged game_predictions
(made before games were played) against actual outcomes (now known, since
the game happened), computes weekly accuracy/log-loss/Brier score per
model version, and writes one row per (model, week) into
model_monitoring_log. The Streamlit dashboard and the /monitoring/accuracy
API endpoint both read from this table.

Run:
    python monitoring/update_drift_log.py
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine, get_session  # noqa: E402


def backfill_actual_labels(engine):
    """Fill in actual_label on game_predictions for games that have now
    completed (were pending at prediction time)."""
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            UPDATE game_predictions gp
            SET actual_label = g.home_win
            FROM games g
            WHERE gp.game_id = g.game_id
              AND gp.actual_label IS NULL
              AND g.home_score IS NOT NULL
        """)


def compute_weekly_metrics(engine) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT gp.model_version_id, g.game_date,
               gp.predicted_home_win_prob, gp.predicted_label, gp.actual_label
        FROM game_predictions gp
        JOIN games g ON g.game_id = gp.game_id
        WHERE gp.actual_label IS NOT NULL
    """, engine)

    if df.empty:
        return pd.DataFrame()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["week_start"] = df["game_date"] - pd.to_timedelta(df["game_date"].dt.dayofweek, unit="D")

    rows = []
    for (model_version_id, week_start), grp in df.groupby(["model_version_id", "week_start"]):
        y_true = grp["actual_label"].astype(int)
        y_pred = grp["predicted_label"].astype(int)
        y_proba = grp["predicted_home_win_prob"].clip(1e-6, 1 - 1e-6)
        rows.append({
            "model_version_id": model_version_id,
            "week_start": week_start.date(),
            "n_predictions": len(grp),
            "accuracy": accuracy_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
            "brier_score": brier_score_loss(y_true, y_proba),
        })
    return pd.DataFrame(rows)


def write_monitoring_log(engine, metrics_df: pd.DataFrame):
    if metrics_df.empty:
        print("No completed predictions to score yet.")
        return
    from sqlalchemy import text
    with engine.begin() as conn:
        for _, row in metrics_df.iterrows():
            conn.execute(text("""
                INSERT INTO model_monitoring_log
                    (model_version_id, week_start, n_predictions, accuracy, log_loss, brier_score)
                VALUES (:model_version_id, :week_start, :n_predictions, :accuracy, :log_loss, :brier_score)
                ON CONFLICT (model_version_id, week_start) DO UPDATE SET
                    n_predictions = EXCLUDED.n_predictions,
                    accuracy = EXCLUDED.accuracy,
                    log_loss = EXCLUDED.log_loss,
                    brier_score = EXCLUDED.brier_score,
                    logged_at = now()
            """), row.to_dict())
    print(f"Wrote {len(metrics_df)} weekly monitoring rows.")


def check_for_drift(metrics_df: pd.DataFrame, accuracy_drop_threshold: float = 0.05):
    """Simple drift heuristic: flag if the most recent week's accuracy is
    more than `accuracy_drop_threshold` below the trailing 4-week average."""
    alerts = []
    for model_version_id, grp in metrics_df.groupby("model_version_id"):
        grp = grp.sort_values("week_start")
        if len(grp) < 2:
            continue
        trailing_avg = grp["accuracy"].iloc[:-1].tail(4).mean()
        latest = grp["accuracy"].iloc[-1]
        if trailing_avg - latest > accuracy_drop_threshold:
            alerts.append({
                "model_version_id": model_version_id,
                "latest_accuracy": latest,
                "trailing_avg_accuracy": trailing_avg,
                "drop": trailing_avg - latest,
            })
    return alerts


def main():
    engine = get_engine()
    backfill_actual_labels(engine)
    metrics_df = compute_weekly_metrics(engine)
    write_monitoring_log(engine, metrics_df)

    alerts = check_for_drift(metrics_df) if not metrics_df.empty else []
    if alerts:
        print("\n*** DRIFT ALERTS ***")
        for a in alerts:
            print(f"  model_version_id={a['model_version_id']}: "
                  f"latest={a['latest_accuracy']:.3f} vs trailing_avg={a['trailing_avg_accuracy']:.3f} "
                  f"(drop={a['drop']:.3f})")
    else:
        print("No drift detected.")


if __name__ == "__main__":
    main()
