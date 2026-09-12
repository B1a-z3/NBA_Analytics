"""
Logs model predictions for upcoming/recent games into game_predictions,
BEFORE the outcome is known where possible (in practice here: for games
already in the warehouse, since this is a portfolio simulation rather than
a live in-season deployment). This is what update_drift_log.py later scores
against actual outcomes.

Registers a model_versions row if one doesn't exist yet, then predicts for
all games in team_game_features that don't already have a logged
prediction for that model version.

Run:
    python monitoring/log_predictions.py --model game_outcome_xgb --version v1
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"


def get_or_create_model_version(engine, model_name: str, version_tag: str) -> int:
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT model_version_id FROM model_versions
            WHERE model_name = :model_name AND version_tag = :version_tag
        """), {"model_name": model_name, "version_tag": version_tag}).fetchone()
        if row:
            return row[0]
        result = conn.execute(text("""
            INSERT INTO model_versions (model_name, version_tag)
            VALUES (:model_name, :version_tag)
            RETURNING model_version_id
        """), {"model_name": model_name, "version_tag": version_tag})
        return result.fetchone()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="game_outcome_xgb")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    engine = get_engine()
    model_version_id = get_or_create_model_version(engine, args.model, args.version)

    bundle = joblib.load(MODEL_DIR / f"{args.model}.joblib")
    model, features = bundle["model"], bundle["features"]

    team_feats = pd.read_sql("SELECT * FROM team_game_features", engine)
    opp_rest = team_feats[["game_id", "team_id", "days_rest", "is_back_to_back"]].rename(
        columns={"team_id": "opponent_id", "days_rest": "opponent_days_rest",
                 "is_back_to_back": "opponent_is_back_to_back"}
    )
    df = team_feats.merge(opp_rest, on=["game_id", "opponent_id"], how="left")
    df["rest_advantage"] = df["days_rest"].fillna(0) - df["opponent_days_rest"].fillna(0)
    df = df[df["is_home"] == True]  # noqa: E712 -- one prediction per game, from home team's perspective
    df = df.dropna(subset=features)

    if df.empty:
        print("No eligible games to predict.")
        return

    X = df[features]
    proba = model.predict_proba(X)[:, 1]
    df["predicted_home_win_prob"] = proba
    df["predicted_label"] = proba >= 0.5

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO game_predictions
                    (game_id, model_version_id, predicted_home_win_prob, predicted_label)
                VALUES (:game_id, :model_version_id, :prob, :label)
                ON CONFLICT (game_id, model_version_id) DO NOTHING
            """), {
                "game_id": row["game_id"], "model_version_id": model_version_id,
                "prob": float(row["predicted_home_win_prob"]), "label": bool(row["predicted_label"]),
            })

    print(f"Logged {len(df)} predictions for model_version_id={model_version_id}")


if __name__ == "__main__":
    main()
