"""
Baseline model: logistic regression predicting home team win/loss from
team-form features only (rolling win %, point diff, home/away, opponent
rolling win %). This intentionally does NOT include rest-day / fatigue
features -- see train_game_outcome_xgb.py for the documented pivot where
adding a rest-day feature improved accuracy.

Run:
    python models/train_game_outcome_baseline.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "rolling_win_pct", "rolling_point_diff", "is_home", "opponent_rolling_win_pct",
]


def load_training_data(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM team_game_features", engine)
    df = df.dropna(subset=FEATURE_COLS + ["won"])
    return df


def main():
    engine = get_engine()
    df = load_training_data(engine)
    if df.empty:
        print("No feature data found -- run features/build_features.py first.")
        return

    X = df[FEATURE_COLS]
    y = df["won"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print("Baseline logistic regression (team-form-only features):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    joblib.dump(pipeline, MODEL_DIR / "game_outcome_logreg.joblib")
    print(f"\nSaved model to {MODEL_DIR / 'game_outcome_logreg.joblib'}")
    return metrics


if __name__ == "__main__":
    main()
