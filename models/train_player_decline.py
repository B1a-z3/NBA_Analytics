"""
Player decline-risk model: binary classifier flagging whether a player's
NEXT game will show a meaningful efficiency drop (TS% at least 5 points
below their season-to-date baseline), using rolling performance + load +
fatigue + age features available BEFORE that next game is played.

Label construction: decline_flag = 1 if next game's ts_pct is more than
0.05 below the player's season_to_date_ts_pct at the time (mirrors the
threshold used in sql/analytics/02_player_efficiency_decline.sql).

Run:
    python models/train_player_decline.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "rolling_ts_pct", "season_to_date_ts_pct", "rolling_minutes",
    "rolling_load_index", "days_rest", "is_back_to_back", "age_years",
]
DECLINE_THRESHOLD = 0.05


def build_labeled_dataset(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM player_game_features ORDER BY player_id, game_date", engine)
    df = df.dropna(subset=["ts_pct", "season_to_date_ts_pct"])

    # label = does the player's OWN ts_pct this game fall > threshold below
    # their season-to-date baseline as of that game (features are already
    # lagged, so this is a same-row label using pre-game features -> in-game outcome)
    df["decline_flag"] = (
        (df["season_to_date_ts_pct"] - df["ts_pct"]) > DECLINE_THRESHOLD
    ).astype(int)

    df = df.dropna(subset=FEATURE_COLS)
    return df


def main():
    engine = get_engine()
    df = build_labeled_dataset(engine)
    if df.empty:
        print("No feature data found -- run features/build_features.py first.")
        return

    X = df[FEATURE_COLS]
    y = df["decline_flag"]

    print(f"Decline rate in dataset: {y.mean():.3f} ({y.sum()} / {len(y)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),  # class imbalance
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }

    print("Player decline-risk model:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    joblib.dump({"model": model, "features": FEATURE_COLS}, MODEL_DIR / "player_decline.joblib")
    print(f"\nSaved model to {MODEL_DIR / 'player_decline.joblib'}")

    return metrics


if __name__ == "__main__":
    main()
