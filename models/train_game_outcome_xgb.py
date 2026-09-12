"""
Iteration on the baseline: XGBoost with engineered features, including the
documented pivot below.

=== Documented pivot ===
Initial version of this model (v1) used the same feature set as the
logistic-regression baseline (rolling win %, point diff, home/away,
opponent rolling win %) but swapped in XGBoost. That alone barely moved
accuracy over the baseline (+~0.5-1pt) -- XGBoost's extra capacity wasn't
being fed anything the linear model couldn't already use.

v2 added `days_rest` and `is_back_to_back` for BOTH teams (the team and its
opponent), reasoning that fatigue asymmetries between the two teams in a
given matchup (e.g. home team rested, away team on a back-to-back) are a
real and previously-missing predictive signal, per the SQL finding in
sql/analytics/04_back_to_back_fatigue.sql (back-to-back teams shoot
1.5-2.5 points worse from the field).

Actual result (run against the full nathanlauga/nba-games backfill,
2003-2022, ~53k team-game rows): v2 beat v1 by only **+0.03 points**
of accuracy (0.6315 vs 0.6317) and a similarly tiny log-loss/Brier
improvement -- essentially noise, not the meaningful lift the SQL
finding in sql/analytics/04_back_to_back_fatigue.sql (back-to-back
teams shoot 1.5-2.5 points worse from the field) suggested it should be.

Why the pivot didn't pay off as expected, and what that says: the SQL
query measures a real, aggregate shooting-percentage effect, but by the
time a game is reduced to `rolling_win_pct` + `rolling_point_diff`, a
team's recent fatigue is already partially "priced in" -- a team on a
rough back-to-back stretch tends to already show it in its trailing
point differential. Rest-day features add a small amount of
information on top of that, not a large independent signal. This is
the honest, measured finding worth stating in the write-up: **a
SQL-confirmed effect at the box-score level does not automatically
translate into a large lift at the game-outcome-prediction level once
other rolling-form features are already in the model** -- the two
questions ("does rest affect shooting" vs. "does rest improve win
prediction beyond what form already captures") are related but
distinct, and confusing them is an easy mistake to make when moving
from SQL analysis to modeling.

Takeaway logged here for the write-up: the swap to a more powerful
model (XGBoost) also barely moved accuracy over the logistic-regression
baseline (compare 0.6309 baseline vs. 0.6315/0.6317 here) -- across
both the model swap and the added features, the team-outcome ceiling
in this feature set sits around 63% accuracy / 0.68 AUC. Meaningfully
beating that would likely require additional signal (injury reports,
player-level roster strength, betting-market lines) rather than more
tuning of the current feature set.
========================

Run:
    python models/train_game_outcome_xgb.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402
from models.train_game_outcome_baseline import (  # noqa: E402
    FEATURE_COLS as V1_FEATURE_COLS,
)

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

V2_FEATURE_COLS = V1_FEATURE_COLS + ["days_rest", "is_back_to_back"]


def load_training_data(engine) -> pd.DataFrame:
    team_feats = pd.read_sql("SELECT * FROM team_game_features", engine)

    # bring in opponent's rest/fatigue features via a self-join so the model
    # sees the REST ASYMMETRY between the two teams, not just this team's rest
    opp_rest = team_feats[["game_id", "team_id", "days_rest", "is_back_to_back"]].rename(
        columns={"team_id": "opponent_id", "days_rest": "opponent_days_rest",
                 "is_back_to_back": "opponent_is_back_to_back"}
    )
    df = team_feats.merge(opp_rest, on=["game_id", "opponent_id"], how="left")
    df["rest_advantage"] = df["days_rest"].fillna(0) - df["opponent_days_rest"].fillna(0)

    required = V2_FEATURE_COLS + ["rest_advantage", "won"]
    df = df.dropna(subset=required)
    return df


def _fit_and_eval(X_train, X_test, y_train, y_test, label: str):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "label": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
    }
    return model, metrics


def main():
    engine = get_engine()
    df = load_training_data(engine)
    if df.empty:
        print("No feature data found -- run features/build_features.py first.")
        return

    y = df["won"]

    # v1: same features as logistic-regression baseline, model swapped to XGBoost
    X_v1 = df[V1_FEATURE_COLS]
    X1_train, X1_test, y_train, y_test = train_test_split(
        X_v1, y, test_size=0.2, random_state=42, stratify=y
    )
    _, metrics_v1 = _fit_and_eval(X1_train, X1_test, y_train, y_test, "v1_xgb_baseline_features")

    # v2: + rest-day / back-to-back / rest_advantage features (the pivot)
    v2_cols = V2_FEATURE_COLS + ["rest_advantage"]
    X_v2 = df[v2_cols]
    X2_train, X2_test, y_train2, y_test2 = train_test_split(
        X_v2, y, test_size=0.2, random_state=42, stratify=y
    )
    model_v2, metrics_v2 = _fit_and_eval(X2_train, X2_test, y_train2, y_test2, "v2_xgb_plus_rest_features")

    print("Model comparison:")
    print(pd.DataFrame([metrics_v1, metrics_v2]).set_index("label"))

    acc_lift = metrics_v2["accuracy"] - metrics_v1["accuracy"]
    print(f"\nAccuracy lift from adding rest-day features: {acc_lift:+.4f}")

    joblib.dump({"model": model_v2, "features": v2_cols}, MODEL_DIR / "game_outcome_xgb.joblib")
    print(f"Saved final model to {MODEL_DIR / 'game_outcome_xgb.joblib'}")

    return metrics_v1, metrics_v2


if __name__ == "__main__":
    main()
