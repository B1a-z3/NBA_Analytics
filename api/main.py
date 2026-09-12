"""
FastAPI serving layer.

Endpoints:
    POST /predict/game-outcome        -> home win probability
    POST /predict/player-decline-risk -> decline risk score/flag
    GET  /health
    GET  /monitoring/accuracy         -> weekly accuracy history (drift log)

Run:
    uvicorn api.main:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"

app = FastAPI(
    title="NBA Player Performance & Game Outcome Prediction API",
    description="Serves game-outcome and player-decline-risk predictions.",
    version="1.0.0",
)

_models = {}


def _load_model(name: str):
    if name not in _models:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Model '{name}' not trained yet: {path} missing")
        _models[name] = joblib.load(path)
    return _models[name]


class GameOutcomeRequest(BaseModel):
    rolling_win_pct: float = Field(..., ge=0, le=1, description="Team's rolling 10-game win %")
    rolling_point_diff: float = Field(..., description="Team's rolling 10-game avg point differential")
    is_home: bool = Field(..., description="Is this team the home team")
    opponent_rolling_win_pct: float = Field(..., ge=0, le=1)
    days_rest: Optional[float] = Field(None, description="Days since team's last game")
    is_back_to_back: Optional[int] = Field(None, ge=0, le=1)
    rest_advantage: Optional[float] = Field(None, description="This team's rest days minus opponent's")


class GameOutcomeResponse(BaseModel):
    home_win_probability: float
    predicted_home_win: bool
    model_used: str


class PlayerDeclineRequest(BaseModel):
    rolling_ts_pct: float
    season_to_date_ts_pct: float
    rolling_minutes: float
    rolling_load_index: float
    days_rest: float
    is_back_to_back: int
    age_years: float


class PlayerDeclineResponse(BaseModel):
    decline_risk_score: float
    decline_flag: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/game-outcome", response_model=GameOutcomeResponse)
def predict_game_outcome(req: GameOutcomeRequest):
    # Prefer the XGBoost model (has rest-day features); fall back to logreg baseline
    xgb_path = MODEL_DIR / "game_outcome_xgb.joblib"
    if xgb_path.exists() and req.days_rest is not None:
        bundle = _load_model("game_outcome_xgb")
        model, features = bundle["model"], bundle["features"]
        row = {
            "rolling_win_pct": req.rolling_win_pct,
            "rolling_point_diff": req.rolling_point_diff,
            "is_home": req.is_home,
            "opponent_rolling_win_pct": req.opponent_rolling_win_pct,
            "days_rest": req.days_rest,
            "is_back_to_back": req.is_back_to_back or 0,
            "rest_advantage": req.rest_advantage or 0,
        }
        X = pd.DataFrame([{k: row[k] for k in features}])
        proba = float(model.predict_proba(X)[0, 1])
        model_used = "game_outcome_xgb"
    else:
        pipeline = _load_model("game_outcome_logreg")
        X = pd.DataFrame([{
            "rolling_win_pct": req.rolling_win_pct,
            "rolling_point_diff": req.rolling_point_diff,
            "is_home": req.is_home,
            "opponent_rolling_win_pct": req.opponent_rolling_win_pct,
        }])
        proba = float(pipeline.predict_proba(X)[0, 1])
        model_used = "game_outcome_logreg"

    return GameOutcomeResponse(
        home_win_probability=round(proba, 4),
        predicted_home_win=proba >= 0.5,
        model_used=model_used,
    )


@app.post("/predict/player-decline-risk", response_model=PlayerDeclineResponse)
def predict_player_decline(req: PlayerDeclineRequest):
    bundle = _load_model("player_decline")
    model, features = bundle["model"], bundle["features"]
    X = pd.DataFrame([req.dict()])[features]
    proba = float(model.predict_proba(X)[0, 1])
    return PlayerDeclineResponse(
        decline_risk_score=round(proba, 4),
        decline_flag=proba >= 0.5,
    )


@app.get("/monitoring/accuracy")
def monitoring_accuracy(model_name: Optional[str] = None):
    engine = get_engine()
    query = """
        SELECT mv.model_name, mv.version_tag, mml.week_start, mml.n_predictions,
               mml.accuracy, mml.log_loss, mml.brier_score
        FROM model_monitoring_log mml
        JOIN model_versions mv ON mv.model_version_id = mml.model_version_id
    """
    if model_name:
        query += " WHERE mv.model_name = %(model_name)s"
    query += " ORDER BY mml.week_start"
    df = pd.read_sql(query, engine, params={"model_name": model_name} if model_name else None)
    return df.to_dict(orient="records")
