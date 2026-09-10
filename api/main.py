from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

# ── load artifacts ──────────────────────────────────────────────────────────
try:
    model_apc  = joblib.load(MODELS_DIR / "model_apc.pkl")
    model_pdp  = joblib.load(MODELS_DIR / "model_pdp.pkl")
    scaler     = joblib.load(MODELS_DIR / "scaler_pdp.pkl")
    with open(DATA_DIR / "features.json") as f:
        feature_names = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# ── app ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nigeria Election Predictor API",
    description="Predicts 2023 state-level vote share for APC, PDP, and LP",
    version="1.0.0"
)

# ── schema ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    Zone_Encoded:              float
    Turnout_pct:               float
    Gov_Party_Encoded:         float
    Incumbent_Encoded:         float
    Gov_Aligns_Incumbent:      float
    National_Security_Incidents: float
    Prev_APC_pct:              float
    Prev_PDP_pct:              float
    Prev_Third_Party_pct:      float
    Third_Party_pct:           float
    State_Avg_APC_pct:         float
    State_Avg_PDP_pct:         float
    Zone_Year_APC_pct:         float
    Zone_Year_PDP_pct:         float

class PredictResponse(BaseModel):
    apc_pct:    float
    pdp_pct:    float
    lp_pct:     float
    winner:     str

# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Nigeria Election Predictor API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Map request fields to feature names (% signs not valid in JSON keys)
        input_dict = {
            "Zone_Encoded":               request.Zone_Encoded,
            "Turnout_%":                  request.Turnout_pct,
            "Gov_Party_Encoded":          request.Gov_Party_Encoded,
            "Incumbent_Encoded":          request.Incumbent_Encoded,
            "Gov_Aligns_Incumbent":       request.Gov_Aligns_Incumbent,
            "National_Security_Incidents":request.National_Security_Incidents,
            "Prev_APC_%":                 request.Prev_APC_pct,
            "Prev_PDP_%":                 request.Prev_PDP_pct,
            "Prev_Third_Party_%":         request.Prev_Third_Party_pct,
            "Third_Party_%":              request.Third_Party_pct,
            "State_Avg_APC_%":            request.State_Avg_APC_pct,
            "State_Avg_PDP_%":            request.State_Avg_PDP_pct,
            "Zone_Year_APC_%":            request.Zone_Year_APC_pct,
            "Zone_Year_PDP_%":            request.Zone_Year_PDP_pct,
        }

        input_df = pd.DataFrame([input_dict])[feature_names]

        apc_pct = float(model_apc.predict(input_df)[0])
        pdp_pct = float(model_pdp.predict(scaler.transform(input_df))[0])
        lp_pct  = float(max(0.0, 100.0 - apc_pct - pdp_pct))

        scores  = {"APC": apc_pct, "PDP": pdp_pct, "LP": lp_pct}
        winner  = max(scores, key=scores.get)

        return PredictResponse(
            apc_pct=round(apc_pct, 2),
            pdp_pct=round(pdp_pct, 2),
            lp_pct=round(lp_pct,  2),
            winner=winner
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
