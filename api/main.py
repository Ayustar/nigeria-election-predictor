from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
MODELS_V2   = BASE_DIR / "models_v2"
DATA_DIR    = BASE_DIR / "data"

# ── Load v1 artifacts ─────────────────────────────────────────────────────
try:
    model_apc_v1  = joblib.load(MODELS_DIR / "model_apc.pkl")
    model_pdp_v1  = joblib.load(MODELS_DIR / "model_pdp.pkl")
    scaler_v1     = joblib.load(MODELS_DIR / "scaler_pdp.pkl")
    with open(DATA_DIR / "features.json") as f:
        features_v1 = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load v1 artifacts: {e}")

# ── Load v2 artifacts ─────────────────────────────────────────────────────
try:
    model_apc_v2 = joblib.load(MODELS_V2 / "model_apc_v2.pkl")
    model_adc_v2 = joblib.load(MODELS_V2 / "model_adc_v2.pkl")
    model_ndc_v2 = joblib.load(MODELS_V2 / "model_ndc_v2.pkl")
    with open(DATA_DIR / "features_apc_adc_v2.json") as f:
        features_apc_adc_v2 = json.load(f)
    with open(DATA_DIR / "features_ndc_v2.json") as f:
        features_ndc_v2 = json.load(f)
    with open(MODELS_V2 / "apm_weights.json") as f:
        apm_config = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load v2 artifacts: {e}")

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nigeria Election Predictor API",
    description="v1: 2023 predictions | v2: 2027 predictions",
    version="2.0.0"
)

# ── v1 Schema ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    Zone_Encoded:                float
    Turnout_pct:                 float
    Gov_Party_Encoded:           float
    Incumbent_Encoded:           float
    Gov_Aligns_Incumbent:        float
    National_Security_Incidents: float
    Prev_APC_pct:                float
    Prev_PDP_pct:                float
    Prev_Third_Party_pct:        float
    Third_Party_pct:             float
    State_Avg_APC_pct:           float
    State_Avg_PDP_pct:           float
    Zone_Year_APC_pct:           float
    Zone_Year_PDP_pct:           float

class PredictResponse(BaseModel):
    apc_pct:  float
    pdp_pct:  float
    lp_pct:   float
    winner:   str

# ── v2 Schema ─────────────────────────────────────────────────────────────
class PredictRequest2027(BaseModel):
    State:                         str
    Zone_Encoded:                  float
    Turnout_pct:                   float
    Gov_Encoded:                   float
    Incumbent_Encoded:             float
    Gov_Aligns_Incumbent:          float
    National_Security_Incidents:   float
    Prev_APC_pct:                  float
    Prev_ADC_pct:                  float
    Prev_NDC_pct:                  float
    State_Avg_APC:                 float
    State_Avg_ADC:                 float
    State_Avg_NDC:                 float
    Zone_Avg_APC:                  float
    Zone_Avg_ADC:                  float
    Zone_Avg_NDC:                  float
    Gov_Is_Presidential_Candidate: float
    Federal_Admin_Control:         float
    Geopolitical_Zone:             str

class PredictResponse2027(BaseModel):
    apc_pct:  float
    adc_pct:  float
    ndc_pct:  float
    apm_pct:  float
    winner:   str

# ── APM rule-based score ──────────────────────────────────────────────────
def compute_apm(state: str, zone: str) -> float:
    overrides = apm_config['state_overrides']
    weights   = apm_config['zone_weights']
    if state in overrides:
        return overrides[state]
    return weights.get(zone, 2.0)

# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Nigeria Election Predictor API v2.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# v1 endpoint
@app.post("/predict", response_model=PredictResponse)
def predict_v1(request: PredictRequest):
    try:
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
        input_df = pd.DataFrame([input_dict])[features_v1]
        apc_pct  = float(model_apc_v1.predict(input_df)[0])
        pdp_pct  = float(model_pdp_v1.predict(scaler_v1.transform(input_df))[0])
        lp_pct   = float(max(0.0, 100.0 - apc_pct - pdp_pct))
        scores   = {"APC": apc_pct, "PDP": pdp_pct, "LP": lp_pct}
        winner   = max(scores, key=scores.get)
        return PredictResponse(
            apc_pct=round(apc_pct, 2),
            pdp_pct=round(pdp_pct, 2),
            lp_pct=round(lp_pct,  2),
            winner=winner
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# v2 endpoint
@app.post("/predict_2027", response_model=PredictResponse2027)
def predict_v2(request: PredictRequest2027):
    try:
        input_dict = {
            "Zone_Encoded":                  request.Zone_Encoded,
            "Turnout_%":                     request.Turnout_pct,
            "Gov_Encoded":                   request.Gov_Encoded,
            "Incumbent_Encoded":             request.Incumbent_Encoded,
            "Gov_Aligns_Incumbent":          request.Gov_Aligns_Incumbent,
            "National_Security_Incidents":   request.National_Security_Incidents,
            "Prev_APC_%":                    request.Prev_APC_pct,
            "Prev_ADC_%":                    request.Prev_ADC_pct,
            "Prev_NDC_%":                    request.Prev_NDC_pct,
            "State_Avg_APC":                 request.State_Avg_APC,
            "State_Avg_ADC":                 request.State_Avg_ADC,
            "State_Avg_NDC":                 request.State_Avg_NDC,
            "Zone_Avg_APC":                  request.Zone_Avg_APC,
            "Zone_Avg_ADC":                  request.Zone_Avg_ADC,
            "Zone_Avg_NDC":                  request.Zone_Avg_NDC,
            "Gov_Is_Presidential_Candidate": request.Gov_Is_Presidential_Candidate,
            "Federal_Admin_Control":         request.Federal_Admin_Control,
        }

        # APC and ADC predictions
        input_apc_adc = pd.DataFrame([input_dict])[features_apc_adc_v2]
        apc_raw = float(model_apc_v2.predict(input_apc_adc)[0])
        adc_raw = float(model_adc_v2.predict(input_apc_adc)[0])

        # NDC prediction
        input_ndc = pd.DataFrame([input_dict])[features_ndc_v2]
        ndc_raw = float(model_ndc_v2.predict(input_ndc)[0])

        # APM rule-based
        apm_raw = compute_apm(request.State, request.Geopolitical_Zone)

        # Normalise to 100%
        total   = apc_raw + adc_raw + ndc_raw + apm_raw
        apc_pct = round(apc_raw / total * 100, 2)
        adc_pct = round(adc_raw / total * 100, 2)
        ndc_pct = round(ndc_raw / total * 100, 2)
        apm_pct = round(apm_raw / total * 100, 2)

        scores = {"APC": apc_pct, "ADC": adc_pct, "NDC": ndc_pct, "APM": apm_pct}
        winner = max(scores, key=scores.get)

        return PredictResponse2027(
            apc_pct=apc_pct,
            adc_pct=adc_pct,
            ndc_pct=ndc_pct,
            apm_pct=apm_pct,
            winner=winner
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
