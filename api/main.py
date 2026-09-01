import joblib
import pandas as pd

from fastapi import FastAPI

from api.schemas import SensorReading, PredictionResponse
from src.preprocessing import preprocess_features


MODEL_PATH = "models/xgb_model.joblib"
THRESHOLD_PATH = "models/decision_threshold.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"


app = FastAPI(
    title="AI4I Predictive Maintenance API",
    version="1.0.0",
)


model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    raw_input = pd.DataFrame(
        [
            {
                "Type": reading.product_type,
                "Air temperature": reading.air_temperature,
                "Process temperature": reading.process_temperature,
                "Rotational speed": reading.rotational_speed,
                "Torque": reading.torque,
                "Tool wear": reading.tool_wear,
            }
        ]
    )

    processed = preprocess_features(raw_input)

    processed = processed[feature_columns]

    risk_score = float(model.predict_proba(processed)[:, 1][0])

    return PredictionResponse(
        risk_score=risk_score,
        failure_predicted=risk_score >= threshold,
        decision_threshold=float(threshold),
    )