"""
IndustrIQ Serving API — FastAPI version
(matches resume skill list: TensorFlow, Keras, Scikit-learn, FastAPI)

Run with:
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict   -> predicted power consumption for given machine settings
    POST /optimize   -> optimal machine settings for a target power consumption
"""

from typing import List, Optional
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.optimize import minimize

# Single source of truth for feature order — used everywhere so a
# feature-count/order mismatch (the original Flask bug) can't happen.
FEATURE_ORDER = [
    "Temperature",
    "MachineUtilization",
    "ProductionRate",
    "Humidity",
    "LoadCurrent",
    "PowerFactor",
]

DEFAULT_FEATURE_BOUNDS = [
    (20, 60),     # Temperature
    (40, 100),    # MachineUtilization
    (50, 200),    # ProductionRate
    (30, 90),     # Humidity
    (25, 35),     # LoadCurrent
    (0.7, 0.85),  # PowerFactor
]

app = FastAPI(
    title="IndustrIQ API",
    description="DNN-based industrial power consumption predictor & optimizer",
    version="1.0.0",
)

# Loaded once at startup, shared across requests (unlike the notebook globals
# the old Flask app relied on).
model = tf.keras.models.load_model("industriq_model.keras")
sc = joblib.load("scaler_x.joblib")
scy = joblib.load("scaler_y.joblib")


class PredictRequest(BaseModel):
    Temperature: float = Field(..., example=45.0)
    MachineUtilization: float = Field(..., example=84.0)
    ProductionRate: float = Field(..., example=120.0)
    Humidity: float = Field(..., example=60.0)
    LoadCurrent: float = Field(..., example=39.0)
    PowerFactor: float = Field(..., example=0.85)


class PredictResponse(BaseModel):
    prediction: float


class OptimizeRequest(BaseModel):
    desired_power: float = Field(..., example=450.0)
    bounds: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional list of [min, max] pairs, one per feature in FEATURE_ORDER",
    )


class OptimizeResponse(BaseModel):
    Temperature: float
    MachineUtilization: float
    ProductionRate: float
    Humidity: float
    LoadCurrent: float
    PowerFactor: float


def optimize_features(desired_power, feature_bounds):
    def objective(features):
        features = np.array(features).reshape(1, -1)
        scaled = sc.transform(features)
        pred_scaled = model.predict(scaled, verbose=0)
        pred_actual = scy.inverse_transform(pred_scaled)
        return np.abs(desired_power - pred_actual[0][0])

    initial_guess = [np.mean(b) for b in feature_bounds]
    result = minimize(objective, initial_guess, bounds=feature_bounds)
    return result.x


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array([[getattr(req, feat) for feat in FEATURE_ORDER]])
    x_scaled = sc.transform(x)
    y_scaled = model.predict(x_scaled, verbose=0)
    y = float(scy.inverse_transform(y_scaled)[0, 0])
    return PredictResponse(prediction=y)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    bounds = req.bounds if req.bounds is not None else DEFAULT_FEATURE_BOUNDS
    if len(bounds) != len(FEATURE_ORDER):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_ORDER)} bound pairs, got {len(bounds)}",
        )
    bounds = [tuple(b) for b in bounds]
    best = optimize_features(req.desired_power, bounds)
    return OptimizeResponse(**dict(zip(FEATURE_ORDER, [float(v) for v in best])))
