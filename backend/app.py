import os
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")


app = FastAPI(title="Minimal MLflow + FastAPI", version="0.1.0")


class PredictRequest(BaseModel):
    # Feature vector; we'll enforce length=4 in the endpoint to avoid Pydantic version differences
    features: List[float]


class PredictResponse(BaseModel):
    predicted_class: int
    probability: float


@app.on_event("startup")
def load_model() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model not found. Run the training script first: python train.py"
        )
    # Load sklearn-flavor model from MLflow directory
    global model
    model = mlflow.sklearn.load_model(MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Validate length explicitly (training script uses 4 features)
    if len(req.features) != 4:
        raise HTTPException(status_code=422, detail="Expected 4 features")

    # Convert to shape (1, 4)
    x = np.array(req.features, dtype=float).reshape(1, -1)

    try:
        # For classifiers, get both class and probability of positive class (1)
        proba = float(model.predict_proba(x)[0, 1])
        cls = int(model.predict(x)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictResponse(predicted_class=cls, probability=proba)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
