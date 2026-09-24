from pathlib import Path

import joblib
import pandas as pd
from data_generation import UserData, generate_synthetic_data
from fastapi import FastAPI, HTTPException
from model_training import ModelMetrics, train_model
from schemas import Prediction

app = FastAPI(
    title="Churn Prediction",
    description="Churn prediction model for online sports betting platform",
)

MODEL_PATH = Path("model.joblib")


@app.get("/")
def root():
    return {"Welcome": "Churn Prediction API is running."}


@app.post("/model/train")
def train_endpoint(n_users: int = 20_000) -> ModelMetrics:
    X, y = generate_synthetic_data(n_users)
    model, metrics = train_model(X, y)
    joblib.dump(model, MODEL_PATH)
    return metrics


@app.post("/model/predict")
def predict(user_data: UserData) -> Prediction:
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=428, detail="Model not trained. Please run /model/train"
        )
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([user_data.model_dump()])
    y = model.predict_proba(X)[:, 1]
    return Prediction(churn_probability=y[0])
