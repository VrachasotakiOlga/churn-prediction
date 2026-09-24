# Churn Prediction

## Overview

This repository contains a simple churn prediction pipeline for an online
sports-betting platform. It uses synthetic data generation, an XGBoost model,
and a small FastAPI app for training and prediction.

- **Generate data:** `data_generation.py` creates a synthetic dataset
  representing user demographics and betting behaviour.
- **Train model:** `model_training.py` builds a preprocessing pipeline and
  trains an XGBoost classifier.
- **API:** `main.py` exposes `/model/train` and `/model/predict` endpoints using
  FastAPI.
- **Testing:** `testing_main.py` tests the functions inside `main.py`.
- **Schemas:** Defines the Pydantic models used by the API for request
  validation and responses.

## Requirements

- Python 3.12+

If you use a dependency file, there is a `pyproject.toml` or a
`requirements.txt`.

## Usage

Run the API server (development):

```bash
uv run fastapi dev main.py
```

Train the model:

```bash
curl -X POST "http://localhost:8000/model/train"
```

Make predictions:

```bash
curl -X POST "http://localhost:8000/model/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1888,
    "days_since_registration": 487,
    "age_group": "35-44",
    "country": "Cyprus",
    "num_bets_last_30d": 40,
    "num_bets_last_7d": 12,
    "days_since_last_bet": 0,
    "num_deposits_30d": 20,
    "num_withdrawals_30d": 10,
    "favorite_sport": "Basketball"
  }'
```

```bash
curl -X POST "http://localhost:8000/model/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1889,
    "days_since_registration": 14,
    "age_group": "18-24",
    "country": "Brazil",
    "num_bets_last_30d": 2,
    "num_bets_last_7d": 1,
    "days_since_last_bet": 6,
    "num_deposits_30d": 1,
    "num_withdrawals_30d": 0,
    "favorite_sport": "Football"
  }'
```

Note: `/model/predict` expects a trained `model.joblib` in the project root;
call `/model/train` or run the training snippet above if missing.

For testing run

```bash
uv run pytest -v
```
