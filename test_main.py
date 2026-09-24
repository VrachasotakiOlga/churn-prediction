import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import app
from pytest_mock import MockerFixture
from schemas import ModelMetrics

client = TestClient(app)


@pytest.fixture
def sample_user_data() -> dict[str, str | int | float]:
    return {
        "user_id": 12345,
        "days_since_registration": 150,
        "age_group": "25-34",
        "country": "Greece",
        "num_bets_last_30d": 25,
        "num_bets_last_7d": 8,
        "days_since_last_bet": 2,
        "num_deposits_30d": 5,
        "num_withdrawals_30d": 2,
        "favorite_sport": "Football",
    }


def test_train_endpoint(mocker: MockerFixture) -> None:
    mock_generate_data = mocker.patch("main.generate_synthetic_data")
    mock_train_model = mocker.patch("main.train_model")
    mock_dump = mocker.patch("main.joblib.dump")

    mock_X = pd.DataFrame({"feature1": [1, 2, 3]})
    mock_y = pd.Series([0, 1, 0])
    mock_generate_data.return_value = (mock_X, mock_y)

    mock_model = mocker.MagicMock()
    mock_metrics = ModelMetrics(
        accuracy=0.85, precision=0.82, recall=0.78, f1=0.80, roc_auc=0.88
    )
    mock_train_model.return_value = (mock_model, mock_metrics)

    response = client.post("/model/train")

    assert response.status_code == 200
    assert response.json() == mock_metrics.model_dump()
    mock_generate_data.assert_called_once_with(20_000)
    mock_dump.assert_called_once()


def test_predict_without_model(
    mocker: MockerFixture,
    sample_user_data: dict[str, str | int | float],
) -> None:
    mock_model_path = mocker.patch("main.MODEL_PATH")
    mock_model_path.is_file.return_value = False

    response = client.post("/model/predict", json=sample_user_data)

    assert response.status_code == 428


def test_predict_with_model(
    mocker: MockerFixture,
    sample_user_data: dict[str, str | int | float],
) -> None:
    mock_model_path = mocker.patch("main.MODEL_PATH")
    mock_load = mocker.patch("main.joblib.load")

    mock_model_path.is_file.return_value = True
    mock_model = mocker.MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - 0.42, 0.42]])
    mock_load.return_value = mock_model

    response = client.post("/model/predict", json=sample_user_data)

    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.42


def test_predict_invalid_data() -> None:
    invalid_data = {"user_id": "not_a_number"}
    response = client.post("/model/predict", json=invalid_data)
    assert response.status_code == 422
