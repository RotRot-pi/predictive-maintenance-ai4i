from fastapi.testclient import TestClient
import pytest

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    payload = {
        "air_temperature": 298.5,
        "process_temperature": 309.2,
        "rotational_speed": 1420,
        "torque": 55.3,
        "tool_wear": 190,
        "product_type": "M",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["risk_score"] == pytest.approx(0.025340067, rel=1e-4)
    assert data["failure_predicted"] is False
    assert data["decision_threshold"] == pytest.approx(0.15371445)


def test_reject_invalid_product_type():
    payload = {
        "air_temperature": 298.5,
        "process_temperature": 309.2,
        "rotational_speed": 1420,
        "torque": 55.3,
        "tool_wear": 190,
        "product_type": "X",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_reject_negative_torque():
    payload = {
        "air_temperature": 298.5,
        "process_temperature": 309.2,
        "rotational_speed": 1420,
        "torque": -5,
        "tool_wear": 190,
        "product_type": "M",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422