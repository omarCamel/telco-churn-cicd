from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
SAMPLE = {
    "tenure": 5, "MonthlyCharges": 85.5, "TotalCharges": 427.5,
    "Contract": "Month-to-month", "InternetService": "Fiber optic"
}

def test_predict():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in ("Bajo", "Medio", "Alto")

def test_invalid_input():
    response = client.post("/predict", json={**SAMPLE, "tenure": -10})
    assert response.status_code == 422
