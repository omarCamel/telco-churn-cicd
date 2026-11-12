import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
SAMPLE = (
    pd.read_csv("data.csv")
    .dropna(axis=1, how="all")
    .drop(columns=["id", "diagnosis"])
    .iloc[0]
    .to_dict()
)


def test_predict():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["diagnosis_probability"] <= 1
    assert body["risk_level"] in ("Bajo", "Medio", "Alto")
    assert any(keyword in body["message"].lower() for keyword in ("benigno", "maligno", "diagnóstico"))


def test_invalid_input():
    invalid = SAMPLE.copy()
    invalid["radius_mean"] = -1
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_dataset_summary():
    response = client.get("/dataset")
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] > 0
    assert body["columns"] > 0
    assert "diagnosis" in body["column_names"]
