from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Diagnóstico de cáncer" in response.text
    assert "<label>Radio promedio</label><input name=\"radius_mean\"" in response.text
    assert "Error estándar del radio" in response.text
    assert "Valor máximo del radio" in response.text

def test_api_title_is_in_spanish():
    assert app.title == "API de diagnóstico de cáncer"
