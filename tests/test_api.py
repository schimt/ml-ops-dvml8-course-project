from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint_returns_service_status():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "model_path" in body
    assert "device" in body


def test_metrics_endpoint_exposes_prometheus_metrics():
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "catdog_model_loaded" in response.text
