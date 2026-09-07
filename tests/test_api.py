"""Smoke tests da API (exigem o modelo treinado em data/fraud_model.joblib)."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from src import config as cfg

pytestmark = pytest.mark.skipif(
    not cfg.MODEL_PATH.exists() or not (cfg.DATA_DIR / "transactions_scored.csv").exists(),
    reason="requer 'python main.py train' antes",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_datasets_listed(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    assert any(d["id"] == "default" for d in r.json())


def test_score_realtime(client):
    body = {
        "customer_id": 999999, "amount": 42.0, "merchant_category": "supermercado",
        "device_id": "dev_a", "city": "Sao Paulo", "lat": -23.55, "lon": -46.63,
        "update_state": True,
    }
    r = client.post("/score", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["decision"] in {"aprovar", "revisar", "bloquear"}
    assert 0.0 <= out["fraud_probability"] <= 1.0
    assert "latency_ms" in out


def test_score_flags_takeover(client):
    cid = 888888
    # aquece o cliente com compras normais
    for i in range(6):
        client.post("/score", json={
            "customer_id": cid, "amount": 100.0, "merchant_category": "supermercado",
            "device_id": "dev_home", "city": "Sao Paulo", "lat": -23.55, "lon": -46.63,
            "timestamp": f"2024-03-0{i+1} 10:00:00",
        })
    r = client.post("/score", json={
        "customer_id": cid, "amount": 900.0, "merchant_category": "cripto_exchange",
        "device_id": "dev_new", "city": "Miami", "lat": 25.76, "lon": -80.19,
        "timestamp": "2024-03-08 03:00:00",
    })
    out = r.json()
    assert out["inferred_pattern"] in {"account_takeover", "impossible_travel", "high_amount"}
    assert out["decision"] in {"revisar", "bloquear"}


def test_feedback_recorded(client, tmp_path, monkeypatch):
    from src import feedback as fb
    monkeypatch.setattr(fb, "FEEDBACK_PATH", tmp_path / "fb.jsonl")
    r = client.post("/api/feedback", json={"transaction_id": 100000, "label": "fraud"})
    assert r.status_code == 200
    assert r.json()["recorded"]["label"] == 1


def test_api_key_enforced(monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")
    c = TestClient(api.app)
    assert c.get("/api/datasets").status_code == 401
    assert c.get("/api/datasets", headers={"x-api-key": "secret"}).status_code == 200
