from __future__ import annotations

from fastapi.testclient import TestClient

from jobguard.api import create_app
from jobguard.heuristics import RuleBasedJobFraudDetector


def test_api_predicts_and_batches():
    client = TestClient(create_app(detector=RuleBasedJobFraudDetector()))

    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["status"] == "ok"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    single = client.post(
        "/predict",
        json={
            "title": "Senior Software Engineer",
            "description": "Build internal tools with Python, SQL, and AWS. Competitive salary and benefits.",
            "has_company_logo": True,
        },
    )
    assert single.status_code == 200
    single_body = single.json()
    assert single_body["label"] in {"LEGITIMATE", "FRAUDULENT"}
    assert "fraud_probability" in single_body
    assert "model_source" in single_body

    company_alias = client.post(
        "/predict",
        json={
            "title": "Product Analyst",
            "company": "Acme Analytics",
            "description": "Work with Python, SQL, dashboards, and stakeholder reporting.",
            "has_company_logo": False,
        },
    )
    assert company_alias.status_code == 200
    alias_body = company_alias.json()
    assert alias_body["label"] in {"LEGITIMATE", "FRAUDULENT"}
    assert alias_body["input_length"] > 0

    batch = client.post(
        "/predict/batch",
        json={
            "items": [
                {
                    "text": "Work from home and earn $5000 weekly. Send bank details now.",
                }
            ]
        },
    )
    assert batch.status_code == 200
    batch_body = batch.json()
    assert batch_body["count"] == 1
    assert batch_body["items"][0]["label"] in {"LEGITIMATE", "FRAUDULENT"}
