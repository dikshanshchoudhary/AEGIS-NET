from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from imb.detection.engine import DetectionResult


class _FakePreprocessor:
    feature_names = ("Destination Port", "Flow Duration")


class _FakeEngine:
    def __init__(self) -> None:
        self.preprocessor = _FakePreprocessor()
        self.model_version = "test-model"
        self.device = "cpu"

    def predict(self, record: Mapping[str, object]) -> DetectionResult:
        missing = [name for name in self.preprocessor.feature_names if name not in record]
        if missing:
            raise ValueError(f"Flow record is missing required features: {missing}")
        is_attack = float(record["Flow Duration"] or 0) > 100
        category = "DoS Hulk" if is_attack else "BENIGN"
        return DetectionResult(
            predicted_category=category,
            confidence_score=0.9,
            confidence_note="Test model score.",
            xgboost_scores={"BENIGN": 0.1 if is_attack else 0.9, "DoS Hulk": 0.9 if is_attack else 0.1},
            neural_scores={"ann": {"BENIGN": 0.1 if is_attack else 0.9, "DoS Hulk": 0.9 if is_attack else 0.1}},
            anomaly_score=0.8 if is_attack else 0.1,
            anomaly_threshold=0.5,
            anomaly_detected=is_attack,
            attack_probability=0.9 if is_attack else 0.1,
            risk_score=0.9 if is_attack else 0.1,
            risk_level="high" if is_attack else "low",
            reasons=("Test alert.",) if is_attack else ("No threshold exceeded.",),
            model_version=self.model_version,
        )

    def predict_many(
        self,
        records: list[Mapping[str, object]],
    ) -> list[DetectionResult]:
        return [self.predict(record) for record in records]


@pytest.fixture
def client(tmp_path: Path):
    metrics_path = tmp_path / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "xgboost": {"accuracy": 0.9, "macro_f1": 0.8},
                "neural_classifiers": {
                    "ann": {"accuracy": 0.85, "macro_f1": 0.75}
                },
                "autoencoder_anomaly_detection": {
                    "accuracy": 0.7,
                    "macro_f1": 0.6,
                    "reconstruction_mse_threshold": 0.5,
                    "calibration_quantile": 0.95,
                },
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        engine_loader=_FakeEngine,
        database_path=tmp_path / "detections.db",
        evaluation_metrics_path=metrics_path,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_predict_alerts_and_metrics_share_detection_history(client):
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["feature_count"] == 2

    prediction = client.post(
        "/predict",
        json={"features": {"Destination Port": 443, "Flow Duration": 200}},
    )
    assert prediction.status_code == 200
    assert prediction.json()["result"]["predicted_category"] == "DoS Hulk"

    alerts = client.get("/alerts").json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["flow_metadata"] == {
        "destination_port": 443.0,
        "flow_duration": 200.0,
    }

    metrics = client.get("/metrics").json()
    assert metrics["total_flows"] == 1
    assert metrics["suspicious_flows"] == 1
    assert metrics["model_performance"]["classifiers"]["XGBoost"]["accuracy"] == 0.9


def test_predict_rejects_missing_features_without_saving_a_detection(client):
    response = client.post(
        "/predict",
        json={"features": {"Destination Port": 443}},
    )
    assert response.status_code == 422
    assert client.get("/metrics").json()["total_flows"] == 0


def test_analyze_file_records_batch_summary_and_rejects_wrong_schema(client):
    analyzed = client.post(
        "/analyze-file",
        files={
            "file": (
                "flows.csv",
                "Destination Port,Flow Duration,Label\n80,10,BENIGN\n443,300,DoS Hulk\n",
                "text/csv",
            )
        },
    )
    assert analyzed.status_code == 200
    body = analyzed.json()
    assert body["rows_analyzed"] == 2
    assert body["alerts_created"] == 1
    assert body["category_distribution"] == {"BENIGN": 1, "DoS Hulk": 1}

    invalid = client.post(
        "/analyze-file",
        files={"file": ("bad.csv", "Destination Port\n80\n", "text/csv")},
    )
    assert invalid.status_code == 422
    assert client.get("/metrics").json()["total_flows"] == 2


def test_flow_request_rejects_non_finite_feature_values(client):
    response = client.post(
        "/predict",
        json={"features": {"Destination Port": "Infinity", "Flow Duration": 1}},
    )
    assert response.status_code == 422
