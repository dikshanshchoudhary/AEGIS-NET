from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FlowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float | None] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def validate_finite_features(
        cls,
        features: dict[str, float | None],
    ) -> dict[str, float | None]:
        for name, value in features.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Feature {name!r} must be finite or null.")
        return features


class DetectionResultResponse(BaseModel):
    predicted_category: str
    confidence_score: float
    confidence_note: str
    xgboost_scores: dict[str, float]
    neural_scores: dict[str, dict[str, float]]
    anomaly_score: float
    anomaly_threshold: float
    anomaly_detected: bool
    attack_probability: float
    risk_score: float
    risk_level: str
    reasons: list[str]
    model_version: str


class PredictionResponse(BaseModel):
    detection_id: int
    detected_at: datetime
    result: DetectionResultResponse


class AlertResponse(BaseModel):
    id: int
    detected_at: datetime
    predicted_category: str
    confidence_score: float
    anomaly_score: float
    anomaly_detected: bool
    attack_probability: float
    risk_score: float
    risk_level: str
    model_version: str
    reasons: list[str]
    flow_metadata: dict[str, float]


class AlertsResponse(BaseModel):
    alerts: list[AlertResponse]
    limit: int
    offset: int


class AnalyzeFileAlert(BaseModel):
    row_number: int
    predicted_category: str
    risk_score: float
    risk_level: str
    anomaly_detected: bool


class AnalyzeFileResponse(BaseModel):
    rows_analyzed: int
    alerts_created: int
    category_distribution: dict[str, int]
    risk_distribution: dict[str, int]
    alert_examples: list[AnalyzeFileAlert]


class MetricsResponse(BaseModel):
    total_flows: int
    suspicious_flows: int
    category_distribution: dict[str, int]
    risk_distribution: dict[str, int]
    model_performance: dict[str, object]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str
    feature_count: int
