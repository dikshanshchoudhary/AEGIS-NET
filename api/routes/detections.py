from __future__ import annotations

import io
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from database.repository import DetectionRepository
from src.detection.engine import DetectionEngine, DetectionResult
from api.schemas.detection import (
    AlertResponse,
    AlertsResponse,
    AnalyzeFileAlert,
    AnalyzeFileResponse,
    DetectionResultResponse,
    FlowRequest,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
)


router = APIRouter()


def _engine(request: Request) -> DetectionEngine:
    return request.app.state.engine


def _repository(request: Request) -> DetectionRepository:
    return request.app.state.repository


def _result_response(result: DetectionResult) -> DetectionResultResponse:
    payload = asdict(result)
    payload["reasons"] = list(result.reasons)
    return DetectionResultResponse.model_validate(payload)


def _performance_summary(payload: dict[str, Any]) -> dict[str, object]:
    classifier_metrics: dict[str, dict[str, float]] = {}
    xgboost_metrics = payload.get("xgboost")
    if isinstance(xgboost_metrics, dict):
        classifier_metrics["XGBoost"] = {
            "accuracy": float(xgboost_metrics["accuracy"]),
            "macro_f1": float(xgboost_metrics["macro_f1"]),
        }

    neural_metrics = payload.get("neural_classifiers")
    if isinstance(neural_metrics, dict):
        for model_name, metrics in neural_metrics.items():
            if isinstance(metrics, dict):
                classifier_metrics[str(model_name).upper()] = {
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                }

    anomaly_metrics = payload.get("autoencoder_anomaly_detection")
    anomaly_summary: dict[str, float] = {}
    if isinstance(anomaly_metrics, dict):
        anomaly_summary = {
            "accuracy": float(anomaly_metrics["accuracy"]),
            "macro_f1": float(anomaly_metrics["macro_f1"]),
            "threshold": float(anomaly_metrics["reconstruction_mse_threshold"]),
            "calibration_quantile": float(anomaly_metrics["calibration_quantile"]),
        }

    return {
        "classifiers": classifier_metrics,
        "anomaly_detector": anomaly_summary,
        "evaluation_note": (
            "Metrics use a stratified random split from the same dataset; they do "
            "not establish performance on another capture, day, or network."
        ),
    }


@router.get("/health", response_model=HealthResponse, summary="Check service readiness")
def health(request: Request) -> HealthResponse:
    engine = _engine(request)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=engine.model_version,
        device=str(engine.device),
        feature_count=len(engine.preprocessor.feature_names),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify one network flow",
)
def predict(
    body: FlowRequest,
    request: Request,
) -> PredictionResponse:
    engine = _engine(request)
    try:
        result = engine.predict(body.features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    saved = _repository(request).record_detection(result, body.features)
    return PredictionResponse(
        detection_id=saved["id"],
        detected_at=saved["detected_at"],
        result=_result_response(result),
    )


@router.post(
    "/analyze-file",
    response_model=AnalyzeFileResponse,
    summary="Analyze a CSV of network flows",
)
def analyze_file(
    request: Request,
    file: UploadFile = File(...),
) -> AnalyzeFileResponse:
    filename = Path(file.filename or "").name
    if Path(filename).suffix.casefold() != ".csv":
        raise HTTPException(status_code=415, detail="Upload a .csv file.")

    max_upload_bytes = request.app.state.max_upload_bytes
    content = file.file.read(max_upload_bytes + 1)
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {max_upload_bytes}-byte upload limit.",
        )
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            encoding="cp1252",
            nrows=request.app.state.max_csv_rows + 1,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {error}") from error
    if frame.empty:
        raise HTTPException(status_code=422, detail="CSV contains no flow records.")
    if len(frame) > request.app.state.max_csv_rows:
        raise HTTPException(
            status_code=413,
            detail=(
                f"CSV exceeds the {request.app.state.max_csv_rows}-row analysis limit."
            ),
        )

    engine = _engine(request)
    feature_names = list(engine.preprocessor.feature_names)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "CSV is missing required flow features.", "missing": missing},
        )

    numeric_features = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    records = numeric_features.to_dict(orient="records")
    try:
        results = engine.predict_many(records)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    metadata = frame.to_dict(orient="records")
    saved_count, alert_count = _repository(request).record_detections(
        list(zip(results, metadata))
    )
    category_distribution = dict(
        sorted(Counter(result.predicted_category for result in results).items())
    )
    risk_distribution = {"low": 0, "medium": 0, "high": 0}
    risk_distribution.update(
        Counter(result.risk_level for result in results)
    )
    alert_examples = [
        AnalyzeFileAlert(
            row_number=index + 1,
            predicted_category=result.predicted_category,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            anomaly_detected=result.anomaly_detected,
        )
        for index, result in enumerate(results)
        if DetectionRepository.is_suspicious(result)
    ][:20]

    return AnalyzeFileResponse(
        rows_analyzed=saved_count,
        alerts_created=alert_count,
        category_distribution=category_distribution,
        risk_distribution=risk_distribution,
        alert_examples=alert_examples,
    )


@router.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="List recent suspicious detections",
)
def alerts(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> AlertsResponse:
    rows = _repository(request).list_alerts(limit=limit, offset=offset)
    return AlertsResponse(
        alerts=[AlertResponse.model_validate(row) for row in rows],
        limit=limit,
        offset=offset,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get detection and model metrics",
)
def metrics(request: Request) -> MetricsResponse:
    summary = _repository(request).get_summary()
    performance = _performance_summary(request.app.state.evaluation_metrics)
    return MetricsResponse(**summary, model_performance=performance)
