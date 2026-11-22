from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imb.detection.engine import DetectionResult


_METADATA_FIELDS = {
    "Destination Port": "destination_port",
    "Flow Duration": "flow_duration",
    "Flow Packets/s": "flow_packets_per_second",
    "Flow Bytes/s": "flow_bytes_per_second",
}


class DetectionRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        detected_at TEXT NOT NULL,
                        predicted_category TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        anomaly_score REAL NOT NULL,
                        anomaly_detected INTEGER NOT NULL,
                        attack_probability REAL NOT NULL,
                        risk_score REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        is_alert INTEGER NOT NULL,
                        reasons_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_detections_alert_time
                    ON detections (is_alert, detected_at DESC, id DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_detections_category
                    ON detections (predicted_category)
                    """
                )

    def record_detection(
        self,
        result: DetectionResult,
        flow: Mapping[str, object],
    ) -> dict[str, Any]:
        values = self._values(result, flow)
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    self._insert_sql(),
                    values,
                )
                detection_id = cursor.lastrowid
        if detection_id is None:
            raise RuntimeError("SQLite did not return an ID for the saved detection.")
        return {
            "id": detection_id,
            "detected_at": values[0],
        }

    def record_detections(
        self,
        detections: Sequence[tuple[DetectionResult, Mapping[str, object]]],
    ) -> tuple[int, int]:
        if not detections:
            return 0, 0
        values = [self._values(result, flow) for result, flow in detections]
        with closing(self._connect()) as connection:
            with connection:
                connection.executemany(self._insert_sql(), values)
        alerts_created = sum(value[9] for value in values)
        return len(values), alerts_created

    def list_alerts(self, *, limit: int, offset: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, detected_at, predicted_category, confidence_score,
                       anomaly_score, anomaly_detected, attack_probability,
                       risk_score, risk_level, model_version, reasons_json,
                       metadata_json
                FROM detections
                WHERE is_alert = 1
                ORDER BY detected_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_summary(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            totals = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(is_alert), 0) AS alerts
                FROM detections
                """
            ).fetchone()
            categories = connection.execute(
                """
                SELECT predicted_category, COUNT(*) AS count
                FROM detections
                GROUP BY predicted_category
                ORDER BY count DESC, predicted_category
                """
            ).fetchall()
            risks = connection.execute(
                """
                SELECT risk_level, COUNT(*) AS count
                FROM detections
                GROUP BY risk_level
                """
            ).fetchall()
        risk_distribution = {"low": 0, "medium": 0, "high": 0}
        risk_distribution.update(
            {str(row["risk_level"]): int(row["count"]) for row in risks}
        )
        return {
            "total_flows": int(totals["total"]),
            "suspicious_flows": int(totals["alerts"]),
            "category_distribution": {
                str(row["predicted_category"]): int(row["count"])
                for row in categories
            },
            "risk_distribution": risk_distribution,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _insert_sql() -> str:
        return """
            INSERT INTO detections (
                detected_at, predicted_category, confidence_score, anomaly_score,
                anomaly_detected, attack_probability, risk_score, risk_level,
                model_version, is_alert, reasons_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

    @staticmethod
    def _values(
        result: DetectionResult,
        flow: Mapping[str, object],
    ) -> tuple[Any, ...]:
        metadata: dict[str, float] = {}
        for feature_name, storage_name in _METADATA_FIELDS.items():
            value = flow.get(feature_name)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                metadata[storage_name] = numeric_value

        is_alert = DetectionRepository.is_suspicious(result)
        return (
            datetime.now(timezone.utc).isoformat(),
            result.predicted_category,
            result.confidence_score,
            result.anomaly_score,
            int(result.anomaly_detected),
            result.attack_probability,
            result.risk_score,
            result.risk_level,
            result.model_version,
            int(is_alert),
            json.dumps(list(result.reasons), allow_nan=False),
            json.dumps(metadata, allow_nan=False),
        )

    @staticmethod
    def is_suspicious(result: DetectionResult) -> bool:
        return (
            result.predicted_category.strip().casefold() not in {"benign", "normal"}
            or result.anomaly_detected
            or result.risk_level != "low"
        )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "detected_at": str(row["detected_at"]),
            "predicted_category": str(row["predicted_category"]),
            "confidence_score": float(row["confidence_score"]),
            "anomaly_score": float(row["anomaly_score"]),
            "anomaly_detected": bool(row["anomaly_detected"]),
            "attack_probability": float(row["attack_probability"]),
            "risk_score": float(row["risk_score"]),
            "risk_level": str(row["risk_level"]),
            "model_version": str(row["model_version"]),
            "reasons": json.loads(row["reasons_json"]),
            "flow_metadata": json.loads(row["metadata_json"]),
        }
