from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import xgboost as xgb

from src.models.torch_models import build_autoencoder, build_classifier
from src.preprocessing.flow_preprocessor import FlowPreprocessor
from src.utils.config import PROJECT_ROOT, load_config


NEURAL_MODEL_NAMES = ("ann", "cnn", "rnn", "lstm", "transformer")


@dataclass(frozen=True)
class DetectionResult:
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
    reasons: tuple[str, ...]
    model_version: str


class DetectionEngine:
    def __init__(
        self,
        *,
        preprocessor: FlowPreprocessor,
        classifier: xgb.Booster,
        neural_models: Mapping[str, torch.nn.Module],
        autoencoder: torch.nn.Module,
        labels: tuple[str, ...],
        normal_labels: frozenset[str],
        anomaly_threshold: float,
        classifier_weights: Mapping[str, float],
        risk_weights: Mapping[str, float],
        risk_levels: Mapping[str, float],
        model_version: str,
        device: torch.device,
    ) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.neural_models = dict(neural_models)
        self.autoencoder = autoencoder
        self.labels = labels
        self.normal_labels = normal_labels
        self.anomaly_threshold = anomaly_threshold
        self.classifier_weights = dict(classifier_weights)
        self.risk_weights = dict(risk_weights)
        self.risk_levels = dict(risk_levels)
        self.model_version = model_version
        self.device = device

    @classmethod
    def load(
        cls,
        artifact_root: Path = PROJECT_ROOT / "artifacts",
        config_path: Path | None = None,
        device: torch.device | None = None,
    ) -> DetectionEngine:
        root = artifact_root.resolve()
        manifest_path = root / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
        if manifest.get("schema_version") != 2:
            raise ValueError(f"Unsupported model manifest schema: {manifest_path}")

        def artifact_path(key: str) -> Path:
            try:
                relative = Path(manifest["artifacts"][key])
            except (KeyError, TypeError) as error:
                raise ValueError(f"Manifest is missing artifact path: {key}") from error
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Artifact path must stay under {root}: {relative}")
            path = (root / relative).resolve()
            if root not in path.parents:
                raise ValueError(f"Artifact path escapes model directory: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"Model artifact not found: {path}")
            return path

        preprocessor = FlowPreprocessor.load(artifact_path("preprocessing"))
        with artifact_path("labels").open("r", encoding="utf-8") as source:
            label_payload = json.load(source)
        if label_payload.get("schema_version") != 1:
            raise ValueError("Unsupported label artifact schema.")
        labels_value = label_payload.get("classes")
        if (
            not isinstance(labels_value, list)
            or len(labels_value) < 2
            or any(not isinstance(label, str) or not label for label in labels_value)
            or len(set(labels_value)) != len(labels_value)
        ):
            raise ValueError("Label artifact must contain at least two unique class names.")
        labels = tuple(labels_value)
        if manifest.get("classes") != list(labels):
            raise ValueError("Manifest class order does not match the label artifact.")
        if manifest.get("feature_order") != list(preprocessor.feature_names):
            raise ValueError("Manifest feature order does not match preprocessing artifact.")

        with artifact_path("thresholds").open("r", encoding="utf-8") as source:
            threshold_payload = json.load(source)
        if threshold_payload.get("schema_version") != 1:
            raise ValueError("Unsupported threshold artifact schema.")
        anomaly_threshold = float(threshold_payload["autoencoder_threshold"])
        if not math.isfinite(anomaly_threshold) or anomaly_threshold < 0:
            raise ValueError("Autoencoder threshold must be a finite non-negative value.")
        normal_values = threshold_payload.get("normal_labels")
        if (
            not isinstance(normal_values, list)
            or not normal_values
            or any(label not in labels for label in normal_values)
        ):
            raise ValueError("Threshold artifact contains invalid normal labels.")

        with artifact_path("autoencoder_config").open("r", encoding="utf-8") as source:
            autoencoder_config = json.load(source)
        if autoencoder_config.get("input_size") != len(preprocessor.feature_names):
            raise ValueError("Autoencoder configuration does not match preprocessing.")

        selected_device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        classifier = xgb.Booster()
        classifier.load_model(artifact_path("classifier"))
        neural_models: dict[str, torch.nn.Module] = {}
        for model_name in NEURAL_MODEL_NAMES:
            with artifact_path(f"{model_name}_config").open(
                "r",
                encoding="utf-8",
            ) as source:
                model_config = json.load(source)
            if (
                model_config.get("input_size") != len(preprocessor.feature_names)
                or model_config.get("num_classes") != len(labels)
            ):
                raise ValueError(
                    f"{model_name.upper()} configuration does not match "
                    "preprocessing and labels."
                )
            model = build_classifier(model_config)
            model_state = torch.load(
                artifact_path(model_name),
                map_location=selected_device,
                weights_only=True,
            )
            model.load_state_dict(model_state, strict=True)
            neural_models[model_name] = model.to(selected_device).eval()

        autoencoder = build_autoencoder(autoencoder_config)
        autoencoder_state = torch.load(
            artifact_path("autoencoder"),
            map_location=selected_device,
            weights_only=True,
        )
        autoencoder.load_state_dict(autoencoder_state, strict=True)
        autoencoder.to(selected_device).eval()

        config = load_config(config_path or PROJECT_ROOT / "config.yaml")
        detection_config = config.get("detection")
        if not isinstance(detection_config, dict):
            raise ValueError("Application config must include a detection mapping.")
        classifier_weights = cls._validated_weights(
            detection_config.get("classifier_weights"),
            {"xgboost", *NEURAL_MODEL_NAMES},
        )
        risk_weights = cls._validated_weights(
            detection_config.get("risk_weights"),
            {"attack_probability", "anomaly_score"},
        )
        risk_levels = detection_config.get("risk_levels")
        if not isinstance(risk_levels, dict):
            raise ValueError("Detection config must define risk levels.")
        medium = float(risk_levels.get("medium", -1))
        high = float(risk_levels.get("high", -1))
        if not (math.isfinite(medium) and math.isfinite(high) and 0 <= medium < high <= 1):
            raise ValueError("Risk levels must satisfy 0 <= medium < high <= 1.")

        return cls(
            preprocessor=preprocessor,
            classifier=classifier,
            neural_models=neural_models,
            autoencoder=autoencoder,
            labels=labels,
            normal_labels=frozenset(normal_values),
            anomaly_threshold=anomaly_threshold,
            classifier_weights=classifier_weights,
            risk_weights=risk_weights,
            risk_levels={"medium": medium, "high": high},
            model_version=str(manifest.get("created_at_utc", "unknown")),
            device=selected_device,
        )

    @staticmethod
    def _validated_weights(
        value: object,
        required_keys: set[str],
    ) -> dict[str, float]:
        if not isinstance(value, dict) or set(value) != required_keys:
            raise ValueError(f"Model weights must define exactly: {sorted(required_keys)}")
        weights = {key: float(value[key]) for key in required_keys}
        if (
            not all(math.isfinite(weight) and weight >= 0 for weight in weights.values())
            or not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise ValueError("Model weights must be finite, non-negative, and sum to 1.")
        return weights

    def predict(self, record: Mapping[str, object]) -> DetectionResult:
        features = self.preprocessor.transform_record(record)
        feature_batch = features.reshape(1, -1)
        xgb_scores_array = np.asarray(
            self.classifier.predict(xgb.DMatrix(feature_batch))[0],
            dtype=np.float64,
        )
        with torch.no_grad():
            tensor = torch.from_numpy(feature_batch).to(self.device)
            neural_scores_arrays = {
                name: torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                for name, model in self.neural_models.items()
            }
            reconstruction = self.autoencoder(tensor)
            anomaly_score = float(torch.mean((reconstruction - tensor) ** 2).item())

        if len(xgb_scores_array) != len(self.labels) or any(
            len(scores) != len(self.labels)
            for scores in neural_scores_arrays.values()
        ):
            raise ValueError("Model output dimensions do not match the exported label order.")
        combined = self.classifier_weights["xgboost"] * xgb_scores_array
        for model_name, scores in neural_scores_arrays.items():
            combined += self.classifier_weights[model_name] * scores
        predicted_index = int(np.argmax(combined))
        predicted_category = self.labels[predicted_index]
        confidence_score = float(combined[predicted_index])
        normal_probability = sum(
            float(combined[index])
            for index, label in enumerate(self.labels)
            if label in self.normal_labels
        )
        attack_probability = float(np.clip(1.0 - normal_probability, 0.0, 1.0))
        anomaly_detected = anomaly_score > self.anomaly_threshold
        if self.anomaly_threshold > 0:
            anomaly_component = min(anomaly_score / self.anomaly_threshold, 1.0)
        else:
            anomaly_component = float(anomaly_detected)
        risk_score = (
            self.risk_weights["attack_probability"] * attack_probability
            + self.risk_weights["anomaly_score"] * anomaly_component
        )
        if risk_score >= self.risk_levels["high"]:
            risk_level = "high"
        elif risk_score >= self.risk_levels["medium"]:
            risk_level = "medium"
        else:
            risk_level = "low"

        reasons: list[str] = []
        if predicted_category not in self.normal_labels:
            reasons.append(f"Classifier ensemble predicted {predicted_category}.")
        if anomaly_detected:
            reasons.append("Autoencoder reconstruction error exceeded its training threshold.")
        if not reasons:
            reasons.append("No configured classification or anomaly threshold was exceeded.")

        return DetectionResult(
            predicted_category=predicted_category,
            confidence_score=confidence_score,
            confidence_note="Model score, not a guarantee or calibrated probability of correctness.",
            xgboost_scores={
                label: float(score)
                for label, score in zip(self.labels, xgb_scores_array)
            },
            neural_scores={
                model_name: {
                    label: float(score)
                    for label, score in zip(self.labels, scores)
                }
                for model_name, scores in neural_scores_arrays.items()
            },
            anomaly_score=anomaly_score,
            anomaly_threshold=self.anomaly_threshold,
            anomaly_detected=anomaly_detected,
            attack_probability=attack_probability,
            risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            risk_level=risk_level,
            reasons=tuple(reasons),
            model_version=self.model_version,
        )
