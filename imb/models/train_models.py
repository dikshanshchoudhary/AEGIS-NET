from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from imb.models.torch_models import build_autoencoder, build_classifier
from imb.preprocessing.cicids import (
    DATASET_ID,
    FlowDatasetSample,
    download_cicids_dataset,
    sample_cicids_csvs,
)
from imb.preprocessing.flow_preprocessor import FlowPreprocessor
from imb.utils.config import PROJECT_ROOT
from imb.utils.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
NORMAL_LABELS = {"benign", "normal"}
NEURAL_MODEL_NAMES = ("ann", "cnn", "rnn", "lstm", "transformer")


def _stratified_split(
    labels: np.ndarray,
    class_count: int,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    singleton_classes: list[int] = []

    for class_index in range(class_count):
        indices = np.flatnonzero(labels == class_index)
        rng.shuffle(indices)
        if len(indices) < 2:
            train_parts.append(indices)
            singleton_classes.append(class_index)
            continue
        test_count = max(1, int(round(len(indices) * test_fraction)))
        test_count = min(test_count, len(indices) - 1)
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])

    if not test_parts:
        raise ValueError("No class has enough samples for a held-out evaluation split.")
    return np.concatenate(train_parts), np.concatenate(test_parts), singleton_classes


def _classification_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    evaluated = support > 0

    return {
        "accuracy": float(true_positive.sum() / max(int(support.sum()), 1)),
        "macro_f1": float(f1[evaluated].mean()) if evaluated.any() else 0.0,
        "macro_f1_classes": [
            name for name, has_support in zip(class_names, evaluated) if has_support
        ],
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion.tolist(),
    }


def _predict_classifier(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    if not probabilities:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
    scores = np.concatenate(probabilities)
    return scores, scores.argmax(axis=1)


def _reconstruction_errors(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            scores = torch.mean((model(batch) - batch) ** 2, dim=1)
            errors.append(scores.cpu().numpy())
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)


def _artifact_paths() -> dict[str, Path]:
    paths = {
        "classifier": ARTIFACT_ROOT / "classifiers" / "xgboost_classifier.json",
        "autoencoder": ARTIFACT_ROOT / "autoencoders" / "autoencoder_state_dict.pt",
        "autoencoder_config": ARTIFACT_ROOT / "autoencoders" / "autoencoder_config.json",
        "preprocessing": ARTIFACT_ROOT / "preprocessing" / "preprocessing.json",
        "labels": ARTIFACT_ROOT / "preprocessing" / "labels.json",
        "thresholds": ARTIFACT_ROOT / "preprocessing" / "thresholds.json",
        "metrics": ARTIFACT_ROOT / "preprocessing" / "evaluation_metrics.json",
        "manifest": ARTIFACT_ROOT / "manifest.json",
    }
    for model_name in NEURAL_MODEL_NAMES:
        paths[model_name] = (
            ARTIFACT_ROOT / "classifiers" / f"{model_name}_state_dict.pt"
        )
        paths[f"{model_name}_config"] = (
            ARTIFACT_ROOT / "classifiers" / f"{model_name}_config.json"
        )
    return paths


def _neural_model_configs(
    input_size: int,
    num_classes: int,
) -> dict[str, dict[str, object]]:
    common = {"input_size": input_size, "num_classes": num_classes, "dropout": 0.1}
    return {
        "ann": {
            **common,
            "architecture": "dense_ann_v1",
            "hidden_sizes": [128, 64],
        },
        "cnn": {
            **common,
            "architecture": "feature_cnn1d_v2",
            "channels": 16,
            "head_size": 128,
        },
        "rnn": {
            **common,
            "architecture": "feature_rnn_v2",
            "embedding_size": 16,
            "hidden_size": 64,
        },
        "lstm": {
            **common,
            "architecture": "feature_lstm_v2",
            "embedding_size": 16,
            "hidden_size": 64,
        },
        "transformer": {
            **common,
            "architecture": "feature_transformer_v1",
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
        },
    }


def _train_neural_classifier(
    model_config: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    model_name: str,
) -> nn.Module:
    torch.manual_seed(seed)
    model = build_classifier(model_config).to(device)
    model_batch_size = (
        min(batch_size, 256) if model_name == "transformer" else batch_size
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=model_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_features)
        LOGGER.info(
            "%s epoch %d/%d loss=%.6f",
            model_name.upper(),
            epoch + 1,
            epochs,
            total_loss / len(x_train),
        )
    return model


def train_and_export(
    sample: FlowDatasetSample,
    *,
    max_rows_per_class: int,
    test_fraction: float,
    seed: int,
    xgb_estimators: int,
    epochs: int,
    batch_size: int,
    anomaly_calibration_quantile: float,
    device: torch.device,
    xgb_device: str,
    overwrite: bool,
) -> dict[str, Any]:
    paths = _artifact_paths()
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Model artifacts already exist. Review them and rerun with "
            "--overwrite-artifacts to replace: "
            + ", ".join(str(path) for path in existing)
        )

    class_names = sorted(set(str(label) for label in sample.labels))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray(
        [class_to_index[str(label)] for label in sample.labels],
        dtype=np.int64,
    )
    train_indices, test_indices, singleton_classes = _stratified_split(
        labels,
        len(class_names),
        test_fraction,
        seed,
    )
    train_frame = sample.features.iloc[train_indices].reset_index(drop=True)
    test_frame = sample.features.iloc[test_indices].reset_index(drop=True)
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    preprocessor = FlowPreprocessor.fit(
        train_frame,
        list(sample.features.columns),
    )
    x_train = preprocessor.transform_frame(train_frame)
    x_test = preprocessor.transform_frame(test_frame)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Preprocessing produced non-finite model inputs.")

    class_counts = np.bincount(y_train, minlength=len(class_names))
    if np.any(class_counts == 0):
        absent = [class_names[index] for index, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"Training split is missing classes: {absent}")
    class_weights = len(y_train) / (len(class_names) * class_counts)
    sample_weights = class_weights[y_train]
    LOGGER.info(
        "Training with %d rows, %d features, %d classes on %s",
        len(y_train),
        len(preprocessor.feature_names),
        len(class_names),
        device,
    )

    train_matrix = xgb.DMatrix(
        x_train,
        label=y_train,
        weight=sample_weights,
    )
    test_matrix = xgb.DMatrix(x_test)
    classifier = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": len(class_names),
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": xgb_device,
            "eval_metric": "mlogloss",
            "nthread": min(4, os.cpu_count() or 1),
            "seed": seed,
            "verbosity": 0,
        },
        train_matrix,
        num_boost_round=xgb_estimators,
        verbose_eval=False,
    )
    xgb_scores = np.asarray(classifier.predict(test_matrix), dtype=np.float64)
    if xgb_scores.shape != (len(test_indices), len(class_names)):
        raise ValueError(
            "XGBoost returned an unexpected probability shape: "
            f"{xgb_scores.shape}"
        )
    xgb_predictions = xgb_scores.argmax(axis=1).astype(np.int64)
    xgb_metrics = _classification_metrics(y_test, xgb_predictions, class_names)

    neural_configs = _neural_model_configs(
        len(preprocessor.feature_names),
        len(class_names),
    )
    neural_models: dict[str, nn.Module] = {}
    neural_metrics: dict[str, dict[str, Any]] = {}
    for model_index, model_name in enumerate(NEURAL_MODEL_NAMES):
        neural_model = _train_neural_classifier(
            neural_configs[model_name],
            x_train,
            y_train,
            class_weights,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + model_index,
            model_name=model_name,
        )
        _, predictions = _predict_classifier(
            neural_model,
            x_test,
            device,
            min(batch_size, 256) if model_name == "transformer" else batch_size,
        )
        neural_metrics[model_name] = _classification_metrics(
            y_test,
            predictions,
            class_names,
        )
        neural_models[model_name] = neural_model

    normal_class_ids = [
        index for index, name in enumerate(class_names)
        if name.strip().casefold() in NORMAL_LABELS
    ]
    if not normal_class_ids:
        raise ValueError(f"No BENIGN/Normal class found in labels: {class_names}")
    normal_training = x_train[np.isin(y_train, normal_class_ids)]
    if len(normal_training) < 20:
        raise ValueError(
            f"At least 20 normal training rows are required; found {len(normal_training)}."
        )

    rng = np.random.default_rng(seed + 2)
    normal_order = rng.permutation(len(normal_training))
    calibration_size = max(10, int(round(len(normal_training) * 0.2)))
    if calibration_size >= len(normal_training):
        raise ValueError("Not enough normal rows for fit and calibration subsets.")
    calibration = normal_training[normal_order[:calibration_size]]
    autoencoder_training = normal_training[normal_order[calibration_size:]]

    autoencoder_config: dict[str, object] = {
        "architecture": "dense_autoencoder_v1",
        "input_size": len(preprocessor.feature_names),
        "layer_sizes": [
            len(preprocessor.feature_names),
            128,
            64,
            32,
            64,
            128,
            len(preprocessor.feature_names),
        ],
        "activation": "relu",
    }
    torch.manual_seed(seed + 1)
    autoencoder = build_autoencoder(autoencoder_config).to(device)
    autoencoder_loader = DataLoader(
        TensorDataset(torch.from_numpy(autoencoder_training)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    autoencoder_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3)
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0.0
        for (batch_features,) in autoencoder_loader:
            batch_features = batch_features.to(device)
            autoencoder_optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((autoencoder(batch_features) - batch_features) ** 2)
            loss.backward()
            autoencoder_optimizer.step()
            total_loss += loss.item() * len(batch_features)
        LOGGER.info(
            "Autoencoder epoch %d/%d loss=%.6f",
            epoch + 1,
            epochs,
            total_loss / len(autoencoder_training),
        )

    calibration_errors = _reconstruction_errors(
        autoencoder,
        calibration,
        device,
        batch_size,
    )
    anomaly_threshold = float(
        np.quantile(calibration_errors, anomaly_calibration_quantile)
    )
    test_errors = _reconstruction_errors(autoencoder, x_test, device, batch_size)
    normal_test = np.isin(y_test, normal_class_ids)
    true_anomaly = (~normal_test).astype(np.int64)
    predicted_anomaly = (test_errors > anomaly_threshold).astype(np.int64)
    anomaly_metrics = _classification_metrics(
        true_anomaly,
        predicted_anomaly,
        ["normal", "anomaly"],
    )
    metrics: dict[str, Any] = {
        "xgboost": xgb_metrics,
        "neural_classifiers": neural_metrics,
        "autoencoder_anomaly_detection": {
            "calibration_quantile": anomaly_calibration_quantile,
            "reconstruction_mse_threshold": anomaly_threshold,
            **anomaly_metrics,
        },
        "test_rows": int(len(test_indices)),
        "test_class_support": {
            name: int(np.sum(y_test == index))
            for index, name in enumerate(class_names)
        },
        "single_sample_classes_kept_out_of_test": [
            class_names[index] for index in singleton_classes
        ],
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(paths["classifier"]))
    for model_name, model in neural_models.items():
        torch.save(model.cpu().state_dict(), paths[model_name])
        _write_json(paths[f"{model_name}_config"], neural_configs[model_name])
    torch.save(autoencoder.cpu().state_dict(), paths["autoencoder"])
    preprocessor.save(paths["preprocessing"])
    _write_json(paths["autoencoder_config"], autoencoder_config)
    _write_json(paths["labels"], {"schema_version": 1, "classes": class_names})
    _write_json(paths["thresholds"], {
        "schema_version": 1,
        "autoencoder_score": "mean squared reconstruction error on standardized features",
        "autoencoder_threshold": anomaly_threshold,
        "calibration_quantile": anomaly_calibration_quantile,
        "normal_labels": [class_names[index] for index in normal_class_ids],
    })
    _write_json(paths["metrics"], metrics)

    manifest = {
        "schema_version": 2,
        "project": "IMB",
        "dataset": DATASET_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "device": str(device),
        "xgboost_device": xgb_device,
        "torch_cuda_version": torch.version.cuda,
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
            "torch": torch.__version__,
        },
        "feature_count": len(preprocessor.feature_names),
        "feature_order": list(preprocessor.feature_names),
        "classes": class_names,
        "normal_labels": [class_names[index] for index in normal_class_ids],
        "training_parameters": {
            "seed": seed,
            "max_rows_per_class": max_rows_per_class,
            "test_fraction": test_fraction,
            "xgb_estimators": xgb_estimators,
            "epochs": epochs,
            "batch_size": batch_size,
            "anomaly_calibration_quantile": anomaly_calibration_quantile,
            "xgboost_device": xgb_device,
            "neural_models": list(NEURAL_MODEL_NAMES),
        },
        "sampled_rows_per_class": sample.sampled_counts,
        "scanned_rows_per_class": sample.scanned_counts,
        "score_semantics": (
            "Classifier scores are model outputs, not guarantees of correctness "
            "and not calibrated unless separately validated."
        ),
        "neural_input_semantics": (
            "CNN, RNN, LSTM, and Transformer consume the dataset's fixed feature "
            "order. Recurrent and attention models treat feature indices as tokens; "
            "the CIC-IDS-2017 CSVs contain no per-flow temporal sequence for these "
            "models to learn."
        ),
        "artifacts": {
            name: str(path.relative_to(ARTIFACT_ROOT)).replace("\\", "/")
            for name, path in paths.items()
        },
    }
    _write_json(paths["manifest"], manifest)
    LOGGER.info("Evaluation metrics:\n%s", json.dumps(metrics, indent=2))
    LOGGER.info("Artifacts saved under %s", ARTIFACT_ROOT)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IMB models from CIC-IDS-2017 flow CSVs."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Existing extracted dataset directory; downloads through KaggleHub if omitted.",
    )
    parser.add_argument("--max-rows-per-class", type=int, default=5_000)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-estimators", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument(
        "--anomaly-calibration-quantile",
        type=float,
        default=0.95,
        help="Normal calibration quantile used as the reconstruction-error threshold.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--xgboost-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA for XGBoost only when explicitly requested or PyTorch CUDA is active.",
    )
    parser.add_argument("--overwrite-artifacts", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1.")
    if args.epochs < 1 or args.xgb_estimators < 1 or args.batch_size < 1:
        parser.error("--epochs, --xgb-estimators, and --batch-size must be positive.")
    if not 0 < args.anomaly_calibration_quantile < 1:
        parser.error("--anomaly-calibration-quantile must be between 0 and 1.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch installation.")
    xgb_cuda_supported = bool(xgb.build_info().get("USE_CUDA", False))
    if args.xgboost_device == "cuda" and not xgb_cuda_supported:
        parser.error("CUDA was requested for XGBoost, but this XGBoost build lacks CUDA support.")

    if args.dataset_dir is None:
        dataset_dir = download_cicids_dataset()
    else:
        dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {dataset_dir}")

    sample = sample_cicids_csvs(
        dataset_dir,
        max_rows_per_class=args.max_rows_per_class,
        chunk_rows=args.chunk_rows,
        seed=args.seed,
    )
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    xgb_device = (
        "cuda"
        if args.xgboost_device == "cuda"
        or (args.xgboost_device == "auto" and device.type == "cuda" and xgb_cuda_supported)
        else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    train_and_export(
        sample,
        max_rows_per_class=args.max_rows_per_class,
        test_fraction=args.test_fraction,
        seed=args.seed,
        xgb_estimators=args.xgb_estimators,
        epochs=args.epochs,
        batch_size=args.batch_size,
        anomaly_calibration_quantile=args.anomaly_calibration_quantile,
        device=device,
        xgb_device=xgb_device,
        overwrite=args.overwrite_artifacts,
    )


if __name__ == "__main__":
    main()
