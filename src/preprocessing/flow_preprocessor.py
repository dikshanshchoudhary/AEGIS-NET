from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlowPreprocessor:
    feature_names: tuple[str, ...]
    imputation_medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if width == 0:
            raise ValueError("The feature list must not be empty.")
        if len(set(self.feature_names)) != width:
            raise ValueError("Feature names must be unique.")
        if any(len(values) != width for values in (
            self.imputation_medians,
            self.means,
            self.scales,
        )):
            raise ValueError("Preprocessing arrays must match the feature list length.")
        if not all(np.isfinite(value) for values in (
            self.imputation_medians,
            self.means,
            self.scales,
        ) for value in values):
            raise ValueError("Preprocessing arrays must contain only finite numbers.")
        if any(value <= 0 for value in self.scales):
            raise ValueError("All scaling values must be positive.")

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...] | list[str],
    ) -> FlowPreprocessor:
        names = tuple(feature_names)
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise ValueError(f"Training frame is missing features: {missing}")

        values = frame.loc[:, names].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        usable = np.isfinite(values).any(axis=0)
        values = values[:, usable]
        names = tuple(name for name, keep in zip(names, usable) if keep)
        if not names:
            raise ValueError("No numeric features contain finite training values.")

        medians = np.nanmedian(values, axis=0)
        imputed = np.where(np.isfinite(values), values, medians)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0, ddof=0)
        scales[~np.isfinite(scales) | (scales <= 0)] = 1.0

        return cls(
            feature_names=names,
            imputation_medians=tuple(float(value) for value in medians),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
        )

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Input frame is missing required features: {missing}")

        values = frame.loc[:, self.feature_names].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        return self._transform(values)

    def transform_record(self, record: Mapping[str, object]) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in record]
        if missing:
            raise ValueError(f"Flow record is missing required features: {missing}")

        values = np.empty(len(self.feature_names), dtype=np.float64)
        for index, name in enumerate(self.feature_names):
            value = record[name]
            if value is None:
                values[index] = np.nan
                continue
            try:
                values[index] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Feature {name!r} must be numeric.") from error

        return self._transform(values.reshape(1, -1))[0]

    def _transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected a 2D feature matrix with {len(self.feature_names)} columns."
            )
        values = values.astype(np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        medians = np.asarray(self.imputation_medians)
        means = np.asarray(self.means)
        scales = np.asarray(self.scales)
        imputed = np.where(np.isfinite(values), values, medians)
        return ((imputed - means) / scales).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "transform": "impute non-finite values with training medians, then standardize",
            "feature_order": list(self.feature_names),
            "imputation_medians": list(self.imputation_medians),
            "means": list(self.means),
            "scales": list(self.scales),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path: Path) -> FlowPreprocessor:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported preprocessing schema in {path}.")
        try:
            return cls(
                feature_names=tuple(payload["feature_order"]),
                imputation_medians=tuple(float(value) for value in payload["imputation_medians"]),
                means=tuple(float(value) for value in payload["means"]),
                scales=tuple(float(value) for value in payload["scales"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid preprocessing artifact: {path}") from error
