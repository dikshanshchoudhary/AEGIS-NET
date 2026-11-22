from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ID = "chethuhn/network-intrusion-dataset"
CSV_ENCODING = "cp1252"
LABEL_COLUMNS = {"label", "class"}
IDENTIFIER_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "source address",
    "destination address",
    "timestamp",
}


@dataclass(frozen=True)
class FlowDatasetSample:
    features: pd.DataFrame
    labels: np.ndarray
    scanned_counts: dict[str, int]
    sampled_counts: dict[str, int]


def normalize_column_name(value: object) -> str:
    without_bom = str(value).replace(chr(65279), "")
    return " ".join(without_bom.strip().split())


def normalize_label(value: object) -> str:
    label = " ".join(str(value).strip().split())
    return label.replace("ï¿½", "-").replace("\ufffd", "-")


def download_cicids_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "KaggleHub is required to download the dataset. Install project requirements."
        ) from error

    return Path(kagglehub.dataset_download(DATASET_ID))


def _read_schema(csv_path: Path) -> tuple[list[str], str, list[str], list[str]]:
    raw_columns = pd.read_csv(
        csv_path,
        nrows=0,
        encoding=CSV_ENCODING,
    ).columns.tolist()
    normalized_columns = [normalize_column_name(name) for name in raw_columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError(f"Duplicate columns after normalization in {csv_path.name}.")

    label_columns = [
        name for name in normalized_columns
        if name.casefold() in LABEL_COLUMNS
    ]
    if len(label_columns) != 1:
        raise ValueError(
            f"Expected exactly one Label/Class column in {csv_path.name}; "
            f"found {label_columns}."
        )

    label_column = label_columns[0]
    feature_columns = [
        name for name in normalized_columns
        if name != label_column
        and name.casefold() not in LABEL_COLUMNS
        and name.casefold() not in IDENTIFIER_COLUMNS
    ]
    selected_raw_columns = [
        raw for raw, normalized in zip(raw_columns, normalized_columns)
        if normalized == label_column or normalized in feature_columns
    ]
    return (
        raw_columns,
        label_column,
        feature_columns,
        selected_raw_columns,
    )


def _update_reservoir(
    label: str,
    new_rows: np.ndarray,
    reservoirs: dict[str, np.ndarray],
    seen_counts: dict[str, int],
    max_rows_per_class: int,
    rng: np.random.Generator,
) -> None:
    previous_count = seen_counts.get(label, 0)
    current = reservoirs.get(label)
    if current is None:
        current = np.empty((0, new_rows.shape[1]), dtype=np.float32)

    add_count = min(max(max_rows_per_class - previous_count, 0), len(new_rows))
    if add_count:
        current = np.concatenate((current, new_rows[:add_count]), axis=0)

    remaining = new_rows[add_count:]
    if len(remaining):
        row_numbers = previous_count + np.arange(
            add_count + 1,
            len(new_rows) + 1,
        )
        slots = (rng.random(len(remaining)) * row_numbers).astype(np.int64)
        replace = slots < max_rows_per_class
        for slot, row in zip(slots[replace], remaining[replace]):
            current[slot] = row

    reservoirs[label] = current
    seen_counts[label] = previous_count + len(new_rows)


def sample_cicids_csvs(
    dataset_dir: Path,
    *,
    max_rows_per_class: int = 5_000,
    chunk_rows: int = 100_000,
    seed: int = 42,
) -> FlowDatasetSample:
    if max_rows_per_class < 2:
        raise ValueError("max_rows_per_class must be at least 2.")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")

    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {dataset_dir}.")

    expected_features: list[str] | None = None
    expected_schema: set[str] | None = None
    reservoirs: dict[str, np.ndarray] = {}
    seen_counts: dict[str, int] = {}
    rng = np.random.default_rng(seed)

    for csv_path in csv_files:
        _, label_column, feature_columns, selected_raw_columns = _read_schema(csv_path)
        schema = set(feature_columns) | {label_column}
        if expected_schema is None:
            expected_schema = schema
            expected_features = feature_columns
        elif schema != expected_schema:
            raise ValueError(
                f"Feature schema differs in {csv_path.name}; "
                "refusing to silently invent or omit features."
            )

        for chunk in pd.read_csv(
            csv_path,
            usecols=selected_raw_columns,
            chunksize=chunk_rows,
            encoding=CSV_ENCODING,
            low_memory=False,
        ):
            chunk.columns = [normalize_column_name(name) for name in chunk.columns]
            raw_labels = chunk[label_column].astype("string").str.strip()
            valid_labels = raw_labels.notna() & raw_labels.ne("")
            if not valid_labels.any():
                continue

            values = chunk.loc[valid_labels, expected_features].apply(
                pd.to_numeric,
                errors="coerce",
            ).to_numpy(dtype=np.float32, copy=True)
            values[~np.isfinite(values)] = np.nan
            labels = raw_labels.loc[valid_labels].map(normalize_label).to_numpy()

            for label in np.unique(labels):
                label_rows = values[labels == label]
                _update_reservoir(
                    str(label),
                    label_rows,
                    reservoirs,
                    seen_counts,
                    max_rows_per_class,
                    rng,
                )

    if not reservoirs or expected_features is None:
        raise ValueError("No labeled flow rows were read from the dataset.")

    class_names = sorted(reservoirs)
    feature_matrix = np.concatenate(
        [reservoirs[name] for name in class_names],
        axis=0,
    )
    labels = np.concatenate([
        np.full(len(reservoirs[name]), name, dtype=object)
        for name in class_names
    ])
    usable_features = np.isfinite(feature_matrix).any(axis=0)
    feature_names = [
        name for name, usable in zip(expected_features, usable_features)
        if usable
    ]
    feature_matrix = feature_matrix[:, usable_features]
    if not feature_names:
        raise ValueError("No numeric flow features remained after input validation.")

    return FlowDatasetSample(
        features=pd.DataFrame(feature_matrix, columns=feature_names),
        labels=labels,
        scanned_counts=seen_counts,
        sampled_counts={
            name: len(reservoirs[name])
            for name in class_names
        },
    )
