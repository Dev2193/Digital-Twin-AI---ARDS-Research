"""Data contracts and split validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TabularContract:
    patient_id_column: str = "patient_id"
    label_column: str = "label"
    split_column: str = "split"

    @property
    def required_columns(self) -> set[str]:
        return {self.patient_id_column, self.label_column, self.split_column}


def validate_tabular_frame(frame: pd.DataFrame, contract: TabularContract) -> None:
    """Validate schema and guard against patient leakage across partitions."""
    missing = contract.required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    allowed_splits = {"train", "validation", "test"}
    observed_splits = set(frame[contract.split_column].dropna().unique())
    invalid_splits = observed_splits.difference(allowed_splits)
    if invalid_splits:
        raise ValueError(f"Unexpected split values: {sorted(invalid_splits)}")
    if observed_splits != allowed_splits:
        raise ValueError(
            f"Expected all splits {sorted(allowed_splits)}, "
            f"got {sorted(observed_splits)}"
        )

    labels = set(frame[contract.label_column].dropna().unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError("Label must be binary and contain both classes.")

    patient_split_counts = (
        frame[[contract.patient_id_column, contract.split_column]]
        .drop_duplicates()
        .groupby(contract.patient_id_column)[contract.split_column]
        .nunique()
    )
    leaking = patient_split_counts[patient_split_counts > 1]
    if not leaking.empty:
        raise ValueError(f"{len(leaking)} patients appear in multiple data splits.")

    feature_columns = [
        column for column in frame.columns if column not in contract.required_columns
    ]
    if not feature_columns:
        raise ValueError("No feature columns were provided.")

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"Baseline features must be numeric or boolean: {non_numeric}")


def partition_frame(
    frame: pd.DataFrame, contract: TabularContract
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Return feature and label partitions with identifiers removed."""
    feature_columns = [
        column for column in frame.columns if column not in contract.required_columns
    ]
    partitions: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for split in ("train", "validation", "test"):
        split_frame = frame.loc[frame[contract.split_column] == split]
        partitions[split] = (
            split_frame[feature_columns].copy(),
            split_frame[contract.label_column].astype(int).copy(),
        )
    return partitions
