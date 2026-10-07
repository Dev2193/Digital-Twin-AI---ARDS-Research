import pandas as pd
import pytest

from ards_ai.data import TabularContract, partition_frame, validate_tabular_frame


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5, 6],
            "label": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "heart_rate_mean": [80.0, 110.0, 75.0, 105.0, 82.0, 115.0],
            "lactate_latest": [1.0, 3.5, 0.9, 2.8, 1.2, 4.0],
        }
    )


def test_valid_frame_partitions() -> None:
    frame = synthetic_frame()
    contract = TabularContract()
    validate_tabular_frame(frame, contract)
    partitions = partition_frame(frame, contract)
    assert set(partitions) == {"train", "validation", "test"}
    assert list(partitions["train"][0].columns) == ["heart_rate_mean", "lactate_latest"]


def test_patient_leakage_is_rejected() -> None:
    frame = synthetic_frame()
    frame.loc[2, "patient_id"] = 1
    with pytest.raises(ValueError, match="multiple data splits"):
        validate_tabular_frame(frame, TabularContract())
