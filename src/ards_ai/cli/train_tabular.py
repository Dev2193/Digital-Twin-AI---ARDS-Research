"""Train privacy-safe aggregate tabular baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ards_ai.data import TabularContract, partition_frame, validate_tabular_frame
from ards_ai.metrics import binary_metrics
from ards_ai.models.tabular import build_logistic_regression, build_xgboost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def ensure_outside_repository(path: Path) -> None:
    """Reject output destinations inside the Git working tree."""
    resolved = path.expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if resolved == repository or repository in resolved.parents:
        raise ValueError("Output must be outside the public Git repository.")


def main() -> None:
    args = parse_args()
    ensure_outside_repository(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    frame = pd.read_parquet(args.input)
    contract = TabularContract(
        patient_id_column=config["patient_id_column"],
        label_column=config["label_column"],
        split_column=config["split_column"],
    )
    validate_tabular_frame(frame, contract)
    partitions = partition_frame(frame, contract)
    x_train, y_train = partitions["train"]
    x_validation, y_validation = partitions["validation"]
    x_test, y_test = partitions["test"]

    models = {
        "logistic": build_logistic_regression(
            list(x_train.columns),
            config["logistic_regression"],
            int(config["random_seed"]),
        ),
        "xgboost": build_xgboost(config["xgboost"], int(config["random_seed"])),
    }

    results: dict[str, dict[str, float | int]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=config["calibration"]["method"],
        )
        calibrated.fit(x_validation, y_validation)
        probabilities = calibrated.predict_proba(x_test)[:, 1]
        results[name] = binary_metrics(y_test.to_numpy(), probabilities)
        joblib.dump(calibrated, args.output / f"{name}_calibrated.joblib")

    with (args.output / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
