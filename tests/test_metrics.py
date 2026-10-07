import numpy as np

from ards_ai.metrics import binary_metrics


def test_binary_metrics_are_aggregate() -> None:
    metrics = binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert metrics["auprc"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
