"""Transparent late fusion for independently trained component models."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_late_fusion(random_seed: int = 42) -> LogisticRegression:
    """Combine out-of-fold component probabilities with logistic regression."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=random_seed,
    )
