"""Logistic-regression and XGBoost model builders."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_logistic_regression(
    feature_columns: list[str], config: dict[str, Any], random_seed: int
) -> Pipeline:
    """Build an imputed and standardized L2 logistic-regression pipeline."""
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=float(config.get("C", 1.0)),
        class_weight=config.get("class_weight", "balanced"),
        max_iter=int(config.get("max_iter", 2000)),
        random_state=random_seed,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def build_xgboost(config: dict[str, Any], random_seed: int) -> XGBClassifier:
    """Build an XGBoost classifier optimized externally with AUPRC."""
    return XGBClassifier(
        n_estimators=int(config.get("n_estimators", 500)),
        max_depth=int(config.get("max_depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.03)),
        subsample=float(config.get("subsample", 0.8)),
        colsample_bytree=float(config.get("colsample_bytree", 0.8)),
        min_child_weight=float(config.get("min_child_weight", 1.0)),
        reg_alpha=float(config.get("reg_alpha", 0.0)),
        reg_lambda=float(config.get("reg_lambda", 1.0)),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=random_seed,
        n_jobs=-1,
    )
