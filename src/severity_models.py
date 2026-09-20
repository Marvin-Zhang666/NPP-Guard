"""Small, reproducible grouped-regression helpers for LOCA severity."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 20260920


def make_group_split(sample_ids: list[str] | pd.Series) -> pd.DataFrame:
    """Assign complete trajectories to deterministic 60/20/20 groups.

    Sorted severity ranks cycle through test, validation, and train. This gives
    every severity band a presence in every split without using feature values.
    """
    unique_ids = sorted(
        {str(sample_id) for sample_id in sample_ids},
        key=lambda value: int(re.search(r"(\d+)$", value).group(1)),
    )
    rows = []
    for rank, sample_id in enumerate(unique_ids):
        remainder = rank % 5
        split = "test" if remainder == 0 else "validation" if remainder == 1 else "train"
        rows.append(
            {
                "sample_id": sample_id,
                "severity": int(re.search(r"(\d+)$", sample_id).group(1)),
                "split": split,
            }
        )
    result = pd.DataFrame(rows)
    if result["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in grouped split")
    return result


def model_suite(random_state: int = RANDOM_STATE) -> dict[str, object | None]:
    """Return the requested baseline models without adding dependencies."""
    return {
        "naive_mean": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=random_state,
        ),
    }


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_total if ss_total else 0.0
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": r2,
    }


def suspicious_train_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Flag only train-derived near-perfect linear label correlations."""
    correlations = x_train.apply(
        lambda column: column.corr(pd.Series(y_train, index=x_train.index))
    ).dropna()
    result = pd.DataFrame(
        {
            "feature": correlations.index,
            "train_abs_correlation": correlations.abs().to_numpy(),
            "flagged": correlations.abs().ge(threshold).to_numpy(),
        }
    ).sort_values("train_abs_correlation", ascending=False).reset_index(drop=True)
    return result


def feature_importance_table(model: object | None, feature_names: list[str]) -> pd.DataFrame:
    """Extract a comparable absolute-importance table for fitted models."""
    if model is None:
        values = np.zeros(len(feature_names), dtype=float)
    elif hasattr(model, "named_steps") and "ridge" in model.named_steps:
        values = np.abs(model.named_steps["ridge"].coef_)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.abs(model[-1].coef_)
    return pd.DataFrame(
        {"feature": feature_names, "importance": values}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
