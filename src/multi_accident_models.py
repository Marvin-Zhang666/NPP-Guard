"""Leakage-safe grouped classification baselines for the 10 milestone."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 20260920
SUMMARY_STATISTICS = ("last", "delta_t0", "mean", "std", "slope")


def model_suite(random_state: int = RANDOM_STATE) -> dict[str, object | None]:
    """Return the requested traditional baselines without new dependencies."""
    return {
        "naive_majority": None,
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=random_state),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def _majority_prediction(y_train: pd.Series, length: int) -> np.ndarray:
    counts = y_train.value_counts().sort_index()
    return np.repeat(counts.index[counts.to_numpy().argmax()], length)


def evaluate_models(
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    class_labels: list[str],
    windows_s: tuple[int, ...] = (30, 60, 120),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on train trajectories and score validation/test trajectories."""
    metric_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for window_s in windows_s:
        window = dataset.loc[dataset["window_s"].eq(window_s)].copy()
        for group_name, columns in feature_groups.items():
            model_columns = [
                f"{feature}__{statistic}"
                for feature in columns
                for statistic in SUMMARY_STATISTICS
            ]
            train = window.loc[window["split"].eq("train")]
            if train.empty or set(train["accident_class"]) != set(class_labels):
                raise AssertionError(f"Training split is incomplete for {group_name}/{window_s}s")
            for model_name, model in model_suite().items():
                if model is None:
                    train_majority = train["accident_class"].value_counts().sort_index().index[0]
                    fitted = None
                else:
                    fitted = model.fit(train[model_columns], train["accident_class"])
                    train_majority = None

                for eval_split in ("validation", "test"):
                    evaluated = window.loc[window["split"].eq(eval_split)]
                    y_true = evaluated["accident_class"].to_numpy()
                    y_pred = (
                        np.repeat(train_majority, len(evaluated))
                        if fitted is None
                        else fitted.predict(evaluated[model_columns])
                    )
                    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
                    metric_rows.append(
                        {
                            "window_s": int(window_s),
                            "input_group": group_name,
                            "model": model_name,
                            "eval_split": eval_split,
                            "n_samples": int(len(evaluated)),
                            "n_features": int(len(model_columns)),
                            "accuracy": float(accuracy_score(y_true, y_pred)),
                            "macro_f1": float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
                            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                            "train_majority_class": train_majority or "",
                        }
                    )
                    recalls = recall_score(
                        y_true, y_pred, labels=class_labels, average=None, zero_division=0
                    )
                    supports = np.bincount(
                        pd.Categorical(y_true, categories=class_labels).codes,
                        minlength=len(class_labels),
                    )
                    for label, recall, support, row in zip(class_labels, recalls, supports, matrix):
                        recall_rows.append(
                            {
                                "window_s": int(window_s),
                                "input_group": group_name,
                                "model": model_name,
                                "eval_split": eval_split,
                                "accident_class": label,
                                "recall": float(recall),
                                "support": int(support),
                                "correct": int(row[class_labels.index(label)]),
                            }
                        )
                    for true_label, row in zip(class_labels, matrix):
                        for predicted_label, count in zip(class_labels, row):
                            confusion_rows.append(
                                {
                                    "window_s": int(window_s),
                                    "input_group": group_name,
                                    "model": model_name,
                                    "eval_split": eval_split,
                                    "true_class": true_label,
                                    "predicted_class": predicted_label,
                                    "count": int(count),
                                }
                            )

    return pd.DataFrame(metric_rows), pd.DataFrame(recall_rows), pd.DataFrame(confusion_rows)


def select_best_validation(
    metrics: pd.DataFrame,
    input_group: str | None = None,
    windows_s: tuple[int, ...] = (30, 60),
) -> pd.Series:
    """Select a setting on validation only, then report its held-out test row."""
    candidates = metrics.loc[
        metrics["eval_split"].eq("validation") & metrics["window_s"].isin(windows_s)
    ].copy()
    if input_group is not None:
        candidates = candidates.loc[candidates["input_group"].eq(input_group)]
    if candidates.empty:
        raise ValueError("No validation candidates available")
    best = candidates.sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy", "window_s", "model"],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    return best


def selected_test_row(metrics: pd.DataFrame, selection: pd.Series) -> pd.Series:
    mask = (
        metrics["eval_split"].eq("test")
        & metrics["window_s"].eq(selection["window_s"])
        & metrics["input_group"].eq(selection["input_group"])
        & metrics["model"].eq(selection["model"])
    )
    rows = metrics.loc[mask]
    if len(rows) != 1:
        raise ValueError("Selected validation setting has no unique test row")
    return rows.iloc[0]
