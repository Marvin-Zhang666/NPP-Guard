"""Small grouped-regression helpers for leakage-safe protection-time prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .severity_models import (
        make_group_split,
        model_suite,
        regression_metrics,
        suspicious_train_features,
    )
except ImportError:
    from severity_models import (
        make_group_split,
        model_suite,
        regression_metrics,
        suspicious_train_features,
    )


INPUT_GROUPS = ("severity_only", "process_only", "severity_plus_process")
SPLITS = ("train", "validation", "test")


def process_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return only derived process columns, excluding labels and metadata."""
    return [column for column in dataset.columns if "__" in column]


def _base_feature(column: str) -> str:
    return column.split("__", 1)[0]


def _feature_columns(
    input_group: str,
    process_columns: list[str],
    excluded_bases: set[str],
) -> list[str]:
    available = [
        column for column in process_columns
        if _base_feature(column) not in excluded_bases
    ]
    if input_group == "severity_only":
        return ["severity"]
    if input_group == "process_only":
        return available
    if input_group == "severity_plus_process":
        return ["severity", *available]
    raise ValueError(f"Unknown input group: {input_group}")


def run_experiment(
    dataset: pd.DataFrame,
    mode: str,
    excluded_bases: set[str] | None = None,
    random_state: int = 20260920,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit fixed baseline models using train only and return metrics/predictions.

    The deterministic grouped split is rebuilt independently for each landmark,
    because already-protected trajectories are removed before splitting.
    """
    excluded_bases = excluded_bases or set()
    process_columns = process_feature_columns(dataset)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for landmark_s in sorted(dataset["landmark_s"].unique()):
        current = dataset.loc[dataset["landmark_s"].eq(landmark_s)].copy()
        split = make_group_split(current["sample_id"].tolist())
        split_map = dict(zip(split["sample_id"], split["split"], strict=True))
        current["split"] = current["sample_id"].map(split_map)
        split_rows.extend(
            {
                "mode": mode,
                "landmark_s": int(landmark_s),
                **row,
            }
            for row in split.to_dict(orient="records")
        )

        train_mask = current["split"].eq("train")
        y_train = current.loc[train_mask, "remaining_time_s"]
        available_process = [
            column for column in process_columns
            if _base_feature(column) not in excluded_bases
        ]
        varying_process = [
            column
            for column in available_process
            if current.loc[train_mask, column].nunique(dropna=False) > 1
        ]
        audit = (
            suspicious_train_features(
                current.loc[train_mask, varying_process], y_train
            )
            if varying_process
            else pd.DataFrame(columns=["feature", "train_abs_correlation", "flagged"])
        )
        for row in audit.to_dict(orient="records"):
            audit_rows.append(
                {
                    "mode": mode,
                    "landmark_s": int(landmark_s),
                    "base_feature": _base_feature(str(row["feature"])),
                    **row,
                }
            )
        suspicious_count = int(audit["flagged"].sum())
        excluded_count = len(process_columns) - len(available_process)

        for input_group in INPUT_GROUPS:
            feature_columns = _feature_columns(
                input_group, process_columns, excluded_bases
            )
            x = current[feature_columns]
            y = current["remaining_time_s"]
            train_x = x.loc[train_mask]
            for model_name, model in model_suite(random_state).items():
                if model_name == "naive_mean":
                    predictions = np.full(len(current), float(y_train.mean()))
                else:
                    model.fit(train_x, y_train)
                    predictions = model.predict(x)
                for split_name in SPLITS:
                    mask = current["split"].eq(split_name).to_numpy()
                    values = regression_metrics(y.loc[mask], predictions[mask])
                    metric_rows.append(
                        {
                            "mode": mode,
                            "landmark_s": int(landmark_s),
                            "input_group": input_group,
                            "model": model_name,
                            "split": split_name,
                            "n_trajectories": int(mask.sum()),
                            "feature_count": len(feature_columns),
                            "excluded_high_coupling_count": excluded_count,
                            "suspicious_train_feature_count": suspicious_count,
                            **values,
                        }
                    )
                    for row_index in np.flatnonzero(mask):
                        row = current.iloc[row_index]
                        prediction_rows.append(
                            {
                                "mode": mode,
                                "landmark_s": int(landmark_s),
                                "input_group": input_group,
                                "model": model_name,
                                "split": split_name,
                                "sample_id": row["sample_id"],
                                "severity": int(row["severity"]),
                                "first_protection_s": float(row["first_protection_s"]),
                                "remaining_time_s": float(row["remaining_time_s"]),
                                "prediction": float(predictions[row_index]),
                            }
                        )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(audit_rows),
    )
