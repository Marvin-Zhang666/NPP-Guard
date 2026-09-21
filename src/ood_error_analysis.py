"""Directional error decomposition for severity extrapolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

try:
    from .ood_validation import CLASS_LABELS
except ImportError:
    from ood_validation import CLASS_LABELS


def _main_confusion_target(true_values: pd.Series, predicted_values: pd.Series, true_label: str) -> str:
    counts = predicted_values.loc[true_values.eq(true_label)].value_counts()
    counts = counts.drop(labels=[true_label], errors="ignore")
    return str(counts.index[0]) if not counts.empty else ""


def directional_error_decomposition(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    input_group: str = "A_absolute_38",
    window_s: int = 120,
) -> dict[str, pd.DataFrame]:
    """Return split-level and direction-level metrics with severity coverage metadata."""
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    active = assignments.loc[assignments["partition"].isin(["train", "validation", "test"])]
    for direction, split in active.loc[active["split_type"].eq("severity_extrapolation")].groupby("split_id", sort=False):
        split_type = "severity_extrapolation"
        train_assign = split.loc[split["partition"].eq("train")]
        test_assign = split.loc[split["partition"].eq("test")]
        test_predictions = predictions.loc[
            predictions["split_type"].eq(split_type)
            & predictions["split_id"].eq(direction)
            & predictions["window_s"].eq(window_s)
            & predictions["input_group"].eq(input_group)
        ].copy()
        if test_predictions.empty:
            raise AssertionError(f"No selected baseline predictions for {direction}")
        y_true = test_predictions["true_class"].to_numpy()
        y_pred = test_predictions["predicted_class"].to_numpy()
        for true_label in CLASS_LABELS:
            class_test = test_assign.loc[test_assign["accident_class"].eq(true_label)]
            class_predictions = test_predictions.loc[test_predictions["true_class"].eq(true_label)]
            support = len(class_predictions)
            if support:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[true_label], average=None, zero_division=0
                )
                gaps = []
                train_values = train_assign.loc[train_assign["accident_class"].eq(true_label), "severity"].to_numpy(dtype=float)
                for severity in class_test["severity"].to_numpy(dtype=float):
                    gaps.append(float(np.min(np.abs(train_values - severity))) if len(train_values) else np.nan)
                test_values = class_test["severity"].to_numpy(dtype=float)
                gap_mean = float(np.nanmean(gaps)) if gaps else np.nan
                train_min = float(np.min(train_values)) if len(train_values) else np.nan
                train_max = float(np.max(train_values)) if len(train_values) else np.nan
                test_min = float(np.min(test_values)) if len(test_values) else np.nan
                test_max = float(np.max(test_values)) if len(test_values) else np.nan
                main_target = _main_confusion_target(
                    class_predictions["true_class"], class_predictions["predicted_class"], true_label
                )
                rows.append(
                    {
                        "split_type": split_type,
                        "direction": direction,
                        "window_s": int(window_s),
                        "input_group": input_group,
                        "accident_class": true_label,
                        "precision": float(precision[0]),
                        "recall": float(recall[0]),
                        "f1": float(f1[0]),
                        "test_support": int(support),
                        "main_confusion_target": main_target,
                        "nearest_severity_gap_mean": gap_mean,
                        "train_severity_min": train_min,
                        "train_severity_max": train_max,
                        "test_severity_min": test_min,
                        "test_severity_max": test_max,
                        "n_train_class": int(len(train_values)),
                    }
                )
            else:
                rows.append(
                    {
                        "split_type": split_type,
                        "direction": direction,
                        "window_s": int(window_s),
                        "input_group": input_group,
                        "accident_class": true_label,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "test_support": 0,
                        "main_confusion_target": "",
                        "nearest_severity_gap_mean": np.nan,
                        "train_severity_min": np.nan,
                        "train_severity_max": np.nan,
                        "test_severity_min": np.nan,
                        "test_severity_max": np.nan,
                        "n_train_class": 0,
                    }
                )
        matrix = pd.crosstab(
            test_predictions["true_class"], test_predictions["predicted_class"],
        ).reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0)
        for true_label in CLASS_LABELS:
            for predicted_label in CLASS_LABELS:
                confusion_rows.append(
                    {
                        "split_type": split_type,
                        "direction": direction,
                        "window_s": int(window_s),
                        "input_group": input_group,
                        "true_class": true_label,
                        "predicted_class": predicted_label,
                        "count": int(matrix.loc[true_label, predicted_label]),
                    }
                )
    decomposition = pd.DataFrame(rows)
    if decomposition.empty:
        raise AssertionError("Directional extrapolation decomposition is empty")
    aggregate_rows: list[dict[str, Any]] = []
    evaluable = decomposition.loc[decomposition["test_support"].gt(0)]
    for (direction, accident_class), group in decomposition.groupby(["direction", "accident_class"], sort=False):
        support_group = group.loc[group["test_support"].gt(0)]
        targets = support_group["main_confusion_target"].replace("", np.nan).dropna()
        row: dict[str, Any] = {
            "split_type": "severity_extrapolation",
            "direction": direction,
            "window_s": int(window_s),
            "input_group": input_group,
            "accident_class": accident_class,
            "n_splits": int(len(support_group)),
            "test_support_mean": float(support_group["test_support"].mean()) if not support_group.empty else 0.0,
            "test_support_min": int(support_group["test_support"].min()) if not support_group.empty else 0,
            "main_confusion_target": str(targets.mode().iloc[0]) if not targets.empty else "",
        }
        for metric in ("precision", "recall", "f1", "nearest_severity_gap_mean"):
            row[f"{metric}_mean"] = float(support_group[metric].mean()) if not support_group.empty else np.nan
            row[f"{metric}_std"] = float(support_group[metric].std(ddof=1)) if len(support_group) > 1 else 0.0
        for field in ("train_severity_min", "train_severity_max", "test_severity_min", "test_severity_max"):
            row[field] = float(support_group[field].mean()) if not support_group.empty else np.nan
        aggregate_rows.append(row)
    directional = pd.DataFrame(aggregate_rows)
    confusion = pd.DataFrame(confusion_rows)
    recall_gap = decomposition.loc[decomposition["test_support"].gt(0), [
        "direction", "accident_class", "recall", "nearest_severity_gap_mean", "test_support"
    ]].copy()
    return {"decomposition": decomposition, "directional": directional, "confusion": confusion, "recall_gap": recall_gap}


def write_outputs(result: dict[str, pd.DataFrame], result_root: str | Path) -> None:
    root = Path(result_root)
    result["decomposition"].to_csv(root / "14_ood_error_decomposition.csv", index=False)
    result["directional"].to_csv(root / "14_directional_extrapolation_metrics.csv", index=False)
    result["confusion"].to_csv(root / "14_confusion_by_direction.csv", index=False)
    result["recall_gap"].to_csv(root / "14_recall_vs_gap.csv", index=False)
