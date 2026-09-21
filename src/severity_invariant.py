"""Severity-invariant feature experiments on the milestone-13 OOD splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

try:
    from .multi_accident_models import model_suite
    from .ood_validation import (
        CLASS_LABELS,
        DEFAULT_INVENTORY,
        DEFAULT_EVENT_INVENTORY,
        build_ood_feature_dataset,
    )
    from .multi_accident_features import strict_process_features
    from .severity_features import SUMMARY_STATISTICS
except ImportError:
    from multi_accident_models import model_suite
    from ood_validation import CLASS_LABELS, DEFAULT_INVENTORY, DEFAULT_EVENT_INVENTORY, build_ood_feature_dataset
    from multi_accident_features import strict_process_features
    from severity_features import SUMMARY_STATISTICS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
WINDOWS_S = (60, 90, 120)
MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
GROUP_NAMES = ("A_absolute_38", "B_relative_change", "C_train_only_screened")


def feature_columns(features: list[str], statistics: tuple[str, ...] = SUMMARY_STATISTICS) -> list[str]:
    return [f"{feature}__{statistic}" for feature in features for statistic in statistics]


def _describe(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(numeric)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if n > 1 else 0.0
    margin = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {"n": int(n), "mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin}


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...] = CLASS_LABELS) -> dict[str, Any]:
    supports = pd.Series(y_true).value_counts().reindex(labels, fill_value=0).to_numpy(dtype=int)
    observed = [label for label, support in zip(labels, supports) if support > 0]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_fixed_12": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "macro_f1_observed_classes": float(
            f1_score(y_true, y_pred, labels=observed, average="macro", zero_division=0)
        ) if observed else np.nan,
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=observed, average="macro", zero_division=0)) if observed else np.nan,
        "n_classes_eval": int(len(observed)),
        "minimum_test_support": int(supports[supports > 0].min()) if np.any(supports > 0) else 0,
    }


def _spearman_abs(left: pd.Series, right: pd.Series) -> float:
    values = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(values) < 4 or values["left"].nunique() < 2 or values["right"].nunique() < 2:
        return 0.0
    return float(abs(values["left"].rank(method="average").corr(values["right"].rank(method="average"))))


def _class_signal(values: pd.Series, labels: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    clean = pd.DataFrame({"value": values, "label": labels}).dropna()
    if clean.empty or clean["label"].nunique() < 2 or clean["value"].nunique() < 2:
        return 0.0
    grand = float(clean["value"].mean())
    between = 0.0
    within = 0.0
    for _, group in clean.groupby("label", sort=False):
        mean = float(group["value"].mean())
        between += len(group) * (mean - grand) ** 2
        within += float(((group["value"] - mean) ** 2).sum())
    return float(between / (between + within + 1e-12))


def screen_train_features(
    train: pd.DataFrame,
    features: list[str],
    window_s: int,
    split_type: str,
    split_id: str,
) -> tuple[list[str], pd.DataFrame]:
    """Screen raw variables using train-only severity dependence and class signal.

    The rule is fixed before looking at validation/test: remove variables with a
    maximum absolute Spearman severity association >= 0.70 when their maximum
    train-only class-signal score is no larger than the train-only median.
    """
    rows: list[dict[str, Any]] = []
    severity = train["severity"]
    labels = train["accident_class"]
    signal_by_feature: dict[str, float] = {}
    dependency_by_feature: dict[str, float] = {}
    for feature in features:
        dependencies = []
        signals = []
        for statistic in SUMMARY_STATISTICS:
            column = f"{feature}__{statistic}"
            dependencies.append(_spearman_abs(train[column], severity))
            signals.append(_class_signal(train[column], labels))
        dependency_by_feature[feature] = max(dependencies, default=0.0)
        signal_by_feature[feature] = max(signals, default=0.0)
    signal_threshold = float(np.median(list(signal_by_feature.values()))) if signal_by_feature else 0.0
    for feature in features:
        dependency = dependency_by_feature[feature]
        signal = signal_by_feature[feature]
        removed = bool(dependency >= 0.70 and signal <= signal_threshold)
        rows.append(
            {
                "split_type": split_type,
                "split_id": split_id,
                "window_s": int(window_s),
                "feature": feature,
                "severity_dependency_max_abs_spearman": dependency,
                "class_signal_max_between_share": signal,
                "class_signal_train_median": signal_threshold,
                "removed": removed,
                "screening_rule": "train_only: severity_abs_spearman>=0.70 and class_signal<=train_median",
            }
        )
    removed_features = {row["feature"] for row in rows if row["removed"]}
    retained = [feature for feature in features if feature not in removed_features]
    if len(retained) < 4:
        keep = sorted(features, key=lambda item: signal_by_feature[item], reverse=True)[:4]
        retained = [feature for feature in features if feature in keep]
        for row in rows:
            row["removed"] = row["feature"] not in retained
            row["screening_rule"] += "; minimum_retained_features=4"
    return retained, pd.DataFrame(rows)


def relative_change_matrix(frame: pd.DataFrame, features: list[str], window_s: int) -> pd.DataFrame:
    """Create a row-wise change representation without absolute last/mean values."""
    output: dict[str, pd.Series] = {}
    for feature in features:
        last = pd.to_numeric(frame[f"{feature}__last"], errors="coerce")
        delta = pd.to_numeric(frame[f"{feature}__delta_t0"], errors="coerce")
        mean = pd.to_numeric(frame[f"{feature}__mean"], errors="coerce")
        std = pd.to_numeric(frame[f"{feature}__std"], errors="coerce")
        slope = pd.to_numeric(frame[f"{feature}__slope"], errors="coerce")
        t0 = last - delta
        scale = pd.concat([t0.abs(), last.abs(), mean.abs(), pd.Series(1.0, index=frame.index)], axis=1).max(axis=1)
        output[f"{feature}__delta_t0"] = delta
        output[f"{feature}__mean_delta_t0"] = mean - t0
        output[f"{feature}__delta_norm"] = delta / scale
        output[f"{feature}__slope_norm"] = slope * float(window_s) / scale
        output[f"{feature}__std_norm"] = std / scale
    result = pd.DataFrame(output, index=frame.index)
    if result.isna().any().any():
        raise AssertionError("Relative-change feature matrix contains missing values")
    return result


def load_ood_context(
    project_root: str | Path = PROJECT_ROOT,
    inventory_path: str | Path = DEFAULT_INVENTORY,
    event_inventory_path: str | Path = DEFAULT_EVENT_INVENTORY,
    assignments_path: str | Path = RESULT_ROOT / "13_split_inventory.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _ = build_ood_feature_dataset(
        project_root=project_root,
        inventory_path=inventory_path,
        event_inventory_path=event_inventory_path,
        windows_s=WINDOWS_S,
        classes=CLASS_LABELS,
    )
    assignments = pd.read_csv(assignments_path)
    active_ids = set(assignments.loc[assignments["partition"].isin(["train", "validation", "test"]), "sample_id"])
    dataset = dataset.loc[dataset["sample_id"].isin(active_ids)].copy()
    if dataset["sample_id"].nunique() != 505:
        raise AssertionError("14 did not reuse the 505-trajectory milestone-13 matched cohort")
    if dataset[["sample_id", "window_s"]].duplicated().any():
        raise AssertionError("14 feature dataset has duplicate trajectory-window rows")
    dataset["family"] = dataset["accident_class"]
    return dataset, assignments


def _prediction_rows(
    key: tuple[str, str, int, str, str, str],
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severities: pd.Series,
) -> list[dict[str, Any]]:
    split_type, split_id, window_s, group_name, model_name, _eval_split = key
    return [
        {
            "split_type": split_type,
            "split_id": split_id,
            "window_s": int(window_s),
            "input_group": group_name,
            "model": model_name,
            "sample_id": sample_id,
            "true_class": true,
            "predicted_class": predicted,
            "severity": float(severity),
        }
        for sample_id, true, predicted, severity in zip(sample_ids, y_true, y_pred, severities)
    ]


def evaluate_feature_groups(dataset: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Evaluate A/B/C with identical milestone-13 splits and train-only selection."""
    strict = strict_process_features()
    feature_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_store: dict[tuple[str, str, int, str, str, str], tuple[pd.Series, np.ndarray, np.ndarray, pd.Series]] = {}

    for (split_type, split_id), split in assignments.groupby(["split_type", "split_id"], sort=False):
        active = split.loc[split["partition"].isin(["train", "validation", "test"])]
        split_frame = dataset.merge(active[["sample_id", "partition"]], on="sample_id", how="inner", validate="many_to_one")
        seed_values = split["seed"].dropna()
        seed = int(seed_values.iloc[0]) if not seed_values.empty else 20260921
        for window_s in WINDOWS_S:
            window = split_frame.loc[split_frame["window_s"].eq(window_s)].copy()
            train = window.loc[window["partition"].eq("train")]
            validation = window.loc[window["partition"].eq("validation")]
            test = window.loc[window["partition"].eq("test")]
            if train.empty or validation.empty or test.empty:
                raise AssertionError(f"Empty 14 partition: {split_type}/{split_id}/{window_s}s")
            matrix_a = window[feature_columns(strict)]
            matrix_b = relative_change_matrix(window, strict, window_s)
            retained, audit = screen_train_features(train, strict, window_s, split_type, split_id)
            feature_rows.extend(audit.to_dict(orient="records"))
            matrices = {
                "A_absolute_38": (matrix_a, feature_columns(strict)),
                "B_relative_change": (matrix_b, list(matrix_b.columns)),
                "C_train_only_screened": (matrix_a, feature_columns(retained)),
            }
            for group_name, (matrix, columns) in matrices.items():
                for model_name in MODEL_NAMES:
                    model = model_suite(seed)[model_name]
                    model.fit(matrix.loc[train.index, columns], train["accident_class"])
                    predictions = {
                        "validation": model.predict(matrix.loc[validation.index, columns]),
                        "test": model.predict(matrix.loc[test.index, columns]),
                    }
                    for eval_name, evaluated in (("validation", validation), ("test", test)):
                        y_true = evaluated["accident_class"].to_numpy()
                        y_pred = predictions[eval_name]
                        values = _metric_values(y_true, y_pred)
                        metric_rows.append(
                            {
                                "split_type": split_type,
                                "split_id": split_id,
                                "seed": seed if not seed_values.empty else np.nan,
                                "window_s": int(window_s),
                                "input_group": group_name,
                                "model": model_name,
                                "eval_split": eval_name,
                                "n_train": int(len(train)),
                                "n_validation": int(len(validation)),
                                "n_test": int(len(test)),
                                "n_features": int(len(columns)),
                                "retained_raw_features": int(len(retained)) if group_name == "C_train_only_screened" else int(len(strict)),
                                **values,
                            }
                        )
                        supports = pd.Series(y_true).value_counts().reindex(CLASS_LABELS, fill_value=0)
                        recalls = recall_score(y_true, y_pred, labels=list(CLASS_LABELS), average=None, zero_division=0)
                        for label, recall in zip(CLASS_LABELS, recalls):
                            recall_rows.append(
                                {
                                    "split_type": split_type,
                                    "split_id": split_id,
                                    "window_s": int(window_s),
                                    "input_group": group_name,
                                    "model": model_name,
                                    "eval_split": eval_name,
                                    "accident_class": label,
                                    "recall": float(recall) if supports[label] > 0 else np.nan,
                                    "support": int(supports[label]),
                                }
                            )
                        if eval_name == "test":
                            prediction_store[(split_type, split_id, int(window_s), group_name, model_name, eval_name)] = (
                                evaluated["sample_id"],
                                y_true,
                                y_pred,
                                evaluated["severity"],
                            )

    metrics = pd.DataFrame(metric_rows)
    metrics["selected_for_test"] = False
    selection_keys = ["split_type", "split_id", "window_s", "input_group"]
    for key, candidates in metrics.loc[metrics["eval_split"].eq("validation")].groupby(selection_keys, sort=False):
        best = candidates.sort_values(
            ["macro_f1_fixed_12", "balanced_accuracy", "accuracy", "model"],
            ascending=[False, False, False, True],
        ).iloc[0]
        mask = np.ones(len(metrics), dtype=bool)
        for column, value in zip(selection_keys, key if isinstance(key, tuple) else (key,)):
            mask &= metrics[column].eq(value).to_numpy()
        mask &= metrics["eval_split"].eq("test").to_numpy() & metrics["model"].eq(best["model"]).to_numpy()
        metrics.loc[mask, "selected_for_test"] = True

    selected = metrics.loc[metrics["selected_for_test"] & metrics["eval_split"].eq("test")]
    for row in selected.itertuples(index=False):
        key = (row.split_type, row.split_id, int(row.window_s), row.input_group, row.model, "test")
        sample_ids, y_true, y_pred, severities = prediction_store[key]
        for item in _prediction_rows(key, sample_ids, y_true, y_pred, severities):
            item["selected_for_test"] = True
            item["severity_definition"] = "class-specific NPPAD severity"
            item["model_selection"] = "validation_macro_f1_then_balanced_accuracy"
            confusion_rows.append(item)

    predictions = pd.DataFrame(confusion_rows)
    if predictions.empty:
        raise AssertionError("14 selected prediction table is empty")
    for (split_type, split_id, window_s, group_name), group in predictions.groupby(
        ["split_type", "split_id", "window_s", "input_group"], sort=False
    ):
        for true_label in CLASS_LABELS:
            subset = group.loc[group["true_class"].eq(true_label)]
            support = len(subset)
            for predicted_label in CLASS_LABELS:
                count = int((subset["predicted_class"] == predicted_label).sum())
                confusion_rows.append(
                    {
                        "split_type": split_type,
                        "split_id": split_id,
                        "window_s": int(window_s),
                        "input_group": group_name,
                        "true_class": true_label,
                        "predicted_class": predicted_label,
                        "count": count,
                        "support": support,
                    }
                )

    per_class = (
        predictions.groupby(["split_type", "split_id", "window_s", "input_group", "true_class"], sort=False)
        .agg(support=("true_class", "size"), recall=("true_class", lambda x: float((predictions.loc[x.index, "predicted_class"] == x.iloc[0]).mean())))
        .reset_index()
        .rename(columns={"true_class": "accident_class"})
    )
    confusion = pd.DataFrame([row for row in confusion_rows if "count" in row])
    if confusion.empty or per_class.empty:
        raise AssertionError("14 confusion/per-class outputs are empty")
    return {
        "metrics": metrics,
        "per_class": per_class,
        "predictions": predictions,
        "confusion": confusion,
        "feature_ablation": pd.DataFrame(feature_rows),
        "dataset": dataset,
        "assignments": assignments,
    }


def summarize_feature_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["selected_for_test"] & metrics["eval_split"].eq("test")]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(["split_type", "window_s", "input_group"], sort=False):
        split_type, window_s, input_group = key
        row: dict[str, Any] = {"split_type": split_type, "window_s": int(window_s), "input_group": input_group}
        for metric in ("accuracy", "macro_f1_fixed_12", "macro_f1_observed_classes", "balanced_accuracy"):
            for name, value in _describe(group[metric]).items():
                row[f"{metric}_{name}"] = value
        row["selected_model_counts"] = ";".join(
            f"{name}={count}" for name, count in group["model"].value_counts().sort_index().items()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_feature_outputs(result: dict[str, pd.DataFrame], result_root: str | Path = RESULT_ROOT) -> dict[str, Any]:
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    summary = summarize_feature_metrics(metrics)
    metrics.to_csv(result_root / "14_severity_invariant_metrics.csv", index=False)
    result["per_class"].to_csv(result_root / "14_severity_invariant_per_class.csv", index=False)
    result["feature_ablation"].to_csv(result_root / "14_feature_ablation.csv", index=False)
    result["confusion"].to_csv(result_root / "14_severity_invariant_confusion.csv", index=False)
    summary.to_csv(result_root / "14_severity_invariant_metric_summary.csv", index=False)
    return {"summary": summary, "metrics": metrics}


if __name__ == "__main__":
    context = load_ood_context()
    outputs = evaluate_feature_groups(*context)
    write_feature_outputs(outputs)
    print(json.dumps({"rows": int(len(outputs["metrics"])), "selected": int(outputs["metrics"]["selected_for_test"].sum())}))
