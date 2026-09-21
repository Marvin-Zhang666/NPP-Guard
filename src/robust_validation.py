"""Repeated grouped validation utilities for milestone 12.

The input is one row per trajectory and window. Splits are made once on
unique ``sample_id`` values and then reused for every window and feature
group, so a trajectory cannot cross train/validation/test.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .multi_accident_features import SUMMARY_STATISTICS
except ImportError:
    from multi_accident_features import SUMMARY_STATISTICS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = tuple(20260921 + 1009 * index for index in range(10))
TRAINING_MODES = ("raw", "class_balanced")
MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
METRIC_NAMES = (
    "accuracy",
    "macro_f1_fixed_12",
    "macro_f1_observed_classes",
    "macro_f1_support_ge_3",
    "macro_f1_support_ge_5",
    "balanced_accuracy",
)


def model_suite(random_state: int, training_mode: str) -> dict[str, object]:
    """Return fixed traditional baselines for one split and training mode."""
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {training_mode}")
    balanced = training_mode == "class_balanced"
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced" if balanced else None,
                random_state=random_state,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample" if balanced else None,
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


def feature_columns(features: list[str]) -> list[str]:
    return [f"{feature}__{statistic}" for feature in features for statistic in SUMMARY_STATISTICS]


def _balanced_sample_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    weight = float(len(labels)) / (len(counts) * counts)
    return labels.map(weight).to_numpy(dtype=float)


def make_repeated_splits(
    frame: pd.DataFrame,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
) -> pd.DataFrame:
    """Create repeated 60/20/20 splits on unique trajectory/sample_id rows."""
    entities = frame[["sample_id", "accident_class"]].drop_duplicates("sample_id")
    if len(entities) != frame["sample_id"].nunique():
        raise AssertionError("A sample_id has more than one accident label")
    entities = entities.sort_values("sample_id").reset_index(drop=True)
    if entities["accident_class"].value_counts().min() < 3:
        counts = entities["accident_class"].value_counts().to_dict()
        raise ValueError(f"Every class needs at least 3 trajectories for 3-way splitting: {counts}")
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 - test_fraction:
        raise ValueError("test_fraction and validation_fraction must leave a positive train fraction")

    rows: list[dict[str, object]] = []
    labels = entities["accident_class"].to_numpy()
    validation_of_remainder = validation_fraction / (1.0 - test_fraction)
    for split_index, seed in enumerate(seeds):
        try:
            outer = StratifiedShuffleSplit(
                n_splits=1, test_size=test_fraction, random_state=int(seed)
            )
            train_val_idx, test_idx = next(outer.split(entities[["sample_id"]], labels))
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validation_of_remainder,
                random_state=int(seed) + 7919,
            )
            train_idx, validation_idx = next(
                inner.split(entities.iloc[train_val_idx][["sample_id"]], labels[train_val_idx])
            )
        except ValueError as exc:
            raise ValueError(f"Unable to create grouped split for seed {seed}: {exc}") from exc

        train_ids = set(entities.iloc[train_val_idx[train_idx]]["sample_id"])
        validation_ids = set(entities.iloc[train_val_idx[validation_idx]]["sample_id"])
        test_ids = set(entities.iloc[test_idx]["sample_id"])
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise AssertionError("Trajectory/sample_id crossed train, validation, and test")
        if train_ids | validation_ids | test_ids != set(entities["sample_id"]):
            raise AssertionError("Repeated split does not cover every trajectory")
        for sample_id in sorted(train_ids):
            rows.append({"sample_id": sample_id, "split_index": split_index, "seed": int(seed), "partition": "train"})
        for sample_id in sorted(validation_ids):
            rows.append({"sample_id": sample_id, "split_index": split_index, "seed": int(seed), "partition": "validation"})
        for sample_id in sorted(test_ids):
            rows.append({"sample_id": sample_id, "split_index": split_index, "seed": int(seed), "partition": "test"})

    assignments = pd.DataFrame(rows)
    if assignments[["sample_id", "split_index"]].duplicated().any():
        raise AssertionError("A trajectory received two partitions in one repeated split")
    return assignments


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    support = pd.Series(y_true).value_counts().reindex(class_labels, fill_value=0).to_numpy(dtype=int)
    observed = [label for label, count in zip(class_labels, support) if count > 0]
    support_ge_3 = [label for label, count in zip(class_labels, support) if count >= 3]
    support_ge_5 = [label for label, count in zip(class_labels, support) if count >= 5]

    def macro(labels: list[str]) -> float:
        if not labels:
            return float("nan")
        return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_fixed_12": macro(class_labels),
        "macro_f1_observed_classes": macro(observed),
        "macro_f1_support_ge_3": macro(support_ge_3),
        "macro_f1_support_ge_5": macro(support_ge_5),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "minimum_test_support": int(support[support > 0].min()) if np.any(support > 0) else 0,
        "n_classes_eval": int(len(observed)),
        "n_support_lt_3": int(np.sum(support < 3)),
        "n_support_lt_5": int(np.sum(support < 5)),
    }, support


def run_repeated_validation(
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    class_labels: list[str],
    windows_s: tuple[int, ...] = (30, 60, 90, 120),
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    scope: str = "strict_pre_protection_only",
    cohort: str = "strict_matched_30_60_90_120",
) -> dict[str, pd.DataFrame]:
    """Run fixed traditional models over repeated, shared trajectory splits."""
    if dataset[["sample_id", "window_s"]].duplicated().any():
        raise AssertionError("Robust validation requires one row per trajectory and window")
    if set(dataset["window_s"].unique()) != set(windows_s):
        raise AssertionError("Robust validation dataset is missing a requested window")
    if scope == "strict_pre_protection_only":
        if dataset["first_protection_s"].isna().any() or (~dataset["strict_pre_protection"]).any():
            raise AssertionError("Unknown or post-protection trajectories entered strict validation")

    assignments = make_repeated_splits(dataset, seeds=seeds)

    metric_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []

    for split_index, seed in enumerate(seeds):
        split_assignments = assignments.loc[assignments["split_index"].eq(split_index)]
        split_frame = dataset.merge(split_assignments, on="sample_id", how="left", validate="many_to_one")
        if split_frame["partition"].isna().any():
            raise AssertionError(f"A dataset trajectory has no assignment for split {split_index}")
        for window_s in windows_s:
            window = split_frame.loc[split_frame["window_s"].eq(window_s)].copy()
            train = window.loc[window["partition"].eq("train")]
            validation = window.loc[window["partition"].eq("validation")]
            test = window.loc[window["partition"].eq("test")]
            if train.empty or validation.empty or test.empty:
                raise AssertionError(f"Empty train/validation/test partition at {window_s}s, seed={seed}")
            partition_sets = {name: set(frame["sample_id"]) for name, frame in (("train", train), ("validation", validation), ("test", test))}
            if any(partition_sets[left] & partition_sets[right] for left, right in combinations(partition_sets, 2)):
                raise AssertionError(f"Trajectory leakage at {window_s}s, seed={seed}")
            cohort_counts = window["accident_class"].value_counts()
            partition_counts = {
                name: frame["accident_class"].value_counts().reindex(class_labels, fill_value=0).to_numpy(dtype=int)
                for name, frame in (("train", train), ("validation", validation), ("test", test))
            }
            for label, test_support in zip(class_labels, partition_counts["test"]):
                support_rows.append(
                    {
                        "cohort": cohort,
                        "scope": scope,
                        "split_index": split_index,
                        "seed": int(seed),
                        "window_s": int(window_s),
                        "accident_class": label,
                        "cohort_support": int(cohort_counts.get(label, 0)),
                        "train_support": int(partition_counts["train"][class_labels.index(label)]),
                        "validation_support": int(partition_counts["validation"][class_labels.index(label)]),
                        "test_support": int(test_support),
                        "test_support_lt_3": bool(test_support < 3),
                        "test_support_lt_5": bool(test_support < 5),
                        "support_status": (
                            "not_evaluable_no_strict_trajectory"
                            if cohort_counts.get(label, 0) == 0
                            else "low_support"
                            if test_support < 5
                            else "adequate_for_audit"
                        ),
                    }
                )

            for input_group, features in feature_groups.items():
                columns = feature_columns(features)
                for training_mode in TRAINING_MODES:
                    for model_name, model in model_suite(int(seed), training_mode).items():
                        if train["accident_class"].nunique() < 2:
                            majority = train["accident_class"].value_counts().sort_index().index[0]
                            predictions = {
                                "validation": np.repeat(majority, len(validation)),
                                "test": np.repeat(majority, len(test)),
                            }
                            fit_status = "majority_fallback"
                        else:
                            if model_name == "hist_gradient_boosting" and training_mode == "class_balanced":
                                model.fit(
                                    train[columns],
                                    train["accident_class"],
                                    sample_weight=_balanced_sample_weights(train["accident_class"]),
                                )
                            else:
                                model.fit(train[columns], train["accident_class"])
                            predictions = {
                                "validation": model.predict(validation[columns]),
                                "test": model.predict(test[columns]),
                            }
                            fit_status = "fitted"

                        for eval_name, evaluated in (("validation", validation), ("test", test)):
                            y_true = evaluated["accident_class"].to_numpy()
                            y_pred = predictions[eval_name]
                            values, _ = _metric_values(y_true, y_pred, class_labels)
                            metric_rows.append(
                                {
                                    "cohort": cohort,
                                    "scope": scope,
                                    "split_index": split_index,
                                    "seed": int(seed),
                                    "window_s": int(window_s),
                                    "input_group": input_group,
                                    "training_mode": training_mode,
                                    "model": model_name,
                                    "eval_split": eval_name,
                                    "n_train": int(len(train)),
                                    "n_samples": int(len(evaluated)),
                                    "n_features": int(len(columns)),
                                    "n_classes_train": int(train["accident_class"].nunique()),
                                    "fit_status": fit_status,
                                    **values,
                                }
                            )

                            if eval_name != "test":
                                continue
                            matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
                            recalls = recall_score(
                                y_true, y_pred, labels=class_labels, average=None, zero_division=0
                            )
                            supports = pd.Series(y_true).value_counts().reindex(class_labels, fill_value=0).to_numpy(dtype=int)
                            for label, recall, support, row in zip(class_labels, recalls, supports, matrix):
                                recall_rows.append(
                                    {
                                        "cohort": cohort,
                                        "scope": scope,
                                        "split_index": split_index,
                                        "seed": int(seed),
                                        "window_s": int(window_s),
                                        "input_group": input_group,
                                        "training_mode": training_mode,
                                        "model": model_name,
                                        "accident_class": label,
                                        "recall": float(recall) if support > 0 else np.nan,
                                        "support": int(support),
                                        "correct": int(row[class_labels.index(label)]),
                                        "cohort_support": int(cohort_counts.get(label, 0)),
                                    }
                                )
                            for true_label, row in zip(class_labels, matrix):
                                for predicted_label, count in zip(class_labels, row):
                                    confusion_rows.append(
                                        {
                                            "cohort": cohort,
                                            "scope": scope,
                                            "split_index": split_index,
                                            "seed": int(seed),
                                            "window_s": int(window_s),
                                            "input_group": input_group,
                                            "training_mode": training_mode,
                                            "model": model_name,
                                            "true_class": true_label,
                                            "predicted_class": predicted_label,
                                            "count": int(count),
                                        }
                                    )

    metrics = pd.DataFrame(metric_rows)
    metrics["validation_selected"] = False
    selection_keys = ["cohort", "scope", "split_index", "seed", "window_s", "input_group"]
    validation = metrics.loc[metrics["eval_split"].eq("validation")].copy()
    for key, candidates in validation.groupby(selection_keys, dropna=False, sort=False):
        best = candidates.sort_values(
            ["macro_f1_fixed_12", "macro_f1_support_ge_3", "balanced_accuracy", "accuracy", "training_mode", "model"],
            ascending=[False, False, False, False, True, True],
        ).iloc[0]
        mask = np.ones(len(metrics), dtype=bool)
        for column, value in zip(selection_keys, key if isinstance(key, tuple) else (key,)):
            mask &= metrics[column].eq(value).to_numpy()
        mask &= metrics["training_mode"].eq(best["training_mode"]).to_numpy()
        mask &= metrics["model"].eq(best["model"]).to_numpy()
        metrics.loc[mask, "validation_selected"] = True

    selection_columns = selection_keys + ["training_mode", "model", "validation_selected"]
    selected = metrics.loc[metrics["eval_split"].eq("test"), selection_columns]
    recalls = pd.DataFrame(recall_rows).merge(selected, on=selection_columns[:-1], how="left", validate="many_to_one")
    confusions = pd.DataFrame(confusion_rows).merge(selected, on=selection_columns[:-1], how="left", validate="many_to_one")
    assignments["cohort"] = cohort
    assignments["scope"] = scope
    return {
        "metrics": metrics,
        "per_class": recalls,
        "confusion": confusions,
        "support_audit": pd.DataFrame(support_rows),
        "assignments": assignments,
    }


def _describe(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(numeric))
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if n > 1 else 0.0
    margin = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def aggregate_metric_summary(
    metrics: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize split metrics with normal-approximation 95% CIs."""
    if group_cols is None:
        group_cols = [
            "cohort",
            "scope",
            "window_s",
            "input_group",
            "training_mode",
            "model",
            "eval_split",
            "validation_selected",
        ]
    rows: list[dict[str, object]] = []
    for key, group in metrics.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        for metric in METRIC_NAMES:
            description = _describe(group[metric])
            for statistic, value in description.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_class_stability(
    recalls: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize recall only over splits with positive test support."""
    if group_cols is None:
        group_cols = [
            "cohort",
            "scope",
            "window_s",
            "input_group",
            "training_mode",
            "model",
            "validation_selected",
            "accident_class",
        ]
    rows: list[dict[str, object]] = []
    for key, group in recalls.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        cohort_support = int(group["cohort_support"].max())
        supported = group.loc[group["support"].gt(0), "recall"]
        description = _describe(supported)
        row.update(
            {
                "cohort_support": cohort_support,
                "n_splits": int(len(group)),
                "n_evaluable_splits": int(group["support"].gt(0).sum()),
                "support_mean": float(group["support"].mean()),
                "support_min": int(group["support"].min()),
                "support_max": int(group["support"].max()),
                "n_splits_support_ge_3": int(group["support"].ge(3).sum()),
                "n_splits_support_ge_5": int(group["support"].ge(5).sum()),
                "status": (
                    "not_evaluable_no_strict_trajectory"
                    if cohort_support == 0
                    else "low_support"
                    if group["support"].lt(5).any()
                    else "evaluable"
                ),
                "reliable_for_gate": bool(cohort_support > 0 and group["support"].ge(5).all()),
            }
        )
        for statistic, value in description.items():
            row[f"recall_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_confusion_stability(
    confusions: pd.DataFrame,
    class_labels: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate unordered off-diagonal confusion-pair frequency."""
    config_cols = [
        "cohort",
        "scope",
        "split_index",
        "seed",
        "window_s",
        "input_group",
        "training_mode",
        "model",
        "validation_selected",
    ]
    pair_rows: list[dict[str, object]] = []
    for key, group in confusions.groupby(config_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        config = dict(zip(config_cols, key_values))
        counts = {
            (row.true_class, row.predicted_class): int(row.count)
            for row in group.itertuples(index=False)
        }
        for left, right in combinations(class_labels, 2):
            forward = counts.get((left, right), 0)
            reverse = counts.get((right, left), 0)
            pair_rows.append(
                {
                    **config,
                    "pair": f"{left}\u2194{right}",
                    "left_class": left,
                    "right_class": right,
                    "forward_count": forward,
                    "reverse_count": reverse,
                    "pair_count": forward + reverse,
                    "pair_present": bool(forward + reverse > 0),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    if group_cols is None:
        group_cols = config_cols + ["pair", "left_class", "right_class"]
    else:
        group_cols = list(group_cols) + [column for column in ("pair", "left_class", "right_class") if column not in group_cols]
    rows: list[dict[str, object]] = []
    for key, group in pairs.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        description = _describe(group["pair_count"])
        row.update(
            {
                "n_splits": int(group[["seed", "split_index"]].drop_duplicates().shape[0]),
                "split_count_present": int(group["pair_present"].sum()),
                "pair_frequency": float(group["pair_present"].mean()),
                "stable_frequency_ge_50pct": bool(group["pair_present"].mean() >= 0.5),
            }
        )
        for statistic, value in description.items():
            row[f"pair_count_{statistic}"] = value
        row["forward_count_mean"] = float(group["forward_count"].mean())
        row["reverse_count_mean"] = float(group["reverse_count"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
