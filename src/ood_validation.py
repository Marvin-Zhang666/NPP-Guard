"""Severity-blocked and extrapolation validation for milestone 13."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from .event_parser import scan_event_inventory, write_event_outputs
    from .multi_accident_features import input_feature_groups, strict_process_features
    from .multi_accident_models import model_suite
    from .severity_features import INJECTION_S, SUMMARY_STATISTICS, extract_window_features, window_sample_points
except ImportError:
    from event_parser import scan_event_inventory, write_event_outputs
    from multi_accident_features import input_feature_groups, strict_process_features
    from multi_accident_models import model_suite
    from severity_features import INJECTION_S, SUMMARY_STATISTICS, extract_window_features, window_sample_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
DEFAULT_INVENTORY = RESULT_ROOT / "09_multi_accident_class_inventory.csv"
DEFAULT_EVENT_INVENTORY = RESULT_ROOT / "09_multi_accident_event_inventory.csv"
WINDOWS_S = (30, 60, 90, 120)
OOD_WINDOWS_S = (60, 90, 120)
CLASS_LABELS = (
    "FLB", "LLB", "LOCA", "LOCAC", "LR", "MD", "RI", "RW",
    "SGATR", "SGBTR", "SLBIC", "SLBOC",
)
SEEDS = tuple(20260921 + 1009 * index for index in range(10))
MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")

SEVERITY_DEFINITIONS = {
    "LOCA": "% of 100 cm2 hot-leg break",
    "LOCAC": "% of 100 cm2 cold-leg break",
    "SLBIC": "% of 100 cm2 inside-containment steam-line break",
    "SLBOC": "% of 100 cm2 outside-containment steam-line break",
    "SGATR": "% of one full steam-generator-A tube rupture",
    "SGBTR": "% of one full steam-generator-B tube rupture",
    "RW": "% of rods withdrawn",
    "RI": "% of rods inserted",
    "FLB": "% of 100 cm2 feedwater-line break",
    "MD": "% of unborated injection",
    "LR": "% of full load rejected",
    "LLB": "% of nominal letdown flow",
}


def _feature_columns(features: list[str]) -> list[str]:
    return [f"{feature}__{statistic}" for feature in features for statistic in SUMMARY_STATISTICS]


def build_ood_feature_dataset(
    project_root: str | Path = PROJECT_ROOT,
    inventory_path: str | Path = DEFAULT_INVENTORY,
    event_inventory_path: str | Path = DEFAULT_EVENT_INVENTORY,
    windows_s: tuple[int, ...] = WINDOWS_S,
    classes: tuple[str, ...] = CLASS_LABELS,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per complete trajectory/window, including severity metadata."""
    project_root = Path(project_root)
    features = features or strict_process_features()
    inventory = pd.read_csv(inventory_path)
    events = pd.read_csv(event_inventory_path)[["sample_id", "first_protection_s"]]
    selected = inventory.loc[inventory["accident_class"].isin(classes)].merge(
        events, on="sample_id", how="left", validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for record in selected.sort_values("sample_id").itertuples(index=False):
        path = project_root / Path(record.operation_csv)
        frame = pd.read_csv(path)
        if "TIME" not in frame or any(feature not in frame for feature in features):
            raise ValueError(f"Strict schema missing from {path}")
        time = pd.to_numeric(frame["TIME"], errors="coerce")
        first_protection = float(record.first_protection_s) if pd.notna(record.first_protection_s) else np.nan
        for window_s in windows_s:
            if float(time.max()) < window_s:
                continue
            points = window_sample_points(frame, window_s, INJECTION_S)
            extracted = extract_window_features(frame, features, window_s, INJECTION_S)
            last_sample_s = float(points[-1])
            known = bool(np.isfinite(first_protection))
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "accident_class": record.accident_class,
                    "numeric_case_id": int(record.numeric_case_id),
                    "severity": float(record.numeric_case_id),
                    "severity_definition": SEVERITY_DEFINITIONS[record.accident_class],
                    "severity_source": "NPPAD_README_case_number",
                    "window_s": int(window_s),
                    "first_protection_s": first_protection if known else np.nan,
                    "protection_time_known": known,
                    "last_sample_s": last_sample_s,
                    "strict_pre_protection": bool(known and first_protection > last_sample_s),
                    **extracted,
                }
            )
            point_rows.append(
                {
                    "sample_id": record.sample_id,
                    "accident_class": record.accident_class,
                    "severity": float(record.numeric_case_id),
                    "window_s": int(window_s),
                    "first_protection_s": first_protection if known else np.nan,
                    "last_sample_s": last_sample_s,
                    "strict_pre_protection": bool(known and first_protection > last_sample_s),
                    "sample_count": len(points),
                    "sample_points_s": ",".join(f"{point:g}" for point in points),
                }
            )
    dataset = pd.DataFrame(rows).sort_values(["window_s", "sample_id"]).reset_index(drop=True)
    points = pd.DataFrame(point_rows).sort_values(["window_s", "sample_id"]).reset_index(drop=True)
    if dataset.empty or dataset[["sample_id", "window_s"]].duplicated().any():
        raise AssertionError("OOD feature dataset is empty or has duplicate trajectory-window rows")
    if dataset[_feature_columns(features)].isna().any().any():
        raise AssertionError("OOD feature matrix contains missing values")
    return dataset, points


def _entities(dataset: pd.DataFrame) -> pd.DataFrame:
    entities = dataset[
        ["sample_id", "accident_class", "severity", "severity_definition", "severity_source"]
    ].drop_duplicates("sample_id")
    if len(entities) != dataset["sample_id"].nunique():
        raise AssertionError("A trajectory has inconsistent class or severity metadata")
    return entities.sort_values(["accident_class", "severity", "sample_id"]).reset_index(drop=True)


def _assignment_row(entity: pd.Series, split_type: str, split_id: str, seed: int | None, partition: str) -> dict[str, Any]:
    return {
        "split_type": split_type,
        "split_id": split_id,
        "seed": seed,
        "sample_id": entity.sample_id,
        "accident_class": entity.accident_class,
        "severity": float(entity.severity),
        "severity_definition": entity.severity_definition,
        "partition": partition,
    }


def random_assignments(entities: pd.DataFrame, seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """Repeated 60/20/20 trajectory-level stratified splits."""
    labels = entities["accident_class"].to_numpy()
    rows: list[dict[str, Any]] = []
    validation_of_remainder = 0.20 / 0.80
    for split_index, seed in enumerate(seeds):
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=int(seed))
        train_val_idx, test_idx = next(outer.split(entities[["sample_id"]], labels))
        inner = StratifiedShuffleSplit(n_splits=1, test_size=validation_of_remainder, random_state=int(seed) + 7919)
        train_idx, validation_idx = next(
            inner.split(entities.iloc[train_val_idx][["sample_id"]], labels[train_val_idx])
        )
        partition_indices = {
            "train": train_val_idx[train_idx],
            "validation": train_val_idx[validation_idx],
            "test": test_idx,
        }
        assigned: set[str] = set()
        for partition, indices in partition_indices.items():
            for index in indices:
                entity = entities.iloc[int(index)]
                if entity.sample_id in assigned:
                    raise AssertionError("Random split crossed trajectory/sample_id partitions")
                assigned.add(entity.sample_id)
                rows.append(_assignment_row(entity, "random", f"seed_{seed}", int(seed), partition))
        if len(assigned) != len(entities):
            raise AssertionError("Random split did not cover every trajectory")
    return pd.DataFrame(rows)


def _block_positions(n: int, block: str, test_count: int) -> tuple[list[int], list[int]]:
    if block == "low":
        test = list(range(test_count))
        guard = [test_count] if test_count < n else []
    elif block == "high":
        test = list(range(n - test_count, n))
        guard = [n - test_count - 1] if test_count < n else []
    elif block == "middle":
        start = max(0, (n - test_count) // 2)
        test = list(range(start, start + test_count))
        guard = [index for index in (start - 1, start + test_count) if 0 <= index < n]
    else:
        raise ValueError(f"Unknown severity block: {block}")
    return test, guard


def _class_block_assignments(
    entities: pd.DataFrame,
    block: str,
    split_type: str,
    split_id: str,
    min_class_size: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    for accident_class, subset in entities.groupby("accident_class", sort=True):
        ordered = subset.sort_values(["severity", "sample_id"]).reset_index(drop=True)
        n = len(ordered)
        if n < min_class_size:
            excluded[accident_class] = f"class_size_{n}_below_{min_class_size}"
            continue
        test_count = max(1, int(np.ceil(0.20 * n)))
        test_positions, guard_positions = _block_positions(n, block, test_count)
        available = [index for index in range(n) if index not in set(test_positions + guard_positions)]
        validation_count = max(1, int(round(0.20 * len(available))))
        if len(available) - validation_count < 1:
            excluded[accident_class] = "insufficient_non_test_rows_for_train_and_validation"
            continue
        validation_positions = available[-validation_count:]
        train_positions = [index for index in available if index not in set(validation_positions)]
        for index in train_positions:
            rows.append(_assignment_row(ordered.iloc[index], split_type, split_id, None, "train"))
        for index in validation_positions:
            rows.append(_assignment_row(ordered.iloc[index], split_type, split_id, None, "validation"))
        for index in test_positions:
            rows.append(_assignment_row(ordered.iloc[index], split_type, split_id, None, "test"))
        for index in guard_positions:
            rows.append(_assignment_row(ordered.iloc[index], split_type, split_id, None, "gap_excluded"))
    if not rows:
        raise ValueError(f"No severity-blocked classes are evaluable for {split_id}")
    return rows, excluded


def severity_blocked_assignments(entities: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    excluded: dict[str, dict[str, str]] = {}
    for block in ("low", "middle", "high"):
        block_rows, block_excluded = _class_block_assignments(
            entities, block, "severity_blocked", f"{block}_block"
        )
        rows.extend(block_rows)
        excluded[f"{block}_block"] = block_excluded
    return pd.DataFrame(rows), excluded


def _extrapolation_assignments(
    entities: pd.DataFrame,
    direction: str,
    min_class_size: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    for accident_class, subset in entities.groupby("accident_class", sort=True):
        ordered = subset.sort_values(["severity", "sample_id"]).reset_index(drop=True)
        n = len(ordered)
        if n < min_class_size:
            excluded[accident_class] = f"class_size_{n}_below_{min_class_size}"
            continue
        test_count = max(1, int(np.ceil(0.20 * n)))
        guard_count = 1
        if direction == "low_train_to_high_test":
            test_positions = list(range(n - test_count, n))
            guard_positions = [n - test_count - 1]
            lower = list(range(0, n - test_count - guard_count))
        elif direction == "high_train_to_low_test":
            test_positions = list(range(test_count))
            guard_positions = [test_count]
            lower = list(range(test_count + guard_count, n))
        else:
            raise ValueError(f"Unknown extrapolation direction: {direction}")
        validation_count = max(1, int(round(0.20 * len(lower))))
        if len(lower) - validation_count < 1:
            excluded[accident_class] = "insufficient_non_test_rows_for_train_and_validation"
            continue
        if direction == "low_train_to_high_test":
            validation_positions = lower[-validation_count:]
            train_positions = lower[:-validation_count]
        else:
            validation_positions = lower[:validation_count]
            train_positions = lower[validation_count:]
        for index in train_positions:
            rows.append(_assignment_row(ordered.iloc[index], "severity_extrapolation", direction, None, "train"))
        for index in validation_positions:
            rows.append(_assignment_row(ordered.iloc[index], "severity_extrapolation", direction, None, "validation"))
        for index in test_positions:
            rows.append(_assignment_row(ordered.iloc[index], "severity_extrapolation", direction, None, "test"))
        for index in guard_positions:
            rows.append(_assignment_row(ordered.iloc[index], "severity_extrapolation", direction, None, "gap_excluded"))
    return rows, excluded


def severity_extrapolation_assignments(entities: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    excluded: dict[str, dict[str, str]] = {}
    for direction in ("low_train_to_high_test", "high_train_to_low_test"):
        direction_rows, direction_excluded = _extrapolation_assignments(entities, direction)
        rows.extend(direction_rows)
        excluded[direction] = direction_excluded
    return pd.DataFrame(rows), excluded


def nearest_severity_gaps(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split_type, split_id), split in assignments.groupby(["split_type", "split_id"], sort=False):
        train = split.loc[split["partition"].eq("train")]
        for test in split.loc[split["partition"].eq("test")].itertuples(index=False):
            same_class = train.loc[train["accident_class"].eq(test.accident_class)]
            if same_class.empty:
                gap = np.nan
                nearest = np.nan
            else:
                differences = (same_class["severity"] - float(test.severity)).abs()
                nearest_index = differences.idxmin()
                gap = float(differences.loc[nearest_index])
                nearest = float(same_class.loc[nearest_index, "severity"])
            rows.append(
                {
                    "split_type": split_type,
                    "split_id": split_id,
                    "sample_id": test.sample_id,
                    "accident_class": test.accident_class,
                    "test_severity": float(test.severity),
                    "nearest_train_severity": nearest,
                    "nearest_severity_gap": gap,
                }
            )
    return pd.DataFrame(rows)


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    supports = pd.Series(y_true).value_counts().reindex(CLASS_LABELS, fill_value=0).to_numpy(dtype=int)
    observed = [label for label, support in zip(CLASS_LABELS, supports) if support > 0]
    observed_recall = recall_score(y_true, y_pred, labels=observed, average="macro", zero_division=0) if observed else np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_fixed_12": float(f1_score(y_true, y_pred, labels=list(CLASS_LABELS), average="macro", zero_division=0)),
        "macro_f1_observed_classes": float(
            f1_score(y_true, y_pred, labels=observed, average="macro", zero_division=0)
        ) if observed else np.nan,
        "balanced_accuracy": float(observed_recall),
        "n_classes_eval": int(len(observed)),
        "minimum_test_support": int(supports[supports > 0].min()) if np.any(supports > 0) else 0,
    }, supports


def evaluate_splits(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    windows_s: tuple[int, ...] = WINDOWS_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on train only, select on validation, and score the held-out test."""
    metric_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    for (split_type, split_id), split in assignments.groupby(["split_type", "split_id"], sort=False):
        active = split.loc[split["partition"].isin(["train", "validation", "test"])]
        split_frame = dataset.merge(
            active[["sample_id", "partition"]], on="sample_id", how="inner", validate="many_to_one"
        )
        seed = int(split["seed"].dropna().iloc[0]) if split["seed"].notna().any() else 20260921
        for window_s in windows_s:
            window = split_frame.loc[split_frame["window_s"].eq(window_s)]
            train = window.loc[window["partition"].eq("train")]
            validation = window.loc[window["partition"].eq("validation")]
            test = window.loc[window["partition"].eq("test")]
            if train.empty or validation.empty or test.empty:
                raise AssertionError(f"Empty train/validation/test partition for {split_type}/{split_id}/{window_s}s")
            for input_group, features in feature_groups.items():
                columns = _feature_columns(features)
                for model_name in MODEL_NAMES:
                    model = model_suite(seed)[model_name]
                    if train["accident_class"].nunique() < 2:
                        majority = train["accident_class"].value_counts().sort_index().index[0]
                        predictions = {"validation": np.repeat(majority, len(validation)), "test": np.repeat(majority, len(test))}
                        fit_status = "majority_fallback"
                    else:
                        model.fit(train[columns], train["accident_class"])
                        predictions = {"validation": model.predict(validation[columns]), "test": model.predict(test[columns])}
                        fit_status = "fitted"
                    for eval_name, evaluated in (("validation", validation), ("test", test)):
                        y_true = evaluated["accident_class"].to_numpy()
                        values, supports = _metric_values(y_true, predictions[eval_name])
                        metric_rows.append(
                            {
                                "split_type": split_type,
                                "split_id": split_id,
                                "seed": seed if split["seed"].notna().any() else np.nan,
                                "window_s": int(window_s),
                                "input_group": input_group,
                                "model": model_name,
                                "eval_split": eval_name,
                                "n_train": int(len(train)),
                                "n_validation": int(len(validation)),
                                "n_test": int(len(test)),
                                "n_features": int(len(columns)),
                                "fit_status": fit_status,
                                **values,
                            }
                        )
                        recalls = recall_score(y_true, predictions[eval_name], labels=list(CLASS_LABELS), average=None, zero_division=0)
                        for label, recall, support in zip(CLASS_LABELS, recalls, supports):
                            recall_rows.append(
                                {
                                    "split_type": split_type,
                                    "split_id": split_id,
                                    "seed": seed if split["seed"].notna().any() else np.nan,
                                    "window_s": int(window_s),
                                    "input_group": input_group,
                                    "model": model_name,
                                    "eval_split": eval_name,
                                    "accident_class": label,
                                    "recall": float(recall) if support > 0 else np.nan,
                                    "support": int(support),
                                }
                            )
    metrics = pd.DataFrame(metric_rows)
    metrics["selected_for_test"] = False
    key_columns = ["split_type", "split_id", "window_s", "input_group"]
    for key, candidates in metrics.loc[metrics["eval_split"].eq("validation")].groupby(key_columns, sort=False):
        best = candidates.sort_values(
            ["macro_f1_fixed_12", "macro_f1_observed_classes", "balanced_accuracy", "accuracy", "model"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        mask = np.ones(len(metrics), dtype=bool)
        for column, value in zip(key_columns, key if isinstance(key, tuple) else (key,)):
            mask &= metrics[column].eq(value).to_numpy()
        mask &= metrics["eval_split"].eq("test").to_numpy()
        mask &= metrics["model"].eq(best["model"]).to_numpy()
        metrics.loc[mask, "selected_for_test"] = True
    selected = metrics.loc[metrics["selected_for_test"], key_columns + ["model"]].drop_duplicates()
    selected["_selected_model"] = True
    recalls = pd.DataFrame(recall_rows).merge(
        selected, on=key_columns + ["model"], how="left", validate="many_to_one"
    )
    recalls["selected_for_test"] = recalls["eval_split"].eq("test") & recalls["_selected_model"].fillna(False)
    return metrics, recalls


def _describe(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(numeric)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if n > 1 else 0.0
    margin = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {"n": int(n), "mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin}


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["selected_for_test"] & metrics["eval_split"].eq("test")]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(["split_type", "window_s", "input_group"], sort=False):
        split_type, window_s, input_group = key
        row: dict[str, Any] = {"split_type": split_type, "window_s": int(window_s), "input_group": input_group}
        for metric in ("accuracy", "macro_f1_fixed_12", "macro_f1_observed_classes", "balanced_accuracy"):
            for name, value in _describe(group[metric]).items():
                row[f"{metric}_{name}"] = value
        row["selected_model_counts"] = ";".join(f"{name}={count}" for name, count in group["model"].value_counts().sort_index().items())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_class(recalls: pd.DataFrame) -> pd.DataFrame:
    selected = recalls.loc[recalls["selected_for_test"] & recalls["eval_split"].eq("test")]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(["split_type", "window_s", "input_group", "accident_class"], sort=False):
        split_type, window_s, input_group, accident_class = key
        description = _describe(group["recall"])
        rows.append(
            {
                "split_type": split_type,
                "window_s": int(window_s),
                "input_group": input_group,
                "accident_class": accident_class,
                "support_mean": float(group["support"].mean()),
                "support_min": int(group["support"].min()),
                "support_max": int(group["support"].max()),
                "n_splits": int(len(group)),
                "n_evaluable_splits": int(group["support"].gt(0).sum()),
                **{f"recall_{name}": value for name, value in description.items()},
            }
        )
    return pd.DataFrame(rows)


def _write_figures(
    metric_summary: pd.DataFrame,
    per_class_summary: pd.DataFrame,
    gaps: pd.DataFrame,
    event_frame: pd.DataFrame,
    result_root: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = result_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    primary = metric_summary.loc[metric_summary["input_group"].eq("A_strict_38") & metric_summary["window_s"].eq(120)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, row in enumerate(primary.itertuples(index=False)):
        ax.errorbar(index, row.macro_f1_fixed_12_mean, yerr=row.macro_f1_fixed_12_std, fmt="o", capsize=4)
    ax.set_xticks(range(len(primary)), primary["split_type"].tolist(), rotation=20)
    ax.set_ylabel("Fixed-12 Macro-F1")
    ax.set_title("120 s baseline vs severity-blocked vs extrapolation")
    fig.tight_layout()
    path = figure_root / "13_baseline_blocked_extrapolation_macro_f1.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.relative_to(result_root.parent).as_posix())

    fig, ax = plt.subplots(figsize=(8, 5))
    comparison = metric_summary.loc[
        metric_summary["input_group"].eq("A_strict_38")
        & metric_summary["split_type"].isin(["severity_blocked", "severity_extrapolation"])
        & metric_summary["window_s"].isin(OOD_WINDOWS_S)
    ]
    for split_type, group in comparison.groupby("split_type", sort=False):
        group = group.sort_values("window_s")
        ax.errorbar(group["window_s"], group["macro_f1_fixed_12_mean"], yerr=group["macro_f1_fixed_12_std"], marker="o", capsize=4, label=split_type)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Fixed-12 Macro-F1")
    ax.set_title("Matched OOD test cohort: 60/90/120 s")
    ax.legend()
    fig.tight_layout()
    path = figure_root / "13_matched_ood_window_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.relative_to(result_root.parent).as_posix())

    recall = per_class_summary.loc[
        per_class_summary["input_group"].eq("A_strict_38")
        & per_class_summary["window_s"].eq(120)
        & per_class_summary["split_type"].isin(["severity_blocked", "severity_extrapolation"])
    ]
    pivot = recall.pivot(index="accident_class", columns="split_type", values="recall_mean")
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.bar(ax=ax, color=["#3579a8", "#d9813d"])
    ax.set_ylabel("Recall")
    ax.set_title("120 s OOD per-class recall")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Split")
    fig.tight_layout()
    path = figure_root / "13_ood_per_class_recall.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.relative_to(result_root.parent).as_posix())

    fig, ax = plt.subplots(figsize=(8, 5))
    gap_groups = [g["nearest_severity_gap"].dropna().to_numpy() for _, g in gaps.groupby("split_type", sort=False)]
    labels = [name for name, _ in gaps.groupby("split_type", sort=False)]
    ax.boxplot(gap_groups, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Nearest same-class train severity gap")
    ax.set_title("Nearest-severity gap audit")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = figure_root / "13_nearest_severity_gap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.relative_to(result_root.parent).as_posix())

    coverage = event_frame.groupby("accident_class", sort=True).agg(
        trajectories=("sample_id", "count"), protection_parsed=("event_parse_ok", "sum")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    coverage.plot.bar(ax=ax, color=["#9dbbd1", "#4e8f6c"])
    ax.set_ylabel("Trajectories")
    ax.set_title("Generic protection-event parser coverage")
    ax.legend(["All reports", "Protection parsed"])
    fig.tight_layout()
    path = figure_root / "13_event_parser_recovery_coverage.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.relative_to(result_root.parent).as_posix())
    return paths


def _gate_summary(metric_summary: pd.DataFrame, per_class_summary: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for split_type in ("severity_blocked", "severity_extrapolation"):
        values = metric_summary.loc[
            metric_summary["split_type"].eq(split_type) & metric_summary["input_group"].eq("A_strict_38")
        ].set_index("window_s")
        if not {60, 90, 120}.issubset(values.index):
            checks[split_type] = {"available": False}
            continue
        m120 = values.loc[120]
        controls = values.loc[[60, 90]]
        checks[split_type] = {
            "available": True,
            "macro_f1_120": float(m120["macro_f1_fixed_12_mean"]),
            "balanced_accuracy_120": float(m120["balanced_accuracy_mean"]),
            "macro_f1_120_exceeds_60_90_by_mean": bool(
                m120["macro_f1_fixed_12_mean"] > controls["macro_f1_fixed_12_mean"].max()
            ),
            "macro_f1_120_exceeds_60_90_by_ci": bool(
                m120["macro_f1_fixed_12_ci95_low"] > controls["macro_f1_fixed_12_ci95_high"].max()
            ),
            "balanced_accuracy_120_exceeds_60_90_by_mean": bool(
                m120["balanced_accuracy_mean"] > controls["balanced_accuracy_mean"].max()
            ),
        }
    split_checks = [item for item in checks.values() if isinstance(item, dict) and item.get("available")]
    checks["ood_120_macro_f1_mean_at_least_0_50"] = all(
        item.get("macro_f1_120", 0.0) >= 0.50 for item in split_checks
    )
    checks["ood_120_balanced_accuracy_mean_at_least_0_50"] = all(
        item.get("balanced_accuracy_120", 0.0) >= 0.50 for item in split_checks
    )
    checks["ood_120_ci_advantage_over_controls"] = all(
        item.get("macro_f1_120_exceeds_60_90_by_ci", False) for item in split_checks
    )
    recall = per_class_summary.loc[
        per_class_summary["input_group"].eq("A_strict_38")
        & per_class_summary["window_s"].eq(120)
        & per_class_summary["split_type"].isin(["severity_blocked", "severity_extrapolation"])
    ]
    checks["all_ood_recall_classes_have_support_ge_5"] = bool(
        not recall.empty and (recall["support_min"].ge(5) | recall["support_min"].eq(0)).all()
    )
    checks["all_ood_recall_classes_mean_at_least_0_80"] = bool(
        not recall.empty and recall.loc[recall["n_evaluable_splits"].gt(0), "recall_mean"].ge(0.80).all()
    )
    required = [
        "ood_120_macro_f1_mean_at_least_0_50",
        "ood_120_balanced_accuracy_mean_at_least_0_50",
        "ood_120_ci_advantage_over_controls",
        "all_ood_recall_classes_have_support_ge_5",
        "all_ood_recall_classes_mean_at_least_0_80",
    ]
    checks["decision"] = "A_ENTER_GRU_LSTM_TCN_GATE" if all(checks.get(name, False) for name in required) else "B_REQUIRE_FURTHER_DATA_OR_VALIDATION"
    return checks


def run_experiment(
    project_root: str | Path = PROJECT_ROOT,
    result_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run parser recovery, severity OOD validation, audits, and figures."""
    project_root = Path(project_root)
    result_root = Path(result_root) if result_root is not None else project_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)

    event_frame, event_summary = scan_event_inventory(project_root, DEFAULT_INVENTORY, CLASS_LABELS)
    event_csv = result_root / "13_event_parser_recovery.csv"
    event_json = result_root / "13_event_parser_recovery.json"
    write_event_outputs(event_frame, event_summary, event_csv, event_json)

    dataset, points = build_ood_feature_dataset(project_root, DEFAULT_INVENTORY, event_csv)
    required_windows = set(WINDOWS_S)
    if set(dataset["window_s"].unique()) != required_windows:
        raise AssertionError("OOD dataset does not contain all 30/60/90/120 s windows")
    per_entity = dataset.groupby("sample_id").agg(
        window_count=("window_s", "nunique"),
        strict_120=("strict_pre_protection", lambda values: bool(dataset.loc[values.index].loc[dataset.loc[values.index, "window_s"].eq(120), "strict_pre_protection"].all())),
        protection_known=("protection_time_known", "all"),
        accident_class=("accident_class", "first"),
    )
    cohort_ids = per_entity.index[
        per_entity["window_count"].eq(len(WINDOWS_S))
        & per_entity["protection_known"]
        & per_entity["strict_120"]
    ]
    cohort = dataset.loc[dataset["sample_id"].isin(cohort_ids)].copy()
    if cohort.empty or cohort["sample_id"].nunique() < 20:
        raise AssertionError("Strict matched OOD cohort is unexpectedly small")
    cohort_entities = _entities(cohort)
    random = random_assignments(cohort_entities)
    blocked, blocked_excluded = severity_blocked_assignments(cohort_entities)
    extrapolation, extrapolation_excluded = severity_extrapolation_assignments(cohort_entities)
    assignments = pd.concat([random, blocked, extrapolation], ignore_index=True)
    for split_type, group in assignments.groupby(["split_type", "split_id"], sort=False):
        active = group.loc[group["partition"].isin(["train", "validation", "test"])]
        if active["sample_id"].duplicated().any():
            raise AssertionError(f"Trajectory crossed split partitions for {split_type}")
        for partition in ("train", "validation", "test"):
            if group.loc[group["partition"].eq(partition)].empty:
                raise AssertionError(f"Empty {partition} partition for {split_type}")
    gaps = nearest_severity_gaps(assignments)
    feature_groups, removed_features = input_feature_groups()
    metrics, recalls = evaluate_splits(cohort, assignments, feature_groups)
    metric_summary = aggregate_metrics(metrics)
    per_class_summary = aggregate_per_class(recalls)
    if metric_summary.empty or per_class_summary.empty:
        raise AssertionError("OOD evaluation produced no selected test metrics")

    split_inventory = assignments.merge(
        cohort_entities[["sample_id", "severity_source"]], on="sample_id", how="left", validate="many_to_one"
    )
    split_inventory.to_csv(result_root / "13_split_inventory.csv", index=False)
    metrics.to_csv(result_root / "13_ood_metrics.csv", index=False)
    per_class_summary.to_csv(result_root / "13_per_class_ood.csv", index=False)
    gaps.to_csv(result_root / "13_nearest_severity_gap.csv", index=False)
    metric_summary.to_csv(result_root / "13_ood_metric_summary.csv", index=False)
    figure_paths = _write_figures(metric_summary, per_class_summary, gaps, event_frame, result_root)

    gap_summary = {
        split_type: _describe(group["nearest_severity_gap"])
        for split_type, group in gaps.groupby("split_type", sort=False)
    }
    cohort_counts = cohort_entities["accident_class"].value_counts().sort_index().to_dict()
    summary = {
        "experiment": "OOD severity-blocked and extrapolation validation",
        "classes_fixed_12": list(CLASS_LABELS),
        "severity_definitions": SEVERITY_DEFINITIONS,
        "severity_source": "NPPAD README documents that case numbers correspond to each class's severity; units remain class-specific.",
        "windows_s": list(WINDOWS_S),
        "ood_matched_cohort_count": int(cohort_entities.shape[0]),
        "ood_matched_cohort_counts": {str(key): int(value) for key, value in cohort_counts.items()},
        "parser_recovery": {
            "csv": "results/13_event_parser_recovery.csv",
            "json": "results/13_event_parser_recovery.json",
            "trajectory_total": event_summary["trajectory_total"],
            "protection_parsed": event_summary["protection_parsed"],
            "class_summary": event_summary["class_summary"],
        },
        "split_protocol": {
            "random": "10 repeated 60/20/20 stratified splits on complete trajectory/sample_id rows",
            "severity_blocked": "low/middle/high contiguous within-class test blocks with one adjacent severity guard row excluded from train/validation",
            "severity_extrapolation": "low train to high test and high train to low test, with one adjacent severity guard row",
        },
        "split_exclusions": {
            "severity_blocked": blocked_excluded,
            "severity_extrapolation": extrapolation_excluded,
        },
        "input_groups": {key: list(value) for key, value in removed_features.items()},
        "metric_summary_120s": metric_summary.loc[
            metric_summary["window_s"].eq(120) & metric_summary["input_group"].eq("A_strict_38")
        ].to_dict(orient="records"),
        "matched_window_summary_A": metric_summary.loc[
            metric_summary["input_group"].eq("A_strict_38") & metric_summary["window_s"].isin(OOD_WINDOWS_S)
        ].to_dict(orient="records"),
        "nearest_severity_gap_summary": gap_summary,
        "gate": _gate_summary(metric_summary, per_class_summary),
        "figure_paths": figure_paths,
        "assertions": {
            "trajectory_level_assignments": True,
            "train_only_model_fit": True,
            "same_ood_cohort_for_60_90_120": True,
            "no_history_overwritten": True,
        },
    }
    (result_root / "13_ood_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps({"decision": result["gate"]["decision"], "cohort": result["ood_matched_cohort_count"]}, ensure_ascii=False))
