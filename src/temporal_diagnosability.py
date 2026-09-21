"""Temporal diagnosability analysis for the 11 milestone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

try:
    from .multi_accident_features import (
        DEFAULT_INVENTORY,
        FIRST_VERSION_CLASSES,
        SUMMARY_STATISTICS,
        feature_columns,
        input_feature_groups,
        strict_process_features,
    )
    from .multi_accident_models import model_suite
    from .severity_features import INJECTION_S, extract_window_features, window_sample_points
except ImportError:
    from multi_accident_features import (
        DEFAULT_INVENTORY,
        FIRST_VERSION_CLASSES,
        SUMMARY_STATISTICS,
        feature_columns,
        input_feature_groups,
        strict_process_features,
    )
    from multi_accident_models import model_suite
    from severity_features import INJECTION_S, extract_window_features, window_sample_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_INVENTORY = PROJECT_ROOT / "results" / "09_multi_accident_event_inventory.csv"
TEMPORAL_WINDOWS_S = (10, 20, 30, 40, 60, 90, 120)
SCOPES = ("all_sample", "strict_pre_protection_only")


def build_temporal_dataset(
    project_root: str | Path = PROJECT_ROOT,
    inventory_path: str | Path = DEFAULT_INVENTORY,
    event_inventory_path: str | Path = DEFAULT_EVENT_INVENTORY,
    windows_s: tuple[int, ...] = TEMPORAL_WINDOWS_S,
    classes: tuple[str, ...] = FIRST_VERSION_CLASSES,
    features: list[str] | None = None,
    injection_s: float = INJECTION_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build arbitrary early-window features and explicit protection eligibility."""
    project_root = Path(project_root)
    features = features or strict_process_features()
    inventory = pd.read_csv(inventory_path)
    events = pd.read_csv(event_inventory_path)[["sample_id", "first_protection_s"]]
    selected = inventory.loc[inventory["accident_class"].isin(classes)].merge(
        events, on="sample_id", how="left", validate="one_to_one"
    )
    if selected["first_protection_s"].notna().sum() == 0:
        raise ValueError("No parsed first protection times are available")

    rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for record in selected.sort_values("sample_id").itertuples(index=False):
        path = project_root / Path(record.operation_csv)
        frame = pd.read_csv(path)
        if "TIME" not in frame or any(feature not in frame for feature in features):
            raise ValueError(f"Strict schema missing from {path}")
        time = pd.to_numeric(frame["TIME"], errors="coerce")
        first_protection = float(record.first_protection_s) if pd.notna(record.first_protection_s) else np.nan
        protection_known = bool(np.isfinite(first_protection))
        for window_s in windows_s:
            if float(time.max()) < window_s:
                continue
            points = window_sample_points(frame, window_s, injection_s)
            last_sample_s = float(points[-1])
            strict_pre = bool(protection_known and first_protection > last_sample_s)
            extracted = extract_window_features(frame, features, window_s, injection_s)
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "accident_class": record.accident_class,
                    "split": record.split,
                    "window_s": int(window_s),
                    "first_protection_s": first_protection if protection_known else np.nan,
                    "protection_time_known": protection_known,
                    "last_sample_s": last_sample_s,
                    "strict_pre_protection": strict_pre,
                    **extracted,
                }
            )
            point_rows.append(
                {
                    "sample_id": record.sample_id,
                    "accident_class": record.accident_class,
                    "split": record.split,
                    "window_s": int(window_s),
                    "first_protection_s": first_protection if protection_known else np.nan,
                    "protection_time_known": protection_known,
                    "strict_pre_protection": strict_pre,
                    "sample_count": len(points),
                    "first_sample_s": float(points[0]),
                    "last_sample_s": last_sample_s,
                    "sample_points_s": ",".join(f"{point:g}" for point in points),
                }
            )

    dataset = pd.DataFrame(rows).sort_values(["window_s", "sample_id"]).reset_index(drop=True)
    points = pd.DataFrame(point_rows).sort_values(["window_s", "sample_id"]).reset_index(drop=True)
    expected = set(feature_columns(features))
    if not expected.issubset(dataset.columns):
        raise AssertionError("Temporal extraction did not produce the expected summary columns")
    if dataset[["sample_id", "window_s"]].duplicated().any():
        raise AssertionError("Duplicate trajectory-window rows")
    if set(dataset["window_s"]) != set(windows_s):
        raise AssertionError("At least one requested window produced no rows")
    if dataset[feature_columns(features)].isna().any().any():
        raise AssertionError("Temporal feature matrix contains missing values")
    strict = dataset.loc[dataset["strict_pre_protection"]]
    if strict["first_protection_s"].isna().any():
        raise AssertionError("Unknown protection times entered strict pre-protection data")
    if not (strict["first_protection_s"] > strict["last_sample_s"]).all():
        raise AssertionError("Strict data contains a window reaching protection time")
    return dataset, points


def _model_columns(features: list[str]) -> list[str]:
    return [f"{feature}__{statistic}" for feature in features for statistic in SUMMARY_STATISTICS]


def _fit_predict(
    model_name: str,
    model: object | None,
    train: pd.DataFrame,
    evaluated: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, str]:
    y_train = train["accident_class"]
    if model is None or y_train.nunique() < 2:
        majority = y_train.value_counts().sort_index().index[0]
        return np.repeat(majority, len(evaluated)), "majority_fallback" if model is not None else "majority_baseline"
    fitted = model.fit(train[columns], y_train)
    return fitted.predict(evaluated[columns]), "fitted"


def evaluate_temporal_models(
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    class_labels: list[str],
    windows_s: tuple[int, ...] = TEMPORAL_WINDOWS_S,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all-sample and strict pre-protection scopes with fixed labels."""
    metric_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for scope in SCOPES:
        scoped = dataset if scope == "all_sample" else dataset.loc[dataset["strict_pre_protection"]]
        if scope == "strict_pre_protection_only" and scoped["first_protection_s"].isna().any():
            raise AssertionError("Unknown protection time is present in strict scope")
        for window_s in windows_s:
            window = scoped.loc[scoped["window_s"].eq(window_s)].copy()
            for group_name, features in feature_groups.items():
                columns = _model_columns(features)
                train = window.loc[window["split"].eq("train")]
                if train.empty:
                    raise AssertionError(f"No train trajectories for {scope}/{group_name}/{window_s}s")
                for model_name, model in model_suite().items():
                    for eval_split in ("validation", "test"):
                        evaluated = window.loc[window["split"].eq(eval_split)]
                        if evaluated.empty:
                            continue
                        y_true = evaluated["accident_class"].to_numpy()
                        y_pred, fit_status = _fit_predict(model_name, model, train, evaluated, columns)
                        matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
                        supports = pd.Series(y_true).value_counts().reindex(class_labels, fill_value=0).to_numpy()
                        observed = [label for label, support in zip(class_labels, supports) if support > 0]
                        macro_f1_observed = (
                            float(f1_score(y_true, y_pred, labels=observed, average="macro", zero_division=0))
                            if observed else np.nan
                        )
                        metric_rows.append(
                            {
                                "scope": scope,
                                "window_s": int(window_s),
                                "input_group": group_name,
                                "model": model_name,
                                "eval_split": eval_split,
                                "n_samples": int(len(evaluated)),
                                "n_features": int(len(columns)),
                                "n_classes_train": int(train["accident_class"].nunique()),
                                "n_classes_eval": int(len(observed)),
                                "n_unknown_protection": int(evaluated["first_protection_s"].isna().sum()),
                                "accuracy": float(accuracy_score(y_true, y_pred)),
                                "macro_f1": float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
                                "macro_f1_observed_classes": macro_f1_observed,
                                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                                "fit_status": fit_status,
                            }
                        )
                        recalls = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
                        for label, recall, support, row in zip(class_labels, recalls, supports, matrix):
                            recall_rows.append(
                                {
                                    "scope": scope,
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
                                        "scope": scope,
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


def select_validation_model(
    metrics: pd.DataFrame,
    scope: str,
    input_group: str,
    window_s: int,
) -> pd.Series:
    """Choose one model using only validation metrics for one scope/window."""
    candidates = metrics.loc[
        metrics["scope"].eq(scope)
        & metrics["input_group"].eq(input_group)
        & metrics["window_s"].eq(window_s)
        & metrics["eval_split"].eq("validation")
    ]
    if candidates.empty:
        raise ValueError(f"No validation rows for {scope}/{input_group}/{window_s}s")
    return candidates.sort_values(
        ["macro_f1", "balanced_accuracy", "macro_f1_observed_classes", "model"],
        ascending=[False, False, False, True],
    ).iloc[0]


def selected_test_row(metrics: pd.DataFrame, selection: pd.Series) -> pd.Series:
    rows = metrics.loc[
        metrics["scope"].eq(selection["scope"])
        & metrics["input_group"].eq(selection["input_group"])
        & metrics["window_s"].eq(selection["window_s"])
        & metrics["model"].eq(selection["model"])
        & metrics["eval_split"].eq("test")
    ]
    if len(rows) != 1:
        raise ValueError("Validation selection does not have one test row")
    return rows.iloc[0]
