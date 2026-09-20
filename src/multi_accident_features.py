"""Feature extraction and input-group definitions for multi-accident classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .features import feature_group_table
    from .severity_features import (
        INJECTION_S,
        SUMMARY_STATISTICS,
        extract_window_features,
        window_sample_points,
    )
except ImportError:
    from features import feature_group_table
    from severity_features import (
        INJECTION_S,
        SUMMARY_STATISTICS,
        extract_window_features,
        window_sample_points,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = PROJECT_ROOT / "results" / "09_multi_accident_class_inventory.csv"
DEFAULT_LEAKAGE_FLAGS = PROJECT_ROOT / "results" / "09_multi_accident_leakage_flags.csv"
DEFAULT_INITIAL_CONDITIONS = PROJECT_ROOT / "results" / "09_multi_accident_initial_conditions.csv"
WINDOWS_S = (30, 60, 120)
PRIMARY_WINDOWS_S = (30, 60)
FIRST_VERSION_CLASSES = (
    "FLB",
    "LLB",
    "LOCA",
    "LOCAC",
    "LR",
    "MD",
    "RI",
    "RW",
    "SGATR",
    "SGBTR",
    "SLBIC",
    "SLBOC",
)


def strict_process_features() -> list[str]:
    """Return exactly the 38 process variables retained by the 03 audit."""
    table = feature_group_table()
    features = table.loc[table["ml_policy"].eq("candidate"), "feature"].tolist()
    if len(features) != 38 or len(set(features)) != len(features):
        raise ValueError("The strict process feature set is not the expected 38 variables")
    return features


def input_feature_groups(
    leakage_path: str | Path = DEFAULT_LEAKAGE_FLAGS,
    initial_conditions_path: str | Path = DEFAULT_INITIAL_CONDITIONS,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build A/B/C groups from the saved 09 audit, without inspecting labels."""
    strict = strict_process_features()
    leakage = pd.read_csv(leakage_path)
    potential = leakage.loc[
        leakage["potential_label_leakage"].astype(str).str.lower().eq("true"), "feature"
    ]
    leakage_in_strict = sorted(set(strict) & set(potential))

    initial = pd.read_csv(initial_conditions_path)
    slbic_initial = initial.loc[
        initial["accident_class"].eq("SLBIC")
        & initial["class_initial_condition_flag"].astype(str).str.lower().eq("true"),
        "feature",
    ]
    initial_in_strict = sorted(set(strict) & set(slbic_initial))

    groups = {
        "A_strict_38": strict,
        "B_without_potential_leakage": [feature for feature in strict if feature not in leakage_in_strict],
        "C_without_SLBIC_initial": [feature for feature in strict if feature not in initial_in_strict],
    }
    removed = {
        "B_without_potential_leakage": leakage_in_strict,
        "C_without_SLBIC_initial": initial_in_strict,
    }
    if leakage_in_strict != ["LVCR"]:
        raise AssertionError(f"Unexpected candidate leakage set: {leakage_in_strict}")
    if len(initial_in_strict) != 14:
        raise AssertionError(f"Unexpected SLBIC initial-condition set: {initial_in_strict}")
    if not set(groups["B_without_potential_leakage"]).issubset(strict):
        raise AssertionError("Input group B contains a non-strict feature")
    if not set(groups["C_without_SLBIC_initial"]).issubset(strict):
        raise AssertionError("Input group C contains a non-strict feature")
    return groups, removed


def feature_columns(features: list[str]) -> list[str]:
    return [f"{feature}__{statistic}" for feature in features for statistic in SUMMARY_STATISTICS]


def build_feature_dataset(
    project_root: str | Path = PROJECT_ROOT,
    inventory_path: str | Path = DEFAULT_INVENTORY,
    windows_s: tuple[int, ...] = WINDOWS_S,
    classes: tuple[str, ...] = FIRST_VERSION_CLASSES,
    features: list[str] | None = None,
    injection_s: float = INJECTION_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per complete trajectory and available early window."""
    project_root = Path(project_root)
    features = features or strict_process_features()
    inventory = pd.read_csv(inventory_path)
    selected = inventory.loc[inventory["accident_class"].isin(classes)].copy()
    if selected.empty:
        raise ValueError("No first-version accident trajectories found")

    rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for record in selected.sort_values("sample_id").itertuples(index=False):
        path = project_root / Path(record.operation_csv)
        frame = pd.read_csv(path)
        if "TIME" not in frame or any(feature not in frame for feature in features):
            raise ValueError(f"Strict schema missing from {path}")
        for window_s in windows_s:
            if not bool(getattr(record, f"available_{window_s}s")):
                continue
            points = window_sample_points(frame, window_s, injection_s)
            extracted = extract_window_features(frame, features, window_s, injection_s)
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "accident_class": record.accident_class,
                    "split": record.split,
                    "window_s": int(window_s),
                    **extracted,
                }
            )
            point_rows.append(
                {
                    "sample_id": record.sample_id,
                    "accident_class": record.accident_class,
                    "split": record.split,
                    "window_s": int(window_s),
                    "sample_count": len(points),
                    "first_sample_s": float(points[0]),
                    "last_sample_s": float(points[-1]),
                    "sample_points_s": ",".join(f"{point:g}" for point in points),
                }
            )

    dataset = pd.DataFrame(rows).sort_values(["window_s", "sample_id"]).reset_index(drop=True)
    points = pd.DataFrame(point_rows).sort_values(["window_s", "sample_id"]).reset_index(drop=True)
    expected = set(feature_columns(features))
    if not expected.issubset(dataset.columns):
        raise AssertionError("Feature extraction did not produce the expected summary columns")
    if dataset[["sample_id", "window_s"]].duplicated().any():
        raise AssertionError("Duplicate trajectory-window rows")
    if set(dataset["window_s"]) != set(windows_s):
        raise AssertionError("At least one requested window produced no rows")
    if dataset[feature_columns(features)].isna().any().any():
        raise AssertionError("Feature matrix contains missing values")
    return dataset, points
