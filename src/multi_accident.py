"""Audit all fixed-power NPPAD accident trajectories before classification."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .features import feature_group_table
except ImportError:
    from features import feature_group_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
OPERATION_ROOT = DATA_ROOT / "Operation_csv_data"
REPORT_ROOT = DATA_ROOT / "NPPAD"
RESULT_ROOT = PROJECT_ROOT / "results"
WINDOWS_S = (30, 60, 120)
KEY_INITIAL_FEATURES = ("P", "PWR", "TAVG", "LVPZ", "TSAT", "VOL")

_EVENT_LINE = re.compile(
    r"^\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s+sec,\s*(?P<description>.*)$"
)
_EVENT_PATTERNS = {
    "injection": r"Malfunction\s*#\s*[0-9]+\s+Fraction",
    "scram": r"Reactor Scram",
    "hpi_pump_1": r"HPI Pump #1 Position Change:\s*100%",
    "hpi_pump_2": r"HPI Pump #2 Position Change:\s*100%",
    "hpsi_start": r"HPSI start",
}


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if pd.isna(value):
        return None
    return str(value)


def discover_accident_classes(
    operation_root: str | Path = OPERATION_ROOT,
) -> list[Path]:
    """Return all fixed-power accident directories, excluding Normal."""
    root = Path(operation_root)
    if not root.exists():
        raise FileNotFoundError(f"Operation data directory not found: {root}")
    classes = [path for path in root.iterdir() if path.is_dir() and path.name.lower() != "normal"]
    return sorted(classes, key=lambda path: path.name)


def parse_transient_report(report_path: str | Path) -> dict[str, Any]:
    """Parse first available accident/protection event times from one report."""
    path = Path(report_path)
    values: dict[str, Any] = {
        "report_exists": path.exists(),
        "report_line_count": 0,
        "injection_s": None,
        "injection_event": None,
        "scram_s": None,
        "scram_event": None,
        "hpi_pump_1_s": None,
        "hpi_pump_1_event": None,
        "hpi_pump_2_s": None,
        "hpi_pump_2_event": None,
        "hpsi_start_s": None,
        "hpsi_start_event": None,
    }
    if not path.exists():
        values["event_parse_ok"] = False
        values["first_protection_s"] = None
        values["first_protection_event"] = None
        return values

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    values["report_line_count"] = len(lines)
    for line in lines:
        match = _EVENT_LINE.match(line)
        if not match:
            continue
        timestamp = float(match.group("time"))
        description = match.group("description").strip()
        for name, pattern in _EVENT_PATTERNS.items():
            time_key = f"{name}_s"
            event_key = f"{name}_event"
            if values[time_key] is None and re.search(pattern, description, flags=re.IGNORECASE):
                values[time_key] = timestamp
                values[event_key] = description

    protection_events = [
        (values["scram_s"], "scram"),
        (values["hpi_pump_1_s"], "hpi_pump_1"),
        (values["hpi_pump_2_s"], "hpi_pump_2"),
    ]
    protection_events = [item for item in protection_events if item[0] is not None]
    if protection_events:
        first_time, first_name = min(protection_events, key=lambda item: item[0])
        values["first_protection_s"] = first_time
        values["first_protection_event"] = values[f"{first_name}_event"]
    else:
        values["first_protection_s"] = None
        values["first_protection_event"] = None
    values["event_parse_ok"] = values["injection_s"] is not None
    return values


def _time_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    time = pd.to_numeric(frame["TIME"], errors="coerce")
    differences = time.diff().dropna()
    return {
        "rows": int(len(frame)),
        "time_start_s": float(time.iloc[0]) if len(time) else None,
        "time_end_s": float(time.iloc[-1]) if len(time) else None,
        "sampling_median_s": float(differences.median()) if len(differences) else None,
        "sampling_min_s": float(differences.min()) if len(differences) else None,
        "sampling_max_s": float(differences.max()) if len(differences) else None,
        "time_numeric": bool(time.notna().all()),
        "time_strictly_increasing": bool(differences.gt(0).all()),
        "time_duplicate_count": int(time.duplicated().sum()),
        "nonuniform_sampling_intervals": int(
            (~np.isclose(differences.to_numpy(float), differences.median(), atol=1e-6)).sum()
        ) if len(differences) else 0,
    }


def _assign_group_splits(inventory: pd.DataFrame) -> pd.Series:
    """Assign each complete trajectory to one split; never split time rows."""
    assignments: dict[str, str] = {}
    for accident_class, subset in inventory.groupby("accident_class", sort=True):
        ordered = subset.sort_values(["numeric_case_id", "sample_id"], na_position="last")
        if len(ordered) < 3:
            split_by_rank = {sample_id: "train" for sample_id in ordered["sample_id"]}
        else:
            split_by_rank = {}
            for rank, sample_id in enumerate(ordered["sample_id"]):
                phase = rank % 10
                split_by_rank[sample_id] = (
                    "test" if phase == 0 else "validation" if phase == 1 else "train"
                )
        assignments.update(split_by_rank)
    return inventory["sample_id"].map(assignments)


def _feature_meta(feature_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        row.feature: row._asdict() if hasattr(row, "_asdict") else row.to_dict()
        for row in feature_table.itertuples(index=False)
    }


def _write_figures(
    class_summary: pd.DataFrame,
    initial_conditions: pd.DataFrame,
    inventory: pd.DataFrame,
    result_root: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = result_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    counts = class_summary.sort_values("trajectory_count")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(counts["accident_class"], counts["trajectory_count"], color="#2f6f9f")
    ax.set_xlabel("Trajectories")
    ax.set_title("Multi-accident trajectory counts")
    fig.tight_layout()
    path = figure_root / "09_class_counts.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(_relative_path(path))

    coverage = class_summary.set_index("accident_class")[
        [f"available_{window}s" for window in WINDOWS_S]
    ].div(class_summary.set_index("accident_class")["trajectory_count"], axis=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(coverage.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(coverage)), coverage.index)
    ax.set_xticks(range(len(WINDOWS_S)), [f"{window} s" for window in WINDOWS_S])
    ax.set_title("Early-window availability by accident class")
    fig.colorbar(image, ax=ax, label="Coverage")
    fig.tight_layout()
    path = figure_root / "09_available_window_coverage.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(_relative_path(path))

    key = initial_conditions[initial_conditions["feature"].isin(KEY_INITIAL_FEATURES)]
    means = key.pivot(index="accident_class", columns="feature", values="initial_mean")
    centered = means - means.median(axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in [name for name in KEY_INITIAL_FEATURES if name in centered.columns]:
        ax.plot(centered.index, centered[feature], marker="o", label=feature)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Difference from class-median t=0 value")
    ax.set_title("Key t=0 initial-condition differences")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(ncol=3)
    fig.tight_layout()
    path = figure_root / "09_initial_condition_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(_relative_path(path))
    return paths


def run_audit(
    project_root: str | Path = PROJECT_ROOT,
    result_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete multi-accident dataset audit and write lightweight outputs."""
    project_root = Path(project_root)
    data_root = project_root / "data" / "NuclearPowerPlantAccidentData"
    operation_root = data_root / "Operation_csv_data"
    report_root = data_root / "NPPAD"
    result_root = Path(result_root) if result_root is not None else project_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)

    feature_table = feature_group_table()
    feature_meta = _feature_meta(feature_table)
    strict_features = feature_table.loc[
        feature_table["ml_policy"].eq("candidate"), "feature"
    ].tolist()
    assert len(strict_features) == 38

    class_dirs = discover_accident_classes(operation_root)
    assert class_dirs, "No non-Normal accident directories found"
    inventory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    initial_rows: list[dict[str, Any]] = []
    class_feature_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"presence": 0, "changed": 0, "low_cardinality": 0})
    )
    feature_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"presence": 0, "changed": 0, "low_cardinality": 0}
    )
    reference_columns = feature_table["feature"].tolist()

    for class_dir in class_dirs:
        csv_paths = sorted(class_dir.glob("*.csv"), key=_numeric_sort_key)
        if not csv_paths:
            raise AssertionError(f"Accident class has no CSV trajectories: {class_dir.name}")
        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path)
            if "TIME" not in frame.columns:
                raise AssertionError(f"TIME missing from {csv_path}")
            numeric_case_id = int(csv_path.stem) if csv_path.stem.lstrip("-").isdigit() else None
            sample_id = f"{class_dir.name}_{csv_path.stem}"
            time_info = _time_metrics(frame)
            report_path = report_root / class_dir.name / f"{csv_path.stem}Transient Report.txt"
            events = parse_transient_report(report_path)
            extra_columns = sorted(set(frame.columns) - set(reference_columns))
            missing_columns = sorted(set(reference_columns) - set(frame.columns))
            inventory_rows.append(
                {
                    "sample_id": sample_id,
                    "accident_class": class_dir.name,
                    "numeric_case_id": numeric_case_id,
                    "operation_csv": _relative_path(csv_path),
                    "rows": time_info["rows"],
                    "column_count": int(len(frame.columns)),
                    "time_start_s": time_info["time_start_s"],
                    "time_end_s": time_info["time_end_s"],
                    "sampling_median_s": time_info["sampling_median_s"],
                    "sampling_min_s": time_info["sampling_min_s"],
                    "sampling_max_s": time_info["sampling_max_s"],
                    "time_numeric": time_info["time_numeric"],
                    "time_strictly_increasing": time_info["time_strictly_increasing"],
                    "time_duplicate_count": time_info["time_duplicate_count"],
                    "nonuniform_sampling_intervals": time_info["nonuniform_sampling_intervals"],
                    "missing_cells": int(frame.isna().sum().sum()),
                    "missing_columns": int(frame.isna().any().sum()),
                    "schema_match_03_97": bool(list(frame.columns) == reference_columns),
                    "schema_extra_columns": ";".join(extra_columns),
                    "schema_missing_columns": ";".join(missing_columns),
                    "split": None,
                    "split_strategy": None,
                    **{f"available_{window}s": bool(time_info["time_end_s"] >= window) for window in WINDOWS_S},
                }
            )
            event_rows.append(
                {
                    "sample_id": sample_id,
                    "accident_class": class_dir.name,
                    "operation_csv": _relative_path(csv_path),
                    "transient_report": _relative_path(report_path),
                    **events,
                }
            )

            for feature in frame.columns:
                if feature == "TIME":
                    continue
                series = pd.to_numeric(frame[feature], errors="coerce")
                values = series.dropna().to_numpy(float)
                if not len(values):
                    presence = changed = False
                    low_cardinality = True
                else:
                    scale = max(float(np.max(np.abs(values))), 1.0)
                    tolerance = max(1e-9, scale * 1e-6)
                    presence = bool(np.any(np.abs(values) > tolerance))
                    changed = bool(np.any(np.abs(values - values[0]) > tolerance))
                    low_cardinality = int(series.round(6).nunique(dropna=True)) <= 5
                stats = class_feature_stats[class_dir.name][feature]
                stats["presence"] += int(presence)
                stats["changed"] += int(changed)
                stats["low_cardinality"] += int(low_cardinality)
                feature_totals[feature]["presence"] += int(presence)
                feature_totals[feature]["changed"] += int(changed)
                feature_totals[feature]["low_cardinality"] += int(low_cardinality)

            for feature in strict_features:
                if feature not in frame:
                    raise AssertionError(f"Strict process feature missing from {csv_path}: {feature}")
                initial = pd.to_numeric(frame[feature], errors="coerce").iloc[0]
                initial_rows.append(
                    {
                        "sample_id": sample_id,
                        "accident_class": class_dir.name,
                        "feature": feature,
                        "initial_value": float(initial) if pd.notna(initial) else np.nan,
                    }
                )

    inventory = pd.DataFrame(inventory_rows)
    inventory["split"] = _assign_group_splits(inventory)
    inventory["split_strategy"] = np.where(
        inventory.groupby("accident_class")["sample_id"].transform("size").lt(3),
        "train_only_class_too_small_for_three_way_split",
        "deterministic_class_stratified_group_interleave",
    )
    events = pd.DataFrame(event_rows)
    initial_values = pd.DataFrame(initial_rows)
    assert inventory["sample_id"].is_unique
    assert set(inventory["sample_id"]) == set(events["sample_id"])
    assert not (set(inventory.loc[inventory.split == "train", "sample_id"]) & set(inventory.loc[inventory.split == "validation", "sample_id"]))
    assert not (set(inventory.loc[inventory.split == "train", "sample_id"]) & set(inventory.loc[inventory.split == "test", "sample_id"]))
    assert not (set(inventory.loc[inventory.split == "validation", "sample_id"]) & set(inventory.loc[inventory.split == "test", "sample_id"]))

    initial_aggregated = (
        initial_values.groupby(["feature", "accident_class"], as_index=False)["initial_value"]
        .agg(initial_count="count", initial_mean="mean", initial_std="std", initial_min="min", initial_max="max")
    )
    feature_initial_ranges = initial_aggregated.groupby("feature")["initial_mean"].agg(
        max_pairwise_mean_diff=lambda values: float(values.max() - values.min()),
        global_class_mean_max=lambda values: float(values.abs().max()),
    ).reset_index()
    global_initial_std = initial_values.groupby("feature")["initial_value"].std().rename("global_initial_std").reset_index()
    initial_aggregated = initial_aggregated.merge(feature_initial_ranges, on="feature").merge(global_initial_std, on="feature")
    initial_aggregated["initial_condition_flag"] = initial_aggregated["max_pairwise_mean_diff"].gt(
        np.maximum(1e-6, initial_aggregated["global_class_mean_max"] * 1e-6)
    )
    initial_aggregated["class_mean_vs_other_max_diff"] = np.nan
    initial_aggregated["class_mean_vs_global_class_median_diff"] = np.nan
    for feature, feature_rows in initial_aggregated.groupby("feature"):
        class_median = float(feature_rows["initial_mean"].median())
        for index, row in feature_rows.iterrows():
            other_means = feature_rows.loc[
                feature_rows["accident_class"] != row["accident_class"], "initial_mean"
            ]
            if len(other_means):
                initial_aggregated.loc[index, "class_mean_vs_other_max_diff"] = float(
                    np.abs(row["initial_mean"] - other_means).max()
                )
            initial_aggregated.loc[index, "class_mean_vs_global_class_median_diff"] = abs(
                float(row["initial_mean"]) - class_median
            )
    initial_aggregated["class_initial_condition_flag"] = initial_aggregated[
        "class_mean_vs_global_class_median_diff"
    ].gt(np.maximum(1e-6, initial_aggregated["global_class_mean_max"] * 1e-6))

    leakage_rows: list[dict[str, Any]] = []
    total_trajectories = len(inventory)
    class_counts = inventory.groupby("accident_class").size().to_dict()
    for feature in sorted(feature_totals):
        meta = feature_meta.get(
            feature,
            {
                "readme_definition": "Not in the 03 feature_groups.csv table",
                "group": "unknown_extra",
                "ml_policy": "review",
            },
        )
        total = feature_totals[feature]
        for accident_class, class_count in sorted(class_counts.items()):
            class_stats = class_feature_stats[accident_class][feature]
            other_count = total_trajectories - class_count
            other_stats = {
                key: total[key] - class_stats[key]
                for key in ("presence", "changed", "low_cardinality")
            }
            class_presence = class_stats["presence"] / class_count
            other_presence = other_stats["presence"] / other_count if other_count else 0.0
            class_changed = class_stats["changed"] / class_count
            other_changed = other_stats["changed"] / other_count if other_count else 0.0
            class_low = class_stats["low_cardinality"] / class_count
            other_low = other_stats["low_cardinality"] / other_count if other_count else 0.0
            triggers: list[str] = []
            if class_presence >= 0.9 and other_presence <= 0.1:
                triggers.append("class_specific_nonzero")
            if class_changed >= 0.9 and other_changed <= 0.1:
                triggers.append("class_specific_state_change")
            if class_low >= 0.9 and other_low <= 0.1 and class_changed >= 0.5:
                triggers.append("class_specific_low_cardinality_state")
            class_initial = initial_aggregated.loc[
                (initial_aggregated["feature"] == feature)
                & (initial_aggregated["accident_class"] == accident_class)
            ]
            initial_code = bool(
                len(class_initial)
                and class_initial["class_initial_condition_flag"].iloc[0]
                and class_count >= 1
            )
            if initial_code:
                triggers.append("class_specific_initial_constant")
            if not triggers:
                continue
            potential_leakage = any(
                trigger != "class_specific_initial_constant" for trigger in triggers
            )
            leakage_rows.append(
                {
                    "feature": feature,
                    "readme_definition": meta.get("readme_definition"),
                    "group": meta.get("group"),
                    "ml_policy_from_03": meta.get("ml_policy"),
                    "accident_class": accident_class,
                    "class_trajectory_count": class_count,
                    "other_trajectory_count": other_count,
                    "class_presence_rate": class_presence,
                    "other_presence_rate": other_presence,
                    "class_state_change_rate": class_changed,
                    "other_state_change_rate": other_changed,
                    "class_low_cardinality_rate": class_low,
                    "other_low_cardinality_rate": other_low,
                    "risk_type": "potential_label_leakage" if potential_leakage else "initial_condition_confounding",
                    "trigger": ";".join(triggers),
                    "confidence_note": "low_sample_class" if class_count < 3 else "review_candidate",
                    "potential_label_leakage": potential_leakage,
                    "auto_removed": False,
                    "recommendation": "Review before training; audit does not delete this variable automatically.",
                }
            )
    leakage = pd.DataFrame(leakage_rows)

    class_summary_rows: list[dict[str, Any]] = []
    for accident_class, subset in inventory.groupby("accident_class", sort=True):
        event_subset = events.loc[events["accident_class"] == accident_class]
        class_initial = initial_aggregated.loc[initial_aggregated["accident_class"] == accident_class]
        count = len(subset)
        class_summary_rows.append(
            {
                "accident_class": accident_class,
                "trajectory_count": count,
                "total_rows": int(subset["rows"].sum()),
                "rows_min": int(subset["rows"].min()),
                "rows_max": int(subset["rows"].max()),
                "time_end_min_s": float(subset["time_end_s"].min()),
                "time_end_max_s": float(subset["time_end_s"].max()),
                "sampling_median_unique": ";".join(str(value) for value in sorted(subset["sampling_median_s"].dropna().unique())),
                "schema_match_03_97_count": int(subset["schema_match_03_97"].sum()),
                "schema_consistent_within_class": bool(subset["column_count"].nunique() == 1 and subset["schema_extra_columns"].nunique() == 1),
                "missing_cell_trajectories": int(subset["missing_cells"].gt(0).sum()),
                "report_count": int(event_subset["report_exists"].sum()),
                "event_parse_success_count": int(event_subset["event_parse_ok"].sum()),
                "injection_parse_success_count": int(event_subset["injection_s"].notna().sum()),
                "scram_parse_success_count": int(event_subset["scram_s"].notna().sum()),
                "first_protection_parse_success_count": int(event_subset["first_protection_s"].notna().sum()),
                **{f"available_{window}s": int(subset[f"available_{window}s"].sum()) for window in WINDOWS_S},
                "train_count": int((subset["split"] == "train").sum()),
                "validation_count": int((subset["split"] == "validation").sum()),
                "test_count": int((subset["split"] == "test").sum()),
                "initial_condition_flagged_feature_count": int(class_initial["class_initial_condition_flag"].sum()),
                "first_version_eligible": bool(count >= 10),
                "class_size_strategy": "three-way grouped split" if count >= 3 else "train-only until more independent trajectories are available",
            }
        )
    class_summary = pd.DataFrame(class_summary_rows)
    strict_process_output = feature_table.loc[feature_table["ml_policy"].eq("candidate")].copy()

    inventory_path = result_root / "09_multi_accident_class_inventory.csv"
    event_path = result_root / "09_multi_accident_event_inventory.csv"
    summary_path = result_root / "09_multi_accident_audit_summary.csv"
    summary_metrics_path = result_root / "09_multi_accident_audit_metrics.csv"
    summary_json_path = result_root / "09_multi_accident_audit_summary.json"
    leakage_path = result_root / "09_multi_accident_leakage_flags.csv"
    initial_path = result_root / "09_multi_accident_initial_conditions.csv"
    strict_path = result_root / "09_multi_accident_strict_process_features.csv"
    inventory.to_csv(inventory_path, index=False)
    events.to_csv(event_path, index=False)
    class_summary.to_csv(summary_path, index=False)
    leakage.to_csv(leakage_path, index=False)
    initial_aggregated.to_csv(initial_path, index=False)
    strict_process_output.to_csv(strict_path, index=False)

    figure_paths = _write_figures(class_summary, initial_aggregated, inventory, result_root)
    available_by_window = {
        str(window): {
            "available_trajectories": int(inventory[f"available_{window}s"].sum()),
            "coverage_rate": float(inventory[f"available_{window}s"].mean()),
            "classes_with_full_coverage": class_summary.loc[
                class_summary[f"available_{window}s"] == class_summary["trajectory_count"], "accident_class"
            ].tolist(),
        }
        for window in WINDOWS_S
    }
    first_version_classes = class_summary.loc[class_summary["first_version_eligible"], "accident_class"].tolist()
    too_small_classes = class_summary.loc[~class_summary["first_version_eligible"], "accident_class"].tolist()
    initial_flagged_features = initial_aggregated.loc[
        initial_aggregated["initial_condition_flag"], "feature"
    ].drop_duplicates().tolist()
    key_initial = initial_aggregated.loc[
        initial_aggregated["feature"].isin(KEY_INITIAL_FEATURES),
        ["feature", "accident_class", "initial_mean", "initial_std", "max_pairwise_mean_diff", "initial_condition_flag"],
    ].to_dict(orient="records")
    split_counts = inventory.groupby(["accident_class", "split"]).size().unstack(fill_value=0).to_dict(orient="index")
    summary = {
        "experiment": "multi-accident dataset audit",
        "normal_excluded": True,
        "accident_class_count": int(len(class_summary)),
        "trajectory_count": int(len(inventory)),
        "total_rows": int(inventory["rows"].sum()),
        "class_trajectory_count_min": int(class_summary["trajectory_count"].min()),
        "class_trajectory_count_max": int(class_summary["trajectory_count"].max()),
        "classes": class_summary.to_dict(orient="records"),
        "strict_process_feature_count": len(strict_features),
        "strict_process_features": strict_features,
        "feature_group_counts": feature_table.groupby(["group", "ml_policy"]).size().to_dict(),
        "event_parse_success": {
            "reports": int(events["report_exists"].sum()),
            "injection": int(events["injection_s"].notna().sum()),
            "scram": int(events["scram_s"].notna().sum()),
            "first_protection": int(events["first_protection_s"].notna().sum()),
            "trajectory_total": len(events),
        },
        "available_window_coverage": available_by_window,
        "split_plan": {
            "unit": "complete trajectory/sample_id",
            "row_random_split_forbidden": True,
            "current_assignment": "deterministic per-class interleaved 80/10/10 group split",
            "small_class_strategy": "classes with fewer than 3 trajectories are train-only in this audit; do not claim held-out generalization for them",
            "counts": split_counts,
        },
        "leakage_audit": {
            "features_checked": len(feature_totals),
            "flag_row_count": int(len(leakage)),
            "potential_label_leakage_row_count": int(leakage["potential_label_leakage"].sum()) if len(leakage) else 0,
            "flagged_features": leakage["feature"].drop_duplicates().tolist() if len(leakage) else [],
            "auto_removed": False,
        },
        "initial_condition_audit": {
            "candidate_features_checked": len(strict_features),
            "flagged_features": initial_flagged_features,
            "confounding_present": bool(initial_flagged_features),
            "key_features": key_initial,
        },
        "recommendation": {
            "first_version_eligible_classes": first_version_classes,
            "defer_or_expand_classes": too_small_classes,
            "recommended_windows": [30, 60],
            "exploratory_window": 120,
            "window_reason": "30/60 s have the broadest early coverage; 120 s should be reported only with per-class availability because some short trajectories end before it.",
            "process_variable_policy": "Use only the 38 candidate variables from results/feature_groups.csv for the strict baseline; keep all flagged variables for explicit review and do not auto-delete them.",
            "metrics": ["Macro-F1", "Balanced Accuracy", "per-class Recall", "confusion matrix"],
            "classification_ready": bool(first_version_classes),
        },
        "figure_paths": figure_paths,
    }
    summary_json_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_flat = pd.DataFrame(
        [
            {"metric": "accident_class_count", "value": len(class_summary)},
            {"metric": "trajectory_count", "value": len(inventory)},
            {"metric": "total_rows", "value": int(inventory["rows"].sum())},
            {"metric": "class_trajectory_count_min", "value": int(class_summary["trajectory_count"].min())},
            {"metric": "class_trajectory_count_max", "value": int(class_summary["trajectory_count"].max())},
            {"metric": "strict_process_feature_count", "value": len(strict_features)},
            {"metric": "event_injection_parse_success", "value": int(events["injection_s"].notna().sum())},
            {"metric": "event_first_protection_parse_success", "value": int(events["first_protection_s"].notna().sum())},
            {"metric": "leakage_flag_row_count", "value": len(leakage)},
            {"metric": "initial_condition_flagged_feature_count", "value": len(initial_flagged_features)},
            {"metric": "first_version_eligible_class_count", "value": len(first_version_classes)},
            {"metric": "recommended_windows_s", "value": "30,60; exploratory=120"},
            {"metric": "split_policy", "value": "trajectory/sample_id grouped; small classes train-only"},
        ]
    )
    summary_flat.to_csv(summary_metrics_path, index=False)

    return {
        "summary": summary,
        "class_inventory": inventory,
        "event_inventory": events,
        "class_summary": class_summary,
        "leakage_flags": leakage,
        "initial_conditions": initial_aggregated,
        "strict_process_features": strict_process_output,
        "output_paths": {
            "class_inventory": _relative_path(inventory_path),
            "event_inventory": _relative_path(event_path),
            "audit_summary": _relative_path(summary_path),
            "audit_metrics": _relative_path(summary_metrics_path),
            "audit_summary_json": _relative_path(summary_json_path),
            "leakage_flags": _relative_path(leakage_path),
            "initial_conditions": _relative_path(initial_path),
            "strict_process_features": _relative_path(strict_path),
        },
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(_json_safe(result["summary"]), indent=2, ensure_ascii=False))
    print("FULL MULTI-ACCIDENT DATASET AUDIT PASSED")
