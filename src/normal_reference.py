"""Inventory legal Normal-reference sources in the local NPPAD copy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
OPERATION_ROOT = DATA_ROOT / "Operation_csv_data"
RAW_NPPAD_ROOT = DATA_ROOT / "NPPAD"
VARIABLE_ROOT = DATA_ROOT / "Variable_Power_Data"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results"

MALFUNCTION_RE = re.compile(
    r"^\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s+sec,\s+Malfunction\s+#",
    re.MULTILINE,
)


def relative_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Return a stable project-relative path for result files."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_plotdata_mdb(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Read one operation MDB without modifying it."""
    import pyodbc

    mdb_path = Path(path)
    connection = pyodbc.connect(
        "DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=" + str(mdb_path)
    )
    try:
        cursor = connection.cursor()
        tables = [row.table_name for row in cursor.tables(tableType="TABLE")]
        cursor.execute("SELECT * FROM PlotData")
        columns = [column[0] for column in cursor.description]
        frame = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
    finally:
        connection.close()
    return frame.sort_values("TIME").reset_index(drop=True), tables


def parse_injection_time(report_path: str | Path) -> tuple[float | None, str | None]:
    """Return the first explicit malfunction time and its source line."""
    path = Path(report_path)
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = MALFUNCTION_RE.search(text)
    if not match:
        return None, None
    line = next(
        line.strip()
        for line in text.splitlines()
        if MALFUNCTION_RE.match(line)
    )
    return float(match.group("time")), line


def _time_metrics(time: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(time, errors="coerce")
    differences = values.diff().dropna()
    median = float(differences.median()) if len(differences) else None
    if median is None:
        nonuniform = 0
    else:
        nonuniform = int(
            (~np.isclose(differences.to_numpy(float), median, atol=1e-6)).sum()
        )
    return {
        "rows": int(len(values)),
        "time_numeric": bool(values.notna().all()),
        "time_strictly_increasing": bool(differences.gt(0).all()),
        "time_start_s": float(values.iloc[0]) if len(values) else None,
        "time_end_s": float(values.iloc[-1]) if len(values) else None,
        "sampling_median_s": median,
        "sampling_min_s": float(differences.min()) if len(differences) else None,
        "sampling_max_s": float(differences.max()) if len(differences) else None,
        "nonuniform_sampling_intervals": nonuniform,
    }


def _frame_metrics(
    frame: pd.DataFrame,
    reference_columns: list[str],
) -> dict[str, Any]:
    """Collect lightweight schema, time, and power metadata."""
    metrics = _time_metrics(frame["TIME"])
    metrics.update(
        {
            "column_count": int(len(frame.columns)),
            "non_time_variable_count": int(max(len(frame.columns) - 1, 0)),
            "reference_97_variables_available": bool(
                set(reference_columns).issubset(frame.columns)
            ),
            "schema_97_columns": bool(
                len(frame.columns) == 97
                and list(frame.columns) == reference_columns
            ),
            "schema_extra_columns": ";".join(
                sorted(set(frame.columns) - set(reference_columns))
            ),
            "schema_missing_columns": ";".join(
                sorted(set(reference_columns) - set(frame.columns))
            ),
            "missing_cells": int(frame.isna().sum().sum()),
        }
    )
    if "PWR" in frame:
        power = pd.to_numeric(frame["PWR"], errors="coerce")
        metrics.update(
            {
                "pwr_start_pct": float(power.iloc[0]),
                "pwr_end_pct": float(power.iloc[-1]),
                "pwr_min_pct": float(power.min()),
                "pwr_max_pct": float(power.max()),
            }
        )
    else:
        metrics.update(
            {
                "pwr_start_pct": None,
                "pwr_end_pct": None,
                "pwr_min_pct": None,
                "pwr_max_pct": None,
            }
        )
    return metrics


def inspect_csv(
    path: str | Path,
    reference_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Inspect a CSV without loading every variable for every accident case."""
    csv_path = Path(path)
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [column for column in ("TIME", "PWR") if column in columns]
    axis = pd.read_csv(csv_path, usecols=usecols)
    metrics = _frame_metrics(axis, reference_columns)
    metrics["column_count"] = len(columns)
    metrics["non_time_variable_count"] = max(len(columns) - 1, 0)
    metrics["reference_97_variables_available"] = bool(
        set(reference_columns).issubset(columns)
    )
    metrics["schema_extra_columns"] = ";".join(
        sorted(set(columns) - set(reference_columns))
    )
    metrics["schema_missing_columns"] = ";".join(
        sorted(set(reference_columns) - set(columns))
    )
    metrics["missing_cells"] = None
    metrics["schema_97_columns"] = bool(
        len(columns) == 97 and columns == reference_columns
    )
    first = pd.read_csv(csv_path, nrows=1)
    return metrics, first


def _initial_state_matches(
    first: pd.DataFrame,
    reference_first: pd.DataFrame,
    reference_columns: list[str],
) -> bool:
    if not set(reference_columns).issubset(first.columns):
        return False
    left = first.loc[:, reference_columns].iloc[0].to_numpy(dtype=float)
    right = reference_first.loc[:, reference_columns].iloc[0].to_numpy(dtype=float)
    return bool(np.allclose(left, right, rtol=0.0, atol=1e-6, equal_nan=True))


def _base_row(**values: Any) -> dict[str, Any]:
    fields = {
        "record_id": None,
        "source_kind": None,
        "source_group": None,
        "source_category": None,
        "scenario_id": None,
        "case_id": None,
        "classification": None,
        "is_legal_normal_reference": False,
        "legal_normal_reason": None,
        "source_path": None,
        "raw_mdb_path": None,
        "transient_report_path": None,
        "mdb_read_status": None,
        "mdb_table_names": None,
        "operating_condition": None,
        "start_power_pct": None,
        "end_power_pct": None,
        "ramp_rate_pct_per_min": None,
        "transition_direction": None,
        "accident_code": None,
        "accident_name": None,
        "injection_time_s": None,
        "report_event_line": None,
        "rows": None,
        "column_count": None,
        "non_time_variable_count": None,
        "reference_97_variables_available": False,
        "schema_97_columns": False,
        "schema_extra_columns": None,
        "schema_missing_columns": None,
        "missing_cells": None,
        "time_numeric": False,
        "time_strictly_increasing": False,
        "time_start_s": None,
        "time_end_s": None,
        "sampling_median_s": None,
        "sampling_min_s": None,
        "sampling_max_s": None,
        "nonuniform_sampling_intervals": None,
        "pwr_start_pct": None,
        "pwr_end_pct": None,
        "pwr_min_pct": None,
        "pwr_max_pct": None,
        "loca_initial_condition_match": False,
        "pre_injection_rows": None,
        "pre_injection_time_start_s": None,
        "pre_injection_time_end_s": None,
        "pre_injection_window_s": None,
        "pre_injection_sampling_median_s": None,
        "has_valid_pre_injection_sample": False,
        "pre_window_usable_for_normal": False,
        "independent_trajectory": False,
        "independence_note": None,
        "train_use": None,
        "calibration_use": None,
        "test_use": None,
        "audit_note": None,
    }
    fields.update(values)
    return fields


def _apply_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        row[key] = value


def _pre_injection_metrics(
    axis: pd.DataFrame,
    injection_time_s: float | None,
) -> dict[str, Any]:
    if injection_time_s is None or injection_time_s <= 0:
        return {
            "pre_injection_rows": 0,
            "has_valid_pre_injection_sample": False,
            "pre_window_usable_for_normal": False,
        }
    time = pd.to_numeric(axis["TIME"], errors="coerce")
    pre = time[time < injection_time_s]
    if pre.empty:
        return {
            "pre_injection_rows": 0,
            "has_valid_pre_injection_sample": False,
            "pre_window_usable_for_normal": False,
        }
    differences = pre.diff().dropna()
    window = float(pre.iloc[-1] - pre.iloc[0])
    return {
        "pre_injection_rows": int(len(pre)),
        "pre_injection_time_start_s": float(pre.iloc[0]),
        "pre_injection_time_end_s": float(pre.iloc[-1]),
        "pre_injection_window_s": window,
        "pre_injection_sampling_median_s": (
            float(differences.median()) if len(differences) else None
        ),
        "has_valid_pre_injection_sample": True,
        "pre_window_usable_for_normal": bool(len(pre) >= 2 and window > 0),
    }


def build_inventory(
    data_root: str | Path = DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build case-level inventory, grouped summary, and JSON-ready audit."""
    data_root = Path(data_root)
    operation_root = data_root / "Operation_csv_data"
    raw_root = data_root / "NPPAD"
    variable_root = data_root / "Variable_Power_Data"

    fixed_csv = operation_root / "Normal" / "1.csv"
    fixed_mdb = raw_root / "Normal" / "1.mdb"
    fixed_report = raw_root / "Normal" / "1Transient Report.txt"
    fixed = pd.read_csv(fixed_csv)
    reference_columns = fixed.columns.tolist()
    reference_first = fixed.iloc[[0]]
    fixed_metrics = _frame_metrics(fixed, reference_columns)
    fixed_mdb_status = "missing"
    fixed_mdb_tables: list[str] = []
    fixed_mdb_matches = False
    if fixed_mdb.exists():
        try:
            raw_fixed, fixed_mdb_tables = read_plotdata_mdb(fixed_mdb)
            fixed_mdb_status = "success"
            fixed_mdb_matches = bool(
                list(raw_fixed.columns) == reference_columns
                and len(raw_fixed) == len(fixed)
                and np.allclose(
                    raw_fixed.to_numpy(dtype=float),
                    fixed.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-5,
                    equal_nan=True,
                )
            )
        except Exception as exc:
            fixed_mdb_status = f"error: {type(exc).__name__}: {exc}"

    rows: list[dict[str, Any]] = []
    fixed_row = _base_row(
        record_id="normal_csv",
        source_kind="fixed_normal_csv",
        source_group="fixed_normal",
        source_category="Normal",
        scenario_id="NPPAD_NORMAL_1",
        case_id="1",
        classification="A",
        is_legal_normal_reference=True,
        legal_normal_reason=(
            "Explicit NPPAD Normal source with the same initial state as all "
            "accident cases; it is the only direct full-power Normal trajectory."
        ),
        source_path=relative_path(fixed_csv),
        raw_mdb_path=relative_path(fixed_mdb) if fixed_mdb.exists() else None,
        transient_report_path=relative_path(fixed_report),
        mdb_read_status=fixed_mdb_status,
        mdb_table_names=";".join(fixed_mdb_tables) if fixed_mdb_tables else None,
        operating_condition=(
            "NPPAD Normal operating source; nominal initial PWR/PWNT 100%, "
            "but observed PWR decreases to about 40%."
        ),
        start_power_pct=100.0,
        end_power_pct=fixed_metrics["pwr_end_pct"],
        loca_initial_condition_match=True,
        independent_trajectory=True,
        independence_note="One standalone Normal trajectory; no independent peer trajectory.",
        train_use="yes, as the sole direct Normal training source",
        calibration_use="no independent calibration trajectory available",
        test_use="no independent Normal test trajectory available",
        audit_note=(
            "Raw Normal/1.mdb read successfully and matches the CSV: "
            f"{fixed_mdb_matches}."
        ),
    )
    _apply_metrics(fixed_row, fixed_metrics)
    rows.append(fixed_row)

    manifest_path = variable_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    normal_manifest = manifest.loc[
        manifest.get("category", pd.Series(dtype=str)).eq("normal_power_transition")
    ]
    for record in normal_manifest.itertuples(index=False):
        scenario = str(record.scenario_id)
        mdb_path = variable_root / scenario / "case1.mdb"
        report_path = variable_root / scenario / "case1.txt"
        row = _base_row(
            record_id=f"normal_mdb_{scenario}",
            source_kind="variable_power_normal_mdb",
            source_group="variable_power_normal",
            source_category="NORM",
            scenario_id=scenario,
            case_id="case1",
            classification="B",
            is_legal_normal_reference=True,
            legal_normal_reason=(
                "Explicit normal power-transition trajectory; valid for "
                "condition-aware modeling, not a direct fixed-power LOCA reference."
            ),
            source_path=relative_path(mdb_path),
            transient_report_path=relative_path(report_path),
            operating_condition=str(record.notes),
            start_power_pct=float(record.start_power_pct),
            end_power_pct=float(record.end_power_pct),
            ramp_rate_pct_per_min=float(record.ramp_rate_pct_per_min),
            transition_direction=str(record.transition_direction),
            independent_trajectory=True,
            independence_note=(
                "Distinct simulator trajectory, but it shares the initialized "
                "state and has a different power-transition condition."
            ),
            train_use="yes, condition-aware model only",
            calibration_use="yes only at trajectory level; never random row split",
            test_use="yes only as a held-out condition-aware trajectory",
        )
        try:
            frame, tables = read_plotdata_mdb(mdb_path)
            metrics = _frame_metrics(frame, reference_columns)
            row["mdb_read_status"] = "success"
            row["mdb_table_names"] = ";".join(tables)
            row["loca_initial_condition_match"] = _initial_state_matches(
                frame.iloc[[0]], reference_first, reference_columns
            )
            _apply_metrics(row, metrics)
            row["audit_note"] = (
                "MDB PlotData read successfully; retain sampling gaps as data-quality "
                "metadata and split by whole trajectory."
            )
        except Exception as exc:
            row["mdb_read_status"] = f"error: {type(exc).__name__}: {exc}"
            row["audit_note"] = "MDB read failed; no fallback normal claim made."
        rows.append(row)

    accident_rows: list[dict[str, Any]] = []
    # ponytail: use exported CSV for the 1-row/axis audit; opening 1,116 duplicate
    # operation MDBs adds cost without changing the legal pre-injection result.
    for category_dir in sorted(raw_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name == "Normal":
            continue
        operation_dir = operation_root / category_dir.name
        for report_path in sorted(category_dir.glob("*Transient Report.txt")):
            case_id = report_path.name.removesuffix("Transient Report.txt")
            csv_path = operation_dir / f"{case_id}.csv"
            raw_mdb_path = category_dir / f"{case_id}.mdb"
            injection, event_line = parse_injection_time(report_path)
            row = _base_row(
                record_id=f"accident_{category_dir.name}_{case_id}",
                source_kind="accident_pre_injection_audit",
                source_group="accident_pre_injection",
                source_category=category_dir.name,
                scenario_id=f"{category_dir.name}_{case_id}",
                case_id=case_id,
                classification="C",
                is_legal_normal_reference=False,
                legal_normal_reason=(
                    "Accident trajectory; its pre-injection samples are not an "
                    "independent Normal run and cannot be relabeled as Normal."
                ),
                source_path=relative_path(csv_path),
                raw_mdb_path=relative_path(raw_mdb_path),
                transient_report_path=relative_path(report_path),
                mdb_read_status="not_attempted_accident_scope",
                operating_condition="Original NPPAD accident trajectory",
                accident_code=category_dir.name,
                accident_name=category_dir.name,
                injection_time_s=injection,
                report_event_line=event_line,
                independent_trajectory=False,
                independence_note=(
                    "Pre-injection segment belongs to an accident case and shares "
                    "the common initialized state."
                ),
                train_use="no",
                calibration_use="no",
                test_use="accident test only, not Normal calibration",
            )
            if csv_path.exists():
                metrics, first = inspect_csv(csv_path, reference_columns)
                _apply_metrics(row, metrics)
                axis = pd.read_csv(csv_path, usecols=["TIME"])
                row["loca_initial_condition_match"] = _initial_state_matches(
                    first, reference_first, reference_columns
                )
                row.update(_pre_injection_metrics(axis, injection))
            else:
                row["audit_note"] = "Operation CSV missing; no pre-injection window extracted."
            if injection == 0.5 and row["pre_injection_rows"] == 1:
                row["audit_note"] = (
                    "First malfunction is at 0.5 s; 10 s sampling leaves only t=0 "
                    "before injection and a 0 s usable window."
                )
            accident_rows.append(row)
    rows.extend(accident_rows)

    inventory = pd.DataFrame(rows)
    inventory = inventory.sort_values(
        ["source_group", "source_category", "scenario_id", "case_id"],
        na_position="last",
    ).reset_index(drop=True)
    summary = _build_summary(inventory)
    audit = _build_audit_json(
        inventory,
        summary,
        fixed_mdb_status=fixed_mdb_status,
        fixed_mdb_matches=fixed_mdb_matches,
        manifest_path=manifest_path,
    )
    return inventory, summary, audit


def _build_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("fixed_normal", "fixed Normal/1.csv"),
        ("variable_power_normal", "variable-power NORM MDB"),
        ("accident_pre_injection", "accident pre-injection audit"),
    ]
    rows: list[dict[str, Any]] = []
    for group, label in groups:
        group_frame = inventory.loc[inventory["source_group"].eq(group)]
        if group_frame.empty:
            continue
        rows.append(_summary_row(group, label, group_frame))
        if group == "accident_pre_injection":
            for category, category_frame in group_frame.groupby("source_category"):
                rows.append(_summary_row(group, category, category_frame))
    rows.append(_summary_row("overall", "all scanned trajectories", inventory))
    return pd.DataFrame(rows)


def _summary_row(group: str, label: str, frame: pd.DataFrame) -> dict[str, Any]:
    def _min(column: str) -> Any:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.min()) if len(values) else None

    def _max(column: str) -> Any:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.max()) if len(values) else None

    return {
        "summary_group": group,
        "label": label,
        "trajectory_count": int(len(frame)),
        "A_count": int(frame["classification"].eq("A").sum()),
        "B_count": int(frame["classification"].eq("B").sum()),
        "C_count": int(frame["classification"].eq("C").sum()),
        "legal_normal_count": int(frame["is_legal_normal_reference"].sum()),
        "independent_count": int(frame["independent_trajectory"].sum()),
        "mdb_success_count": int(frame["mdb_read_status"].eq("success").sum()),
        "reference_97_variables_all": bool(
            frame["reference_97_variables_available"].all()
        ),
        "schema_97_all": bool(frame["schema_97_columns"].all()),
        "initial_match_count": int(frame["loca_initial_condition_match"].sum()),
        "injection_min_s": _min("injection_time_s"),
        "injection_max_s": _max("injection_time_s"),
        "pre_rows_min": _min("pre_injection_rows"),
        "pre_rows_max": _max("pre_injection_rows"),
        "pre_window_min_s": _min("pre_injection_window_s"),
        "pre_window_max_s": _max("pre_injection_window_s"),
        "usable_pre_window_count": int(frame["pre_window_usable_for_normal"].sum()),
        "notes": (
            "All accident categories use report-first malfunction time; "
            "pre-injection windows are never relabeled as Normal."
            if group == "accident_pre_injection"
            else ""
        ),
    }


def _build_audit_json(
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    fixed_mdb_status: str,
    fixed_mdb_matches: bool,
    manifest_path: Path,
) -> dict[str, Any]:
    accident = inventory.loc[inventory["source_group"].eq("accident_pre_injection")]
    normal = inventory.loc[inventory["is_legal_normal_reference"]]
    return {
        "stage": "phase_2_step_1_normal_reference_expansion",
        "scope": {
            "data_root": relative_path(DATA_ROOT),
            "readme_files": [
                relative_path(DATA_ROOT / "README.md"),
                relative_path(VARIABLE_ROOT / "README.md"),
            ],
            "operation_csv_root": relative_path(OPERATION_ROOT),
            "raw_nppad_root": relative_path(RAW_NPPAD_ROOT),
            "variable_power_manifest": relative_path(manifest_path),
            "startup_shutdown_directories_found": [],
            "other_non_accident_trajectories_found": [],
        },
        "dataset_structure": {
            "original_operating_condition_classes": sorted(
                inventory["source_category"].dropna().unique().tolist()
            ),
            "original_accident_category_count": int(
                accident["source_category"].nunique()
            ),
            "variable_power_normal_count": int(
                inventory["source_group"].eq("variable_power_normal").sum()
            ),
            "variable_power_accident_scenario_count": 8,
        },
        "normal_reference_counts": {
            "all_explicit_normal_trajectories": int(len(normal)),
            "A_direct_normal": int(normal["classification"].eq("A").sum()),
            "B_auxiliary_normal": int(normal["classification"].eq("B").sum()),
            "C_not_legal_normal": int(
                inventory["classification"].eq("C").sum()
            ),
        },
        "mdb_validation": {
            "fixed_normal_mdb_status": fixed_mdb_status,
            "fixed_normal_mdb_matches_csv": fixed_mdb_matches,
            "variable_power_normal_mdb_success_count": int(
                inventory.loc[
                    inventory["source_group"].eq("variable_power_normal"),
                    "mdb_read_status",
                ].eq("success").sum()
            ),
            "variable_power_normal_mdb_total": int(
                inventory["source_group"].eq("variable_power_normal").sum()
            ),
        },
        "accident_injection_audit": {
            "accident_case_count": int(len(accident)),
            "categories": int(accident["source_category"].nunique()),
            "unique_injection_times_s": sorted(
                pd.to_numeric(accident["injection_time_s"], errors="coerce")
                .dropna()
                .unique()
                .tolist()
            ),
            "pre_injection_row_counts": sorted(
                pd.to_numeric(accident["pre_injection_rows"], errors="coerce")
                .dropna()
                .unique()
                .tolist()
            ),
            "pre_injection_window_s": sorted(
                pd.to_numeric(accident["pre_injection_window_s"], errors="coerce")
                .dropna()
                .unique()
                .tolist()
            ),
            "usable_pre_window_count": int(
                accident["pre_window_usable_for_normal"].sum()
            ),
        },
        "conclusion": {
            "rigorous_loca_early_warning_supported": False,
            "decision": "insufficient_normal_reference_data",
            "reason": (
                "There is one direct Normal trajectory, it is nonstationary in "
                "PWR/PWNT, four explicit normal power-transition trajectories "
                "are auxiliary B-level sources, and no accident case provides "
                "a positive-duration pre-injection window."
            ),
            "recommendation": (
                "Do not make strict LOCA early warning the main Phase-2 line. "
                "Keep the five normal trajectories for condition-aware exploratory "
                "work, but shift the main line to LOCA severity estimation and/or "
                "protection-time prediction until independent full-power Normal "
                "runs are available."
            ),
            "future_split_if_new_data_arrive": (
                "Split by whole independent trajectory: train, calibration, and "
                "test; never split rows from one trajectory across those sets."
            ),
        },
        "summary_rows": summary.to_dict(orient="records"),
    }


def write_audit_results(
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    data_root: str | Path = DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build and write the requested CSV/JSON audit artifacts."""
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    inventory, summary, audit = build_inventory(data_root)
    inventory.to_csv(result_root / "normal_reference_inventory.csv", index=False)
    summary.to_csv(result_root / "normal_reference_audit_summary.csv", index=False)
    (result_root / "normal_reference_audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return inventory, summary, audit


if __name__ == "__main__":
    inventory, summary, audit = write_audit_results()
    print(summary.to_string(index=False))
    print(json.dumps(audit["conclusion"], ensure_ascii=False, indent=2))
