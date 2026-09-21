"""Pure dashboard data preparation and export helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LIMITATIONS = [
    "Research prototype; not for safety-critical deployment.",
    "Explainability is model attribution and diagnostic assistance, not physical causality.",
    "Family-level diagnosis remains affected by severity out-of-distribution shift.",
    "Subtype OOD is unresolved; conformal empirical coverage has no 90% guarantee.",
    "LOCA severity is exploratory and simulation-specific.",
    "Protection-time prediction is not available in v1.",
]
FALLBACK_TREND_VARIABLES = ("P", "WSTA", "TAVG", "WFWA", "LSGA")
STATUS_LABELS = {
    "accepted": "Research model accepted",
    "requires_review": "Human review required",
    "unknown": "Insufficient model evidence",
    "invalid_input": "Input rejected",
}


def _column_values(frame: pd.DataFrame, name: str) -> np.ndarray | None:
    matches = frame.loc[:, frame.columns == name]
    if matches.shape[1] != 1:
        return None
    return pd.to_numeric(matches.iloc[:, 0], errors="coerce").to_numpy(dtype=float)


def inspect_input(frame: pd.DataFrame, required_features: list[str], minimum_window_s: float = 120.0) -> dict[str, Any]:
    """Validate display-facing input quality before model inference."""
    columns = [str(column) for column in frame.columns]
    missing = [feature for feature in required_features if feature not in columns]
    extra = sorted(set(columns) - {"TIME", *required_features})
    errors: list[str] = []
    if missing:
        errors.append(f"missing_required_columns:{missing}")
    if frame.columns.duplicated().any():
        errors.append("duplicate_column_names")

    quality: dict[str, Any] = {
        "row_count": int(len(frame)),
        "time_present": "TIME" in columns,
        "required_variable_count": int(len(required_features) - len(missing)),
        "required_variable_total": int(len(required_features)),
        "missing_columns": missing,
        "extra_columns": extra,
        "nan_inf_columns": [],
        "repeated_time_points": 0,
        "sampling_interval_median_s": None,
        "sampling_interval_max_deviation_s": None,
        "time_start_s": None,
        "time_end_s": None,
        "window_s": None,
        "coverage_ge_120s": False,
    }
    names = ["TIME", *required_features]
    for name in names:
        values = _column_values(frame, name)
        if values is None:
            continue
        if not np.isfinite(values).all():
            quality["nan_inf_columns"].append(name)
            if name == "TIME":
                errors.append("TIME_contains_NaN_or_Inf")
            else:
                errors.append(f"{name}_contains_NaN_or_Inf")

    time = _column_values(frame, "TIME")
    if time is not None and np.isfinite(time).all():
        quality["time_start_s"] = float(time[0]) if len(time) else None
        quality["time_end_s"] = float(time[-1]) if len(time) else None
        if len(time) >= 2:
            diffs = np.diff(time)
            quality["repeated_time_points"] = int(np.sum(diffs == 0))
            quality["sampling_interval_median_s"] = float(np.median(diffs))
            quality["sampling_interval_max_deviation_s"] = float(np.max(np.abs(diffs - 10.0)))
            if np.any(diffs <= 0):
                errors.append("TIME_not_strictly_increasing_or_duplicate_time_points")
            if np.any(np.abs(diffs - 10.0) > 1.0):
                errors.append("sampling_interval_not_approximately_10_s")
        if len(time):
            quality["window_s"] = float(np.max(time) - np.min(time))
            quality["coverage_ge_120s"] = bool(quality["window_s"] >= minimum_window_s)
            if not quality["coverage_ge_120s"]:
                errors.append("window_shorter_than_120_s")
            if not np.any(time <= 0.5):
                errors.append("baseline_at_or_before_injection_missing")
            if not np.any((time > 0.5) & (time <= 120.5)):
                errors.append("no_samples_in_120_s_window")
    elif "TIME" not in columns:
        errors.append("missing_required_columns:['TIME']")

    quality["errors"] = list(dict.fromkeys(errors))
    quality["valid"] = not quality["errors"]
    return quality


def invalid_response(quality: dict[str, Any], model_version: str = "unknown") -> dict[str, Any]:
    return {
        "status": "invalid_input",
        "accident_family": None,
        "family_top1": None,
        "family_confidence": None,
        "conformal_set": [],
        "ood_warning": {"flag": None, "reason": "No prediction was attempted."},
        "distance_score": None,
        "assessment": {},
        "severity_estimate": None,
        "severity_status": "not_run_due_to_invalid_input",
        "data_quality": quality,
        "limitations": DEFAULT_LIMITATIONS,
        "model_version": model_version,
    }


def prepare_view_model(
    frame: pd.DataFrame,
    response: dict[str, Any],
    explanation: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert frozen inference plus optional 18 explanation into UI data."""
    status = str(response.get("status", "invalid_input"))
    explanation = explanation or {}
    quality = quality or response.get("data_quality", {})
    assessment = response.get("assessment", {})
    capability = assessment.get("family_capability", {})
    family = response.get("accident_family")
    top1 = response.get("family_top1")
    variables = [
        str(item.get("variable"))
        for item in explanation.get("top_variables", [])
        if item.get("variable") in frame.columns
    ]
    variables = list(dict.fromkeys(variables))[:5]
    fallback = not variables
    if fallback:
        variables = [name for name in FALLBACK_TREND_VARIABLES if name in frame.columns][:5]

    loca = response.get("severity_status") == "available_exploratory" and status == "accepted" and family == "primary_coolant_boundary_break" and capability.get("tier") == "Tier A"
    return {
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "input_quality": quality,
        "diagnostic": {
            "accident_family": family,
            "top1_candidate": top1,
            "family_confidence": response.get("family_confidence"),
            "conformal_set": response.get("conformal_set", []),
            "ood_warning": response.get("ood_warning", {}),
            "distance_score": response.get("distance_score"),
            "family_tier": capability.get("tier"),
            "model_version": response.get("model_version"),
        },
        "uncertainty": explanation.get("uncertainty"),
        "explanation": {
            "available": bool(explanation) and status != "invalid_input",
            "top_variables": explanation.get("top_variables", []),
            "top_features": explanation.get("top_features", []),
            "ood_drivers": (explanation.get("ood_drivers") or {}).get("top_variables", []),
            "severity_drivers": (explanation.get("severity_drivers") or {}).get("top_variables", []),
            "limitations": explanation.get("limitations", DEFAULT_LIMITATIONS),
        },
        "trends": {"variables": variables, "fallback": fallback},
        "loca_assessment": {
            "run": bool(loca),
            "status": response.get("severity_status", "not_run_due_to_family_gating"),
            "estimate": response.get("severity_estimate") if loca else None,
            "details": assessment.get("loca_severity", {}),
        },
        "protection_time": assessment.get("protection_time", {"status": "not_available_in_v1"}),
        "limitations": response.get("limitations", DEFAULT_LIMITATIONS),
    }


def build_export(
    view_model: dict[str, Any],
    response: dict[str, Any],
    explanation: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    quality = view_model.get("input_quality", {})
    return {
        "schema_version": "npp_guard_dashboard_export_v1",
        "input": {
            "source": source,
            "row_count": quality.get("row_count"),
            "time_start_s": quality.get("time_start_s"),
            "time_end_s": quality.get("time_end_s"),
            "window_s": quality.get("window_s"),
            "extra_columns": quality.get("extra_columns", []),
        },
        "diagnosis": response,
        "explanation": explanation or {},
        "limitations": view_model.get("limitations", DEFAULT_LIMITATIONS),
    }


def export_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value))
