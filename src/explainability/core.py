"""Reproducible explanations that consume, but never fit, v1 artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.inference.parity import evaluate_pipeline_parity
from src.inference.v1 import (
    ARTIFACT_ROOT,
    INJECTION_S,
    WINDOW_S,
    _aligned_probabilities,
    _distance_values,
    _load_artifacts_cached,
    diagnose_dataframe,
    run_regression_tests,
    verify_artifact_manifest,
)
from src.multi_accident_features import strict_process_features
from src.severity_features import build_severity_features, extract_window_features

from .feature_mapping import (
    SUMMARY_STATISTICS,
    aggregate_variables,
    assert_allowed_features,
    split_feature,
    variable_for_feature,
)
from .stability import jaccard, rank_correlation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results"
FIGURE_ROOT = RESULT_ROOT / "figures"
RANDOM_SEED = 20260921
TOP_ENGINEERED = 10
TOP_VARIABLES = 5
TOP_K = 5
LIMITATIONS = [
    "Feature contribution is model attribution, not physical causality.",
    "Family local scores explain the frozen base classifier around the calibrated output; the sigmoid calibration layer is not directly attributed.",
    "OOD drivers are a diagonal approximation to class-conditional Ledoit-Wolf Mahalanobis distance and ignore cross terms.",
    "LOCA severity is exploratory and simulation-specific; historical validation MAE is reported with the frozen artifact.",
    "Research prototype; not for safety-critical deployment.",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _load_context(project_root: str | Path = PROJECT_ROOT, artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Any]:
    root = Path(project_root)
    artifact_path = Path(artifact_root)
    manifest_check = verify_artifact_manifest(artifact_path)
    if not manifest_check["all_passed"]:
        raise AssertionError("Frozen v1 artifact SHA-256 verification failed")
    artifacts = _load_artifacts_cached(str(artifact_path.resolve()))
    schema = artifacts["feature_schema"]
    features = list(schema["strict_process_features"])
    columns = list(schema["derived_feature_columns"])
    assert features == strict_process_features()
    assert len(features) == 38
    assert columns == [f"{feature}__{statistic}" for feature in features for statistic in SUMMARY_STATISTICS]
    assert_allowed_features(columns)
    scaler = artifacts["distance"]["scaler"]
    reference = pd.Series(np.asarray(scaler.mean_, dtype=float), index=columns)
    records = pd.DataFrame(artifacts["manifest"].get("protocol", {}).get("records", []))
    if records.empty:
        records = pd.DataFrame(json.loads((artifact_path / "release_split_manifest.json").read_text(encoding="utf-8"))["records"])
    release = records.loc[records["partition"].eq("locked_release_test")].copy()
    if len(release) != 101 or release["sample_id"].duplicated().any():
        raise AssertionError("18 requires the locked 101-trajectory release cohort")
    return {
        "root": root,
        "artifact_root": artifact_path,
        "artifacts": artifacts,
        "manifest_check": manifest_check,
        "features": features,
        "columns": columns,
        "reference": reference,
        "release": release.sort_values("sample_id").reset_index(drop=True),
        "severity_reference": None,
        "global_feature_rows": [],
    }


def _frame_for_record(context: dict[str, Any], record: Any) -> pd.DataFrame:
    path = context["root"] / Path(str(record.operation_csv))
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _vector(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    extracted = extract_window_features(frame, context["features"], WINDOW_S, INJECTION_S)
    vector = pd.DataFrame([extracted])[context["columns"]]
    assert vector.shape == (1, len(context["columns"]))
    return vector


def _scores_batch(vectors: pd.DataFrame, context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    artifacts = context["artifacts"]
    labels = list(artifacts["conformal"]["labels"])
    base = _aligned_probabilities(artifacts["base_model"], artifacts["base_model"].predict_proba(vectors), labels)
    calibrated = _aligned_probabilities(artifacts["calibrator"], artifacts["calibrator"].predict_proba(vectors), labels)
    return base, calibrated


def _scores(vector: pd.DataFrame, context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    base, calibrated = _scores_batch(vector, context)
    return base[0], calibrated[0]


def _family_local(vector: pd.DataFrame, family: str, context: dict[str, Any]) -> dict[str, Any] | None:
    if not family:
        return None
    labels = list(context["artifacts"]["conformal"]["labels"])
    if family not in labels:
        return None
    original, _ = _scores(vector, context)
    family_index = labels.index(family)
    ablated_matrix = pd.DataFrame(np.repeat(vector.to_numpy(), len(context["columns"]), axis=0), columns=context["columns"])
    for index, feature in enumerate(context["columns"]):
        ablated_matrix.at[index, feature] = context["reference"].loc[feature]
    replaced_matrix, _ = _scores_batch(ablated_matrix, context)
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(context["columns"]):
        contribution = float(original[family_index] - replaced_matrix[index, family_index])
        base, statistic = split_feature(feature)
        rows.append({
            "feature": feature,
            "variable": base,
            "statistic": statistic,
            "contribution": contribution,
            "abs_contribution": abs(contribution),
            "direction": "supports_top1" if contribution > 0 else "opposes_top1" if contribution < 0 else "neutral",
            "reference_value": float(context["reference"].loc[feature]),
        })
    rows.sort(key=lambda item: (-item["abs_contribution"], item["feature"]))
    variables = aggregate_variables(rows)
    for item in variables:
        item["direction"] = "supports_top1" if item["contribution"] > 0 else "opposes_top1" if item["contribution"] < 0 else "neutral"
    assert_allowed_features([item["feature"] for item in rows])
    return {
        "method": "leave_one_engineered_feature_out_to_frozen_training_center",
        "score_layer": "base_classifier_probability",
        "target_family": family,
        "original_target_score": float(original[family_index]),
        "top_features": rows[:TOP_ENGINEERED],
        "top_variables": variables[:TOP_VARIABLES],
        "all_engineered": rows,
        "all_variables": variables,
    }


def _ood_explanation(vector: pd.DataFrame, family: str, distance_score: float | None, context: dict[str, Any]) -> dict[str, Any] | None:
    if not family or family not in context["artifacts"]["distance"]["class_models"]:
        return None
    distance = context["artifacts"]["distance"]
    scaled = distance["scaler"].transform(vector)[0]
    mean, precision = distance["class_models"][family]
    delta = np.asarray(scaled, dtype=float) - np.asarray(mean, dtype=float)
    diagonal = np.maximum(0.0, delta * delta * np.diag(np.asarray(precision, dtype=float)))
    diagonal_total = float(diagonal.sum())
    full_distance = float(np.einsum("i,ij,j->", delta, precision, delta))
    rows = []
    for index, feature in enumerate(context["columns"]):
        base, statistic = split_feature(feature)
        rows.append({
            "feature": feature,
            "variable": base,
            "statistic": statistic,
            "approx_contribution": float(diagonal[index]),
            "fraction_of_diagonal": float(diagonal[index] / diagonal_total) if diagonal_total else 0.0,
            "standardized_displacement": float(delta[index]),
            "direction": "above_class_center" if delta[index] > 0 else "below_class_center" if delta[index] < 0 else "at_class_center",
        })
    rows.sort(key=lambda item: (-item["approx_contribution"], item["feature"]))
    variables: dict[str, dict[str, float]] = {}
    for row in rows:
        item = variables.setdefault(row["variable"], {"variable": row["variable"], "approx_contribution": 0.0, "weighted_displacement": 0.0})
        item["approx_contribution"] += row["approx_contribution"]
        item["weighted_displacement"] += row["approx_contribution"] * row["standardized_displacement"]
    variable_rows = sorted(variables.values(), key=lambda item: (-item["approx_contribution"], item["variable"]))
    for item in variable_rows:
        item["direction"] = "above_class_center" if item["weighted_displacement"] > 0 else "below_class_center" if item["weighted_displacement"] < 0 else "at_class_center"
    assert_allowed_features([item["feature"] for item in rows])
    threshold = float(context["artifacts"]["policy"]["distance_warning_threshold"])
    return {
        "method": "diagonal_standardized_ledoit_wolf_mahalanobis_approximation",
        "predicted_family": family,
        "distance_score": distance_score,
        "recomputed_distance_score": full_distance,
        "distance_warning_threshold": threshold,
        "distance_warning": bool(distance_score is not None and distance_score > threshold),
        "cross_term_residual": float(full_distance - diagonal_total),
        "top_features": rows[:TOP_ENGINEERED],
        "top_variables": variable_rows[:TOP_VARIABLES],
        "limitations": ["Diagonal contributions quantify an approximate distance driver; Mahalanobis cross terms are not assigned to individual variables."],
    }


def _uncertainty_summary(response: dict[str, Any], calibrated: np.ndarray, context: dict[str, Any]) -> dict[str, Any]:
    labels = list(context["artifacts"]["conformal"]["labels"])
    order = np.argsort(-calibrated)
    top = [{"family": labels[int(index)], "probability": float(calibrated[index])} for index in order[:2]]
    cutoff = float(context["artifacts"]["conformal"]["probability_cutoff"])
    maximum = float(calibrated[order[0]]) if len(order) else float("nan")
    return {
        "maximum_calibrated_probability": maximum,
        "top2": top,
        "top2_margin": float(top[0]["probability"] - top[1]["probability"]) if len(top) > 1 else maximum,
        "conformal_set": list(response.get("conformal_set", [])),
        "conformal_probability_cutoff": cutoff,
        "nonconformity_of_top1": float(1.0 - maximum),
        "nonconformity_quantile": float(context["artifacts"]["conformal"]["nonconformity_score_quantile"]),
        "interpretation": "No unique family is supported by the calibrated conformal threshold." if response.get("status") == "unknown" else "Calibrated probability and conformal membership summary.",
    }


def _severity_training_reference(context: dict[str, Any]) -> pd.Series:
    if context["severity_reference"] is not None:
        return context["severity_reference"]
    info = context["artifacts"]["loca_severity"]
    loca_dir = context["root"] / "data" / "NuclearPowerPlantAccidentData" / "Operation_csv_data" / "LOCA"
    dataset, _ = build_severity_features(loca_dir, windows_s=(WINDOW_S,), features=context["features"])
    dataset["protocol_sample_id"] = dataset["sample_id"].str.replace(r"^LOCA_0+", "LOCA_", regex=True)
    train = dataset.loc[dataset["protocol_sample_id"].isin(info["train_sample_ids"])]
    if train.empty:
        raise AssertionError("Frozen LOCA severity train references could not be reconstructed")
    reference = train[info["columns"]].median()
    assert_allowed_features(info["columns"], severity=True)
    context["severity_reference"] = reference
    return reference


def _severity_local(vector: pd.DataFrame, response: dict[str, Any], context: dict[str, Any]) -> dict[str, Any] | None:
    if not (
        response.get("status") == "accepted"
        and response.get("family_top1") == "primary_coolant_boundary_break"
        and response.get("assessment", {}).get("family_capability", {}).get("tier") == "Tier A"
        and response.get("severity_status") == "available_exploratory"
    ):
        return None
    info = context["artifacts"]["loca_severity"]
    model = info["model"]
    columns = list(info["columns"])
    assert_allowed_features(columns, severity=True)
    vector = vector[columns]
    reference = _severity_training_reference(context)
    original = float(model.predict(vector)[0])
    ablated_matrix = pd.DataFrame(np.repeat(vector.to_numpy(), len(columns), axis=0), columns=columns)
    for index, feature in enumerate(columns):
        ablated_matrix.at[index, feature] = float(reference.loc[feature])
    replaced_matrix = model.predict(ablated_matrix)
    rows: list[dict[str, Any]] = []
    importances = np.asarray(getattr(model, "feature_importances_", np.zeros(len(columns))), dtype=float)
    for index, feature in enumerate(columns):
        contribution = original - float(replaced_matrix[index])
        base, statistic = split_feature(feature)
        rows.append({
            "feature": feature,
            "variable": base,
            "statistic": statistic,
            "contribution": float(contribution),
            "abs_contribution": float(abs(contribution)),
            "direction": "raises_estimate" if contribution > 0 else "lowers_estimate" if contribution < 0 else "neutral",
            "global_model_importance": float(importances[index]) if len(importances) == len(columns) else 0.0,
        })
    rows.sort(key=lambda item: (-item["abs_contribution"], item["feature"]))
    variables = aggregate_variables(rows)
    for item in variables:
        item["direction"] = "raises_estimate" if item["contribution"] > 0 else "lowers_estimate" if item["contribution"] < 0 else "neutral"
    historical_test_mae = None
    historical_path = context["root"] / "results" / "07_severity_summary.json"
    if historical_path.exists():
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        historical_test_mae = float(historical["leakage_audit"]["sensitivity_validation_selected"]["test_mae"])
    return {
        "method": "frozen_random_forest_feature_perturbation_with_global_importance",
        "score_layer": "severity_prediction",
        "estimate": original,
        "historical_validation_mae": float(info["validation_metrics"]["mae"]),
        "historical_test_mae": historical_test_mae,
        "exploratory": True,
        "excluded_bases": list(info["excluded_bases"]),
        "top_features": rows[:TOP_ENGINEERED],
        "top_variables": variables[:TOP_VARIABLES],
        "all_features": rows,
        "all_variables": variables,
    }


def explain_result(
    frame: pd.DataFrame,
    response: dict[str, Any],
    context: dict[str, Any],
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Explain one already-computed frozen v1 response without changing it."""
    if response.get("status") == "invalid_input":
        return {
            "sample_id": sample_id,
            "status": response.get("status"),
            "explanation_type": "frozen_v1_diagnostic_explanation",
            "top_features": [],
            "top_variables": [],
            "direction": None,
            "global_importance_reference": {},
            "ood_drivers": None,
            "severity_drivers": None,
            "uncertainty": None,
            "limitations": LIMITATIONS,
        }
    vector = _vector(frame, context)
    base, calibrated = _scores(vector, context)
    family = str(response.get("family_top1") or "")
    family_result = _family_local(vector, family, context)
    ood = _ood_explanation(vector, family, response.get("distance_score"), context)
    severity = _severity_local(vector, response, context)
    if response.get("status") != "accepted" or family != "primary_coolant_boundary_break":
        assert severity is None
    if severity is not None:
        assert_allowed_features([item["feature"] for item in severity["all_features"]], severity=True)
    global_reference = {
        "method": "permutation_importance",
        "cohort": "locked_release_test",
        "score_layer": "base_classifier_neg_log_loss",
        "aggregation": "sum absolute engineered-feature importance within each raw variable",
        "top_features": context.get("global_feature_rows", [])[:TOP_ENGINEERED],
    }
    return {
        "sample_id": sample_id,
        "status": response.get("status"),
        "family_top1": family or None,
        "explanation_type": "frozen_v1_diagnostic_explanation",
        "top_features": [] if family_result is None else family_result["top_features"],
        "top_variables": [] if family_result is None else family_result["top_variables"],
        "direction": "positive contribution supports the frozen base-classifier top-1 score; negative contribution opposes it",
        "global_importance_reference": global_reference,
        "family_diagnostics": family_result,
        "ood_drivers": ood,
        "uncertainty": _uncertainty_summary(response, calibrated, context),
        "severity_drivers": severity,
        "limitations": LIMITATIONS,
        "window_s": WINDOW_S,
        "input_time_limit_s": WINDOW_S,
        "base_classifier_probabilities": {label: float(value) for label, value in zip(context["artifacts"]["conformal"]["labels"], base)},
    }


def _global_importance(vectors: pd.DataFrame, labels: pd.Series, context: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = context["artifacts"]["base_model"]
    columns = context["columns"]
    label_values = labels.to_numpy()
    label_order = list(context["artifacts"]["conformal"]["labels"])
    baseline_probabilities = _aligned_probabilities(model, model.predict_proba(vectors[columns]), label_order)
    baseline_loss = float(log_loss(label_values, baseline_probabilities, labels=label_order))
    repeats = 8
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for feature in columns:
        permuted_frames = []
        for _ in range(repeats):
            shuffled = vectors.copy()
            shuffled[feature] = vectors[feature].to_numpy()[rng.permutation(len(vectors))]
            permuted_frames.append(shuffled)
        stacked = pd.concat(permuted_frames, ignore_index=True)
        permuted_probabilities = _aligned_probabilities(model, model.predict_proba(stacked[columns]), label_order)
        deltas = np.asarray([
            log_loss(label_values, permuted_probabilities[start : start + len(vectors)], labels=label_order) - baseline_loss
            for start in range(0, len(stacked), len(vectors))
        ], dtype=float)
        base, statistic = split_feature(feature)
        rows.append({
            "feature": feature,
            "variable": base,
            "statistic": statistic,
            "importance": float(np.mean(np.abs(deltas))),
            "mean_delta_neg_log_loss": float(deltas.mean()),
            "std_delta_neg_log_loss": float(deltas.std()),
            "n_repeats": repeats,
            "method": "permutation_importance",
            "score_layer": "base_classifier_neg_log_loss",
            "cohort": "locked_release_test",
        })
    rows.sort(key=lambda item: (-item["importance"], item["feature"]))
    variable_rows = []
    for variable, group in pd.DataFrame(rows).groupby("variable", sort=False):
        variable_rows.append({
            "variable": variable,
            "importance": float(group["importance"].sum()),
            "mean_delta_neg_log_loss": float(group["mean_delta_neg_log_loss"].sum()),
            "engineered_feature_count": int(len(group)),
            "aggregation": "sum across last/delta_t0/mean/std/slope",
            "method": "permutation_importance",
            "cohort": "locked_release_test",
        })
    variable_rows.sort(key=lambda item: (-item["importance"], item["variable"]))
    context["global_feature_rows"] = rows
    return pd.DataFrame(rows), pd.DataFrame(variable_rows)


def _plot_bar(rows: list[dict[str, Any]], value: str, label: str, title: str, path: Path, color: str = "#285f8f") -> None:
    if not rows:
        return
    rows = list(reversed(rows[:15]))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([str(row[label]) for row in rows], [float(row[value]) for row in rows], color=color)
    ax.set_xlabel(value)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_timeseries(frame: pd.DataFrame, variables: list[str], sample_id: str, path: Path) -> None:
    variables = [variable for variable in variables[:5] if variable in frame.columns]
    if not variables:
        return
    time = pd.to_numeric(frame["TIME"], errors="coerce")
    selected = frame.loc[time.le(INJECTION_S + WINDOW_S)].copy()
    fig, axes = plt.subplots(len(variables), 1, figsize=(9, max(2.2, 2.0 * len(variables))), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, variable in zip(axes, variables):
        axis.plot(selected["TIME"], selected[variable], color="#285f8f", lw=1.4)
        axis.axvline(INJECTION_S, color="#777777", ls="--", lw=0.8)
        axis.axvline(INJECTION_S + WINDOW_S, color="#a33f2b", ls=":", lw=0.9)
        axis.set_ylabel(variable)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("TIME (s)")
    fig.suptitle(f"{sample_id}: top explanation variables, 120 s window")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _perturb_frame(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    perturbed = frame.copy()
    time = pd.to_numeric(perturbed["TIME"], errors="coerce")
    mask = time.gt(INJECTION_S) & time.le(INJECTION_S + WINDOW_S)
    for feature in context["features"]:
        scale_column = f"{feature}__delta_t0"
        scale = float(context["artifacts"]["distance"]["scaler"].scale_[context["columns"].index(scale_column)])
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        perturbed.loc[mask, feature] = pd.to_numeric(perturbed.loc[mask, feature], errors="coerce") + 0.001 * scale
    return perturbed


def _faithfulness_rows(explanations: dict[str, dict[str, Any]], frames: dict[str, pd.DataFrame], responses: dict[str, dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(explanations):
        explanation = explanations[sample_id]
        diagnostics = explanation.get("family_diagnostics")
        if not diagnostics:
            continue
        vector = _vector(frames[sample_id], context)
        target = str(diagnostics["target_family"])
        labels = list(context["artifacts"]["conformal"]["labels"])
        target_index = labels.index(target)
        original = float(_scores(vector, context)[0][target_index])
        top = [item["feature"] for item in diagnostics["all_engineered"][:TOP_K]]
        top_ablated = vector.copy()
        top_ablated.loc[0, top] = context["reference"].loc[top].to_numpy()
        top_effect = original - float(_scores(top_ablated, context)[0][target_index])
        random_effects = []
        for _ in range(5):
            random_features = list(rng.choice(context["columns"], size=TOP_K, replace=False))
            random_ablated = vector.copy()
            random_ablated.loc[0, random_features] = context["reference"].loc[random_features].to_numpy()
            random_effects.append(original - float(_scores(random_ablated, context)[0][target_index]))
        rows.append({
            "sample_id": sample_id,
            "target": "family_score",
            "status": responses[sample_id].get("status"),
            "target_family": target,
            "k": TOP_K,
            "original_score": original,
            "top_k_features": json.dumps(top),
            "top_k_removal_effect": float(top_effect),
            "random_removal_effect_mean": float(np.mean(random_effects)),
            "random_removal_effect_std": float(np.std(random_effects)),
            "top_minus_random": float(top_effect - np.mean(random_effects)),
            "top_effect_exceeds_random": bool(top_effect > np.mean(random_effects)),
            "method": "reference replacement in frozen model input space",
        })
        severity = explanation.get("severity_drivers")
        if severity:
            info = context["artifacts"]["loca_severity"]
            model = info["model"]
            columns = list(info["columns"])
            severity_vector = vector[columns]
            reference = _severity_training_reference(context)
            severity_original = float(model.predict(severity_vector)[0])
            severity_top = [item["feature"] for item in severity["all_features"][:TOP_K]]
            ablated = severity_vector.copy()
            ablated.loc[0, severity_top] = reference.loc[severity_top].to_numpy()
            severity_effect = severity_original - float(model.predict(ablated)[0])
            random_effects = []
            for _ in range(5):
                random_features = list(rng.choice(columns, size=TOP_K, replace=False))
                random_ablated = severity_vector.copy()
                random_ablated.loc[0, random_features] = reference.loc[random_features].to_numpy()
                random_effects.append(severity_original - float(model.predict(random_ablated)[0]))
            assert_allowed_features(columns, severity=True)
            rows.append({
                "sample_id": sample_id,
                "target": "loca_severity",
                "status": responses[sample_id].get("status"),
                "target_family": target,
                "k": TOP_K,
                "original_score": severity_original,
                "top_k_features": json.dumps(severity_top),
                "top_k_removal_effect": float(severity_effect),
                "random_removal_effect_mean": float(np.mean(random_effects)),
                "random_removal_effect_std": float(np.std(random_effects)),
                "top_minus_random": float(severity_effect - np.mean(random_effects)),
                "top_effect_exceeds_random": bool(severity_effect > np.mean(random_effects)),
                "method": "reference replacement in frozen severity model input space",
            })
    return rows


def _stability_rows(explanations: dict[str, dict[str, Any]], frames: dict[str, pd.DataFrame], responses: dict[str, dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = []
    for status in ("accepted", "requires_review", "unknown"):
        selected.extend([sample_id for sample_id in sorted(explanations) if responses[sample_id].get("status") == status][:3])
    for sample_id in selected:
        original = explanations[sample_id]
        repeat = explain_result(frames[sample_id], responses[sample_id], context, sample_id)
        left = [item["variable"] for item in original.get("top_features", [])[:TOP_K]]
        right = [item["variable"] for item in repeat.get("top_features", [])[:TOP_K]]
        rows.append({
            "sample_id": sample_id,
            "status": responses[sample_id].get("status"),
            "comparison": "repeat_same_input",
            "top5_variable_jaccard": jaccard(left, right),
            "spearman_rank": rank_correlation(left, right),
            "target_score_change": float(repeat.get("family_diagnostics", {}).get("original_target_score", np.nan) - original.get("family_diagnostics", {}).get("original_target_score", np.nan)),
            "status_changed": False,
            "deterministic_exact": bool(json.dumps(original, sort_keys=True, default=_json_default) == json.dumps(repeat, sort_keys=True, default=_json_default)),
        })
        perturbed_frame = _perturb_frame(frames[sample_id], context)
        perturbed_response = diagnose_dataframe(perturbed_frame, context["artifact_root"])
        perturbed = explain_result(perturbed_frame, perturbed_response, context, sample_id)
        left = [item["variable"] for item in original.get("top_features", [])[:TOP_K]]
        right = [item["variable"] for item in perturbed.get("top_features", [])[:TOP_K]]
        rows.append({
            "sample_id": sample_id,
            "status": responses[sample_id].get("status"),
            "comparison": "0.1_percent_training_scale_perturbation",
            "top5_variable_jaccard": jaccard(left, right),
            "spearman_rank": rank_correlation(left, right),
            "target_score_change": float(perturbed.get("family_diagnostics", {}).get("original_target_score", np.nan) - original.get("family_diagnostics", {}).get("original_target_score", np.nan)),
            "status_changed": bool(perturbed_response.get("status") != responses[sample_id].get("status")),
            "perturbed_status": perturbed_response.get("status"),
            "deterministic_exact": False,
            "note": "Status changes are reported as model-boundary sensitivity; no stability correction is applied.",
        })
    return rows


def _summary_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()


def run_explainability(project_root: str | Path = PROJECT_ROOT, artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """Run the complete milestone-18 report and return its summary."""
    context = _load_context(project_root, artifact_root)
    root = context["root"]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    records = context["release"]
    frames: dict[str, pd.DataFrame] = {}
    responses: dict[str, dict[str, Any]] = {}
    vectors: list[pd.DataFrame] = []
    families: list[str] = []
    for record in records.itertuples(index=False):
        sample_id = str(record.sample_id)
        frame = _frame_for_record(context, record)
        frames[sample_id] = frame
        response = diagnose_dataframe(frame, artifact_root)
        responses[sample_id] = response
        vectors.append(_vector(frame, context))
        families.append(str(record.family))
    matrix = pd.concat(vectors, ignore_index=True)
    global_features, global_variables = _global_importance(matrix, pd.Series(families), context)
    global_features.to_csv(RESULT_ROOT / "18_global_feature_importance.csv", index=False)
    global_variables.to_csv(RESULT_ROOT / "18_global_variable_importance.csv", index=False)

    explanations: dict[str, dict[str, Any]] = {}
    for sample_id in sorted(frames):
        explanations[sample_id] = explain_result(frames[sample_id], responses[sample_id], context, sample_id)
        assert all(variable_for_feature(item["feature"]) in context["features"] for item in explanations[sample_id].get("top_features", []))
        if responses[sample_id].get("status") != "accepted":
            assert explanations[sample_id].get("severity_drivers") is None

    example_ids: dict[str, str] = {}
    for record in records.itertuples(index=False):
        sample_id = str(record.sample_id)
        response = responses[sample_id]
        tier = response.get("assessment", {}).get("family_capability", {}).get("tier")
        family = response.get("family_top1")
        if response.get("status") == "accepted" and tier == "Tier A" and "accepted_tier_a" not in example_ids:
            example_ids["accepted_tier_a"] = sample_id
        if response.get("status") == "accepted" and tier == "Tier A" and family != "primary_coolant_boundary_break" and "accepted_non_loca_tier_a" not in example_ids:
            example_ids["accepted_non_loca_tier_a"] = sample_id
        if response.get("status") == "accepted" and family == "primary_coolant_boundary_break" and response.get("severity_status") == "available_exploratory" and "accepted_loca_severity" not in example_ids:
            example_ids["accepted_loca_severity"] = sample_id
        if response.get("status") == "requires_review" and "requires_review" not in example_ids:
            example_ids["requires_review"] = sample_id
        if response.get("status") == "unknown" and "unknown" not in example_ids:
            example_ids["unknown"] = sample_id
    for key in ("accepted_tier_a", "requires_review", "unknown", "accepted_loca_severity"):
        if key not in example_ids:
            raise AssertionError(f"18 requires a real locked-release example for {key}")

    local_payload = {
        "method": "frozen v1.2 artifacts; local reference replacement; no refit",
        "examples": {key: explanations[sample_id] for key, sample_id in example_ids.items()},
        "example_sample_ids": example_ids,
        "status_counts": pd.Series([responses[sample_id]["status"] for sample_id in responses]).value_counts().to_dict(),
        "limitations": LIMITATIONS,
    }
    _write_json(RESULT_ROOT / "18_local_explanations.json", local_payload)

    ood_rows = []
    severity_rows = []
    for sample_id, explanation in explanations.items():
        ood = explanation.get("ood_drivers") or {}
        uncertainty = explanation.get("uncertainty") or {}
        ood_rows.append({
            "sample_id": sample_id,
            "status": responses[sample_id].get("status"),
            "predicted_family": responses[sample_id].get("family_top1"),
            "conformal_set": json.dumps(responses[sample_id].get("conformal_set", [])),
            "distance_score": responses[sample_id].get("distance_score"),
            "distance_threshold": ood.get("distance_warning_threshold"),
            "ood_warning": responses[sample_id].get("ood_warning", {}).get("flag"),
            "top_ood_features": json.dumps([item["feature"] for item in ood.get("top_features", [])[:TOP_ENGINEERED]]),
            "top_ood_variables": json.dumps([item["variable"] for item in ood.get("top_variables", [])[:TOP_VARIABLES]]),
            "ood_direction": json.dumps([{item["variable"]: item["direction"]} for item in ood.get("top_variables", [])[:TOP_VARIABLES]]),
            "maximum_calibrated_probability": uncertainty.get("maximum_calibrated_probability"),
            "top2_margin": uncertainty.get("top2_margin"),
            "nonconformity_of_top1": uncertainty.get("nonconformity_of_top1"),
            "unknown_interpretation": uncertainty.get("interpretation"),
        })
        severity = explanation.get("severity_drivers")
        if severity:
            for item in severity["top_features"]:
                severity_rows.append({
                    "sample_id": sample_id,
                    "status": responses[sample_id].get("status"),
                    "family": responses[sample_id].get("family_top1"),
                    "estimate": severity["estimate"],
                    "historical_validation_mae": severity["historical_validation_mae"],
                    "historical_test_mae": severity["historical_test_mae"],
                    "exploratory": severity["exploratory"],
                    **item,
                })
    pd.DataFrame(ood_rows).to_csv(RESULT_ROOT / "18_ood_explanations.csv", index=False)
    pd.DataFrame(severity_rows, columns=["sample_id", "status", "family", "estimate", "historical_validation_mae", "historical_test_mae", "exploratory", "feature", "variable", "statistic", "contribution", "abs_contribution", "direction", "global_model_importance"]).to_csv(RESULT_ROOT / "18_severity_explanations.csv", index=False)

    _plot_bar(global_features.to_dict("records"), "importance", "feature", "Global engineered-feature importance", FIGURE_ROOT / "18_global_engineered_feature_importance.png")
    _plot_bar(global_variables.to_dict("records"), "importance", "variable", "Global raw-variable importance", FIGURE_ROOT / "18_global_variable_importance.png")
    accepted = explanations[example_ids["accepted_tier_a"]]
    _plot_bar(accepted.get("top_variables", []), "importance", "variable", "Accepted sample: local family evidence", FIGURE_ROOT / "18_accepted_local_explanation.png", "#3b7f5f")
    review = explanations[example_ids["requires_review"]].get("ood_drivers") or {}
    _plot_bar(review.get("top_variables", []), "approx_contribution", "variable", "Requires-review: approximate OOD drivers", FIGURE_ROOT / "18_requires_review_ood_drivers.png", "#a35b2b")
    unknown = explanations[example_ids["unknown"]].get("uncertainty") or {}
    _plot_bar([{"family": item["family"], "probability": item["probability"]} for item in unknown.get("top2", [])], "probability", "family", "Unknown: calibrated probability summary", FIGURE_ROOT / "18_unknown_probability_summary.png", "#777777")
    loca = explanations[example_ids["accepted_loca_severity"]].get("severity_drivers") or {}
    _plot_bar(loca.get("top_variables", []), "importance", "variable", "Accepted LOCA: local severity evidence", FIGURE_ROOT / "18_accepted_loca_severity_explanation.png", "#8a4d8e")
    for key, sample_id in example_ids.items():
        variables = [item["variable"] for item in explanations[sample_id].get("top_variables", [])]
        _plot_timeseries(frames[sample_id], variables, sample_id, FIGURE_ROOT / f"18_timeseries_{_safe_name(key)}.png")

    stability = _stability_rows(explanations, frames, responses, context)
    pd.DataFrame(stability).to_csv(RESULT_ROOT / "18_explanation_stability.csv", index=False)
    faithfulness = _faithfulness_rows(explanations, frames, responses, context)
    pd.DataFrame(faithfulness).to_csv(RESULT_ROOT / "18_explanation_faithfulness.csv", index=False)

    parity = evaluate_pipeline_parity(root, artifact_root, write_outputs=False)
    regression, _ = run_regression_tests(root, artifact_root, parity_result=parity)
    previous_parity = json.loads((root / "results" / "17_v1_pipeline_parity.json").read_text(encoding="utf-8"))
    previous_regression = pd.read_csv(root / "results" / "17_v1_regression_tests.csv")
    regression_same = regression[["test", "passed", "status"]].reset_index(drop=True).equals(previous_regression[["test", "passed", "status"]].reset_index(drop=True))
    core_unchanged = bool(
        parity["overall_pass"]
        and parity["discrete_decision_agreement_100_percent"]
        and parity["max_float_abs_delta"] <= 1e-9
        and parity["discrete_decision_agreement_rate"] == previous_parity["discrete_decision_agreement_rate"]
        and regression["passed"].all()
        and regression_same
    )
    assert core_unchanged
    assert context["manifest_check"]["all_passed"]
    assert all(explanation.get("window_s") == WINDOW_S for explanation in explanations.values())
    assert all(explanation.get("severity_drivers") is None or not ({"LVPZ", "P", "TSAT", "VOL"} & {item["variable"] for item in explanation["severity_drivers"]["all_features"]}) for explanation in explanations.values())
    assert all(not ({"WLR", "WUP", "WHPI", "WECS", "RM1", "RM2", "FRCL"} & {item["variable"] for item in explanation.get("top_features", [])}) for explanation in explanations.values())
    stability_frame = pd.DataFrame(stability)
    faithfulness_frame = pd.DataFrame(faithfulness)
    summary = {
        "milestone": "18 Explainability",
        "status": "passed",
        "model_version": context["artifacts"]["manifest"]["model_version"],
        "methods": {
            "family_global": "permutation importance on frozen HistGradientBoosting base classifier, neg_log_loss, 8 deterministic repeats",
            "family_local": "leave-one-engineered-feature-out to frozen distance-model training center; explains base score, not sigmoid calibration causally",
            "ood": "diagonal standardized class-conditional Ledoit-Wolf Mahalanobis approximation with signed standardized displacement",
            "severity": "frozen RandomForest feature perturbation to reconstructed frozen training medians plus model feature importance; historical sensitivity Test MAE≈0.972",
        },
        "cohort": {"name": "locked_release_test", "sample_count": int(len(records)), "window_s": WINDOW_S},
        "status_counts": pd.Series([responses[sample_id]["status"] for sample_id in responses]).value_counts().to_dict(),
        "global_top_variables": global_variables.head(10).to_dict("records"),
        "examples": {key: {"sample_id": sample_id, "status": responses[sample_id].get("status"), "family": responses[sample_id].get("family_top1"), "severity_status": responses[sample_id].get("severity_status")} for key, sample_id in example_ids.items()},
        "stability": {
            "row_count": int(len(stability_frame)),
            "same_input_exact": bool(stability_frame.loc[stability_frame["comparison"].eq("repeat_same_input"), "deterministic_exact"].all()),
            "perturbation_mean_top5_jaccard": float(stability_frame.loc[stability_frame["comparison"].ne("repeat_same_input"), "top5_variable_jaccard"].mean()),
            "status_changes": int(stability_frame.loc[stability_frame["comparison"].ne("repeat_same_input"), "status_changed"].sum()),
        },
        "faithfulness": {
            "row_count": int(len(faithfulness_frame)),
            "family_top_minus_random_mean": float(faithfulness_frame.loc[faithfulness_frame["target"].eq("family_score"), "top_minus_random"].mean()),
            "severity_top_minus_random_mean": float(faithfulness_frame.loc[faithfulness_frame["target"].eq("loca_severity"), "top_minus_random"].mean()) if (faithfulness_frame["target"] == "loca_severity").any() else None,
            "family_top_effect_exceeds_random_rate": float(faithfulness_frame.loc[faithfulness_frame["target"].eq("family_score"), "top_effect_exceeds_random"].mean()),
        },
        "severity_reference": {
            "frozen_validation_mae": float(loca.get("historical_validation_mae")) if loca else None,
            "historical_sensitivity_test_mae": float(loca.get("historical_test_mae")) if loca and loca.get("historical_test_mae") is not None else None,
            "exploratory": True,
        },
        "artifact_sha256_passed": bool(context["manifest_check"]["all_passed"]),
        "core_diagnosis_unchanged": core_unchanged,
        "api_batch_parity": {"overall_pass": parity["overall_pass"], "agreement_rate": parity["discrete_decision_agreement_rate"], "max_float_abs_delta": parity["max_float_abs_delta"]},
        "regression_tests": {"passed": int(regression["passed"].sum()), "total": int(len(regression)), "all_passed": bool(regression["passed"].all())},
        "dashboard_gate": {"eligible": True, "reason": "Explainability artifacts generated; frozen artifact hashes, 17.2 parity, and regression checks all passed."},
        "limitations": LIMITATIONS,
    }
    summary["summary_sha256"] = _summary_hash(summary)
    _write_json(RESULT_ROOT / "18_summary.json", summary)
    return summary
