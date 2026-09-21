"""Frozen-artifact parity evaluation against the milestone-16 cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from .v1 import (
    ARTIFACT_ROOT,
    _aligned_probabilities,
    _distance_values,
    _load_artifacts_cached,
    _sample_id_sha256,
    _status_from_policy,
    _write_json,
    verify_artifact_manifest,
)
from ..hierarchical_diagnosis import family_mapping
from ..multi_accident_features import strict_process_features
from ..severity_invariant import PROJECT_ROOT, load_ood_context


PARITY_TOLERANCE = 0.03


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], accepted: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accepted = np.asarray(accepted, dtype=bool)
    n = int(len(y_true))
    n_accepted = int(accepted.sum())
    if not n_accepted:
        return {
            "cohort_size": n,
            "accepted_count": 0,
            "coverage": 0.0,
            "selective_accuracy": np.nan,
            "selective_macro_f1": np.nan,
            "family_balanced_accuracy": np.nan,
            "risk": np.nan,
            "reject_rate": 1.0 if n else np.nan,
        }
    true = y_true[accepted]
    pred = y_pred[accepted]
    accuracy = float(np.mean(true == pred))
    observed = sorted(set(true.tolist()))
    return {
        "cohort_size": n,
        "accepted_count": n_accepted,
        "coverage": float(n_accepted / n) if n else 0.0,
        "selective_accuracy": accuracy,
        "selective_macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "family_balanced_accuracy": float(recall_score(true, pred, labels=observed, average="macro", zero_division=0)),
        "risk": float(1.0 - accuracy),
        "reject_rate": float(1.0 - n_accepted / n) if n else np.nan,
    }


def _target_rows(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics = pd.read_csv(result_root / "16_selective_metrics.csv")
    conformal = pd.read_csv(result_root / "16_conformal_metrics.csv")
    forced = metrics.loc[
        metrics["split_type"].eq("severity_extrapolation")
        & metrics["window_s"].eq(120)
        & metrics["feature_group"].eq("A_strict_38")
        & metrics["eval_split"].eq("test")
        & metrics["mechanism"].eq("forced_family")
    ]
    conf = conformal.loc[
        conformal["split_type"].eq("severity_extrapolation")
        & conformal["window_s"].eq(120)
        & conformal["feature_group"].eq("A_strict_38")
        & conformal["eval_split"].eq("test")
        & conformal["alpha"].eq(0.10)
    ]
    if forced.empty or conf.empty:
        raise AssertionError("Milestone-16 parity target rows are missing")
    return forced, conf, {
        "forced_family": {key: float(forced[key].mean()) for key in ("coverage", "selective_accuracy", "selective_macro_f1", "selective_balanced_accuracy", "risk", "reject_rate")},
        "conformal_alpha_0.10": {key: float(conf[key].mean()) for key in ("empirical_set_coverage", "coverage", "selective_accuracy", "selective_macro_f1", "selective_balanced_accuracy", "risk", "reject_rate")},
    }


def evaluate_artifact_parity(
    project_root: str | Path = PROJECT_ROOT,
    artifact_root: str | Path = ARTIFACT_ROOT,
    tolerance: float = PARITY_TOLERANCE,
    write_outputs: bool = True,
) -> dict[str, Any]:
    root = Path(project_root)
    artifacts = _load_artifacts_cached(str(Path(artifact_root).resolve()))
    manifest_check = verify_artifact_manifest(artifact_root)
    dataset, assignments = load_ood_context(project_root=root)
    held_out = assignments.loc[
        assignments["split_type"].eq("severity_extrapolation") & assignments["partition"].eq("test"),
        ["sample_id", "split_id"],
    ]
    frame = dataset.loc[dataset["window_s"].eq(120)].merge(held_out, on="sample_id", how="inner", validate="one_to_one")
    mapping = family_mapping()
    class_to_family = dict(zip(mapping["class"], mapping["family"]))
    frame["family"] = frame["accident_class"].map(class_to_family)
    labels = list(artifacts["conformal"]["labels"])
    columns = artifacts["feature_schema"]["derived_feature_columns"]
    probabilities = _aligned_probabilities(artifacts["calibrator"], artifacts["calibrator"].predict_proba(frame[columns]), labels)
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    distances = _distance_values(artifacts["distance"], frame[columns], predicted)
    conformal_sets = probabilities >= float(artifacts["conformal"]["probability_cutoff"]) - 1e-12
    singleton = conformal_sets.sum(axis=1) == 1
    contains_true = np.array([conformal_sets[i, labels.index(label)] for i, label in enumerate(frame["family"])], dtype=bool)
    distance_warning = distances > float(artifacts["policy"]["distance_warning_threshold"])
    statuses = np.array([
        _status_from_policy(
            [label for label, included in zip(labels, conformal_sets[i]) if included],
            bool(distance_warning[i]),
            artifacts["capability"].get(predicted[i], {"tier": "Tier C"}),
        )
        for i in range(len(frame))
    ])
    policy_accepted = statuses == "accepted"
    true = frame["family"].to_numpy()
    frozen_metrics = {
        "forced_family": _metrics(true, predicted, labels, np.ones(len(frame), dtype=bool)),
        "conformal_alpha_0.10": _metrics(true, predicted, labels, singleton),
        "v1_status_policy": _metrics(true, predicted, labels, policy_accepted),
    }
    frozen_metrics["conformal_alpha_0.10"]["empirical_set_coverage"] = float(contains_true.mean())
    reject_rates = {
        scope: {
            accident_class: float((~accepted[frame["accident_class"].to_numpy() == accident_class]).mean())
            for accident_class in ("RI", "LOCAC", "SLBIC")
        }
        for scope, accepted in (("conformal_alpha_0.10", singleton), ("v1_status_policy", policy_accepted))
    }
    forced, conf, targets = _target_rows(root / "results")
    rows: list[dict[str, Any]] = []

    def add(scope: str, metric: str, research_target: float, frozen_result: float) -> None:
        delta = float(frozen_result - research_target)
        rows.append({
            "comparison_scope": scope,
            "metric": metric,
            "research_target": research_target,
            "frozen_result": frozen_result,
            "delta": delta,
            "tolerance": float(tolerance),
            "pass": bool(abs(delta) <= tolerance),
            "cohort_size": int(len(frame)),
        })

    for metric in ("coverage", "selective_accuracy", "selective_macro_f1", "family_balanced_accuracy", "risk", "reject_rate"):
        add("forced_family", metric, targets["forced_family"]["selective_balanced_accuracy" if metric == "family_balanced_accuracy" else metric], frozen_metrics["forced_family"][metric])
    for metric in ("coverage", "selective_accuracy", "selective_macro_f1", "family_balanced_accuracy", "risk", "reject_rate"):
        add("conformal_alpha_0.10", metric, targets["conformal_alpha_0.10"]["selective_balanced_accuracy" if metric == "family_balanced_accuracy" else metric], frozen_metrics["conformal_alpha_0.10"][metric])
    add("conformal_alpha_0.10", "empirical_set_coverage", targets["conformal_alpha_0.10"]["empirical_set_coverage"], frozen_metrics["conformal_alpha_0.10"]["empirical_set_coverage"])
    rejection_targets = pd.read_csv(root / "results" / "16_rejection_by_class.csv")
    rejection_targets = rejection_targets.loc[
        rejection_targets["window_s"].eq(120)
        & rejection_targets["feature_group"].eq("A_strict_38")
        & rejection_targets["mechanism"].eq("conformal_alpha_0.10")
        & rejection_targets["accident_class"].isin(["RI", "LOCAC", "SLBIC"])
    ].groupby("accident_class")["reject_rate"].mean()
    for accident_class in ("RI", "LOCAC", "SLBIC"):
        add("conformal_alpha_0.10", f"{accident_class}_reject_rate", float(rejection_targets[accident_class]), reject_rates["conformal_alpha_0.10"][accident_class])

    policy_rows = {
        "status_counts": pd.Series(statuses).value_counts().to_dict(),
        "metrics": frozen_metrics["v1_status_policy"],
        "reject_rates": reject_rates["v1_status_policy"],
        "note": "Diagnostic only: milestone-16 parity targets are the conformal singleton policy, while v1 status policy also applies distance and capability gates.",
    }
    overall_pass = manifest_check["all_passed"] and all(row["pass"] for row in rows)
    result = {
        "model_version": artifacts["manifest"]["model_version"],
        "overall_pass": bool(overall_pass),
        "gate_reason": "All artifact hashes and parity comparisons are within tolerance." if overall_pass else "Frozen artifact parity is outside the declared tolerance or artifact hashes failed.",
        "parity_tolerance": float(tolerance),
        "cohort": {
            "window_s": 120,
            "split_type": "severity_extrapolation",
            "split_ids": sorted(held_out["split_id"].unique().tolist()),
            "sample_count": int(len(frame)),
            "research_target_source": "results/16_selective_metrics.csv and results/16_conformal_metrics.csv",
            "frozen_artifact_training_protocol": artifacts["manifest"]["family_classifier"],
            "cohort_difference": "Same 206 trajectory/sample_id grouped test rows as milestone 16; v1 artifacts were fit once on random/seed_20260921 train rows, whereas milestone 16 fit separate severity-extrapolation direction models.",
        },
        "research_targets": targets,
        "frozen_metrics": frozen_metrics,
        "reject_rates": reject_rates,
        "policy_diagnostic": policy_rows,
        "manifest_verification": manifest_check,
        "comparisons": rows,
        "gate": {
            "comparison_count": len(rows),
            "passed_count": int(sum(row["pass"] for row in rows)),
            "all_comparisons_pass": all(row["pass"] for row in rows),
        },
    }
    if write_outputs:
        output = root / "results"
        pd.DataFrame(rows).to_csv(output / "17_v1_artifact_parity.csv", index=False)
        _write_json(output / "17_v1_artifact_parity.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate_artifact_parity(), ensure_ascii=False, indent=2, default=str))


def _locked_release_context(project_root: str | Path, artifact_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(project_root)
    artifact_path = Path(artifact_root)
    split_manifest = json.loads((artifact_path / "release_split_manifest.json").read_text(encoding="utf-8"))
    records = pd.DataFrame(split_manifest["records"])
    release = records.loc[records["partition"].eq("locked_release_test")].copy()
    dataset, _ = load_ood_context(project_root=root)
    release = release.rename(columns={"accident_class": "true_class", "family": "true_family"})
    frame = dataset.loc[dataset["window_s"].eq(120)].merge(
        release[["sample_id", "true_class", "true_family", "severity", "operation_csv"]],
        on="sample_id", how="inner", validate="one_to_one",
    )
    if len(frame) != len(release) or len(frame) == 0:
        raise AssertionError("Locked release test does not match the 120 s feature dataset")
    return frame.sort_values("sample_id").reset_index(drop=True), split_manifest


def _batch_predictions(frame: pd.DataFrame, artifact_root: str | Path) -> pd.DataFrame:
    artifacts = _load_artifacts_cached(str(Path(artifact_root).resolve()))
    labels = list(artifacts["conformal"]["labels"])
    columns = artifacts["feature_schema"]["derived_feature_columns"]
    probabilities = _aligned_probabilities(artifacts["calibrator"], artifacts["calibrator"].predict_proba(frame[columns]), labels)
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    distances = _distance_values(artifacts["distance"], frame[columns], predicted)
    conformal_sets = [[label for label, probability in zip(labels, probabilities[index]) if probability >= artifacts["conformal"]["probability_cutoff"] - 1e-12] for index in range(len(frame))]
    distance_warning = distances > float(artifacts["policy"]["distance_warning_threshold"])
    statuses = np.array([
        _status_from_policy(conformal_sets[index], bool(distance_warning[index]), artifacts["capability"].get(predicted[index], {"tier": "Tier C"}))
        for index in range(len(frame))
    ])
    severity_run = np.array([
        status == "accepted"
        and predicted[index] == "primary_coolant_boundary_break"
        and artifacts["capability"].get(predicted[index], {}).get("tier") == "Tier A"
        for index, status in enumerate(statuses)
    ], dtype=bool)
    return pd.DataFrame({
        "sample_id": frame["sample_id"].astype(str).to_numpy(),
        "true_family": frame["true_family"].astype(str).to_numpy(),
        "true_class": frame["true_class"].astype(str).to_numpy(),
        "family_top1": predicted,
        "family_confidence": probabilities.max(axis=1),
        "conformal_set": conformal_sets,
        "ood_warning": distance_warning,
        "distance_score": distances,
        "status": statuses,
        "tier": [artifacts["capability"].get(label, {"tier": "Tier C"}).get("tier", "Tier C") for label in predicted],
        "severity_status": np.where(severity_run, "available_exploratory", "not_run_due_to_family_gating"),
    })


def evaluate_pipeline_parity(
    project_root: str | Path = PROJECT_ROOT,
    artifact_root: str | Path = ARTIFACT_ROOT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Compare frozen-artifact batch decisions with the public dataframe API."""
    root = Path(project_root)
    frame, split_manifest = _locked_release_context(root, artifact_root)
    batch = _batch_predictions(frame, artifact_root)
    api_rows: list[dict[str, Any]] = []
    for row in split_manifest["records"]:
        if row["partition"] != "locked_release_test":
            continue
        response = __import__("src.inference.v1", fromlist=["diagnose_dataframe"]).diagnose_dataframe(
            pd.read_csv(root / Path(row["operation_csv"])), artifact_root
        )
        api_rows.append({
            "sample_id": str(row["sample_id"]),
            "api_family_top1": response.get("family_top1"),
            "api_family_confidence": response.get("family_confidence"),
            "api_conformal_set": response.get("conformal_set", []),
            "api_ood_warning": response.get("ood_warning", {}).get("flag"),
            "api_distance_score": response.get("distance_score"),
            "api_status": response.get("status"),
            "api_severity_status": response.get("severity_status"),
        })
    api = pd.DataFrame(api_rows).sort_values("sample_id").reset_index(drop=True)
    batch = batch.sort_values("sample_id").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    discrete_fields = ("family_top1", "status", "conformal_set", "ood_warning", "severity_status")
    for index in range(len(batch)):
        left = batch.iloc[index]
        right = api.iloc[index]
        comparisons = {
            "family_top1": str(left["family_top1"]) == str(right["api_family_top1"]),
            "status": str(left["status"]) == str(right["api_status"]),
            "conformal_set": list(left["conformal_set"]) == list(right["api_conformal_set"]),
            "ood_warning": bool(left["ood_warning"]) == bool(right["api_ood_warning"]),
            "severity_status": str(left["severity_status"]) == str(right["api_severity_status"]),
        }
        confidence_delta = abs(float(left["family_confidence"]) - float(right["api_family_confidence"]))
        distance_delta = abs(float(left["distance_score"]) - float(right["api_distance_score"]))
        rows.append({
            "sample_id": left["sample_id"],
            "family_top1_match": comparisons["family_top1"],
            "status_match": comparisons["status"],
            "conformal_set_match": comparisons["conformal_set"],
            "ood_warning_match": comparisons["ood_warning"],
            "severity_status_match": comparisons["severity_status"],
            "discrete_decisions_match": all(comparisons.values()),
            "family_confidence_abs_delta": confidence_delta,
            "distance_abs_delta": distance_delta,
            "float_values_within_1e-9": confidence_delta <= 1e-9 and distance_delta <= 1e-9,
        })
    comparisons = pd.DataFrame(rows)
    manifest_check = verify_artifact_manifest(artifact_root)
    discrete_rate = float(comparisons["discrete_decisions_match"].mean()) if len(comparisons) else 0.0
    max_float_delta = float(max(comparisons["family_confidence_abs_delta"].max(), comparisons["distance_abs_delta"].max())) if len(comparisons) else np.nan
    result = {
        "model_version": _load_artifacts_cached(str(Path(artifact_root).resolve()))["manifest"]["model_version"],
        "cohort": {"split_manifest": "artifacts/v1/release_split_manifest.json", "sample_count": int(len(comparisons)), "sample_id_sha256": _sample_id_sha256(set(comparisons["sample_id"].astype(str)))},
        "discrete_decision_agreement_rate": discrete_rate,
        "discrete_decision_agreement_100_percent": bool(discrete_rate == 1.0),
        "max_float_abs_delta": max_float_delta,
        "float_tolerance": 1e-9,
        "manifest_verification": manifest_check,
        "overall_pass": bool(manifest_check["all_passed"] and discrete_rate == 1.0 and bool(comparisons["float_values_within_1e-9"].all())),
        "gate_reason": "Exact API/batch discrete parity passed at 100% with float deltas within 1e-9." if manifest_check["all_passed"] and discrete_rate == 1.0 and bool(comparisons["float_values_within_1e-9"].all()) else "API/batch parity failed.",
    }
    if write_outputs:
        comparisons.to_csv(root / "results" / "17_v1_pipeline_parity.csv", index=False)
        _write_json(root / "results" / "17_v1_pipeline_parity.json", result)
    return result


def _benchmark_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], accepted: np.ndarray) -> dict[str, float]:
    accepted = np.asarray(accepted, dtype=bool)
    if not accepted.any():
        return {"coverage": 0.0, "accuracy": np.nan, "macro_f1": np.nan, "balanced_accuracy": np.nan, "risk": np.nan, "reject_rate": 1.0}
    true = y_true[accepted]
    pred = y_pred[accepted]
    observed = [label for label in labels if np.any(true == label)]
    accuracy = float(np.mean(true == pred))
    return {
        "coverage": float(accepted.mean()),
        "accuracy": accuracy,
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(true, pred, labels=observed, average="macro", zero_division=0)),
        "risk": float(1.0 - accuracy),
        "reject_rate": float(1.0 - accepted.mean()),
    }


def evaluate_release_benchmark(
    project_root: str | Path = PROJECT_ROOT,
    artifact_root: str | Path = ARTIFACT_ROOT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Score the locked release cohort without fitting or threshold selection."""
    root = Path(project_root)
    frame, split_manifest = _locked_release_context(root, artifact_root)
    predictions = _batch_predictions(frame, artifact_root)
    labels = sorted(predictions["true_family"].unique().tolist())
    y_true = predictions["true_family"].to_numpy()
    y_pred = predictions["family_top1"].to_numpy()
    forced = _benchmark_metrics(y_true, y_pred, labels, np.ones(len(predictions), dtype=bool))
    accepted = predictions["status"].eq("accepted").to_numpy()
    selective = _benchmark_metrics(y_true, y_pred, labels, accepted)
    conformal_contains = np.array([true in values for true, values in zip(y_true, predictions["conformal_set"])], dtype=bool)
    per_family_recall = {
        family: {"support": int((y_true == family).sum()), "recall": float(np.mean(y_pred[y_true == family] == family)) if np.any(y_true == family) else np.nan}
        for family in labels
    }
    status_counts = predictions["status"].value_counts().sort_index().to_dict()
    tier_status_counts = predictions.groupby(["tier", "status"], sort=True).size().to_dict()
    result = {
        "benchmark_name": "v1 release benchmark",
        "model_version": _load_artifacts_cached(str(Path(artifact_root).resolve()))["manifest"]["model_version"],
        "protocol": {"split_manifest": "artifacts/v1/release_split_manifest.json", "release_sample_count": int(len(predictions)), "release_sample_id_sha256": _sample_id_sha256(set(predictions["sample_id"].astype(str))), "not_the_milestone_16_research_protocol": True},
        "forced_family": forced,
        "selective_policy": selective,
        "conformal": {"empirical_set_coverage": float(conformal_contains.mean()), "average_set_size": float(predictions["conformal_set"].map(len).mean()), "empty_set_rate": float(predictions["conformal_set"].map(len).eq(0).mean())},
        "distance_ood": {"warning_rate": float(predictions["ood_warning"].mean()), "warning_count": int(predictions["ood_warning"].sum())},
        "per_family_recall": per_family_recall,
        "status_distribution": {str(key): int(value) for key, value in status_counts.items()},
        "tier_status_distribution": {f"{key[0]}::{key[1]}": int(value) for key, value in tier_status_counts.items()},
        "sample_results": predictions.assign(conformal_set=predictions["conformal_set"].map(list)).to_dict(orient="records"),
    }
    if write_outputs:
        rows: list[dict[str, Any]] = []
        for scope, values in (("forced_family", forced), ("selective_policy", selective), ("conformal", result["conformal"]), ("distance_ood", result["distance_ood"])):
            for metric, value in values.items():
                rows.append({"metric_group": scope, "metric": metric, "family": "", "value": value})
        for family, values in per_family_recall.items():
            rows.append({"metric_group": "per_family_recall", "metric": "recall", "family": family, "value": values["recall"]})
            rows.append({"metric_group": "per_family_recall", "metric": "support", "family": family, "value": values["support"]})
        for status, count in result["status_distribution"].items():
            rows.append({"metric_group": "status_distribution", "metric": "count", "family": status, "value": count})
        for tier_status, count in result["tier_status_distribution"].items():
            rows.append({"metric_group": "tier_status_distribution", "metric": "count", "family": tier_status, "value": count})
        pd.DataFrame(rows).to_csv(root / "results" / "17_v1_release_benchmark.csv", index=False)
        _write_json(root / "results" / "17_v1_release_benchmark.json", result)
    return result
