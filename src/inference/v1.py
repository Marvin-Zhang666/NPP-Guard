"""NPP-Guard v1 fixed-artifact inference and regression checks.

The artifact builder is intentionally separate from diagnosis: production-like
inference only reads the saved models and never fits a model or threshold.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from ..hierarchical_diagnosis import family_mapping
    from ..multi_accident_features import strict_process_features
    from ..multi_accident_models import model_suite as classification_model_suite
    from ..ood_selective import _aligned_probabilities, _fit_calibrated, _fit_distance, _distance_values
    from ..severity_features import (
        INJECTION_S,
        SUMMARY_STATISTICS,
        build_severity_features,
        extract_window_features,
    )
    from ..severity_models import model_suite as severity_model_suite
    from ..severity_invariant import PROJECT_ROOT, feature_columns, load_ood_context
except ImportError:
    from hierarchical_diagnosis import family_mapping
    from multi_accident_features import strict_process_features
    from multi_accident_models import model_suite as classification_model_suite
    from ood_selective import _aligned_probabilities, _fit_calibrated, _fit_distance, _distance_values
    from severity_features import INJECTION_S, SUMMARY_STATISTICS, build_severity_features, extract_window_features
    from severity_models import model_suite as severity_model_suite
    from severity_invariant import PROJECT_ROOT, feature_columns, load_ood_context


ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "v1"
RESULT_ROOT = PROJECT_ROOT / "results"
MODEL_VERSION = "npp_guard_v1_17_2_20260921"
RANDOM_SEED = 20260921
PROTOCOL_VERSION = "v1_release_protocol_17_2"
WINDOW_S = 120
EXPECTED_INTERVAL_S = 10.0
INTERVAL_TOLERANCE_S = 1.0
CONFORMAL_ALPHA = 0.10
DISTANCE_TARGET_COVERAGE = 0.70
CLASSIFIER_MODEL = "hist_gradient_boosting"
LOCA_EXCLUDED_BASES = ("LVPZ", "P", "TSAT", "VOL")
LIMITATIONS = [
    "Research prototype; not for safety-critical deployment.",
    "Family-level diagnosis remains affected by severity out-of-distribution shift.",
    "The milestone-16 conformal empirical set coverage was 68.9%; no 90% statistical coverage guarantee is claimed.",
    "Subtype OOD is unresolved; v1 never forces a 12-class subtype output.",
    "LOCA severity estimation is exploratory and simulation-specific.",
    "Protection-time prediction is exploratory in the research record and is not available in v1 because a valid chained input was not established.",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_id_sha256(sample_ids: list[str] | set[str]) -> str:
    return _canonical_sha256(sorted(str(sample_id) for sample_id in sample_ids))


def _save_joblib(path: Path, payload: Any) -> None:
    joblib.dump(payload, path, compress=3)


def _calibration_cutoff(probabilities: np.ndarray, y_true: np.ndarray, labels: list[str], alpha: float) -> tuple[float, float]:
    label_index = {label: index for index, label in enumerate(labels)}
    scores = np.asarray([1.0 - probabilities[row, label_index[str(label)]] for row, label in enumerate(y_true)], dtype=float)
    scores = np.sort(scores[np.isfinite(scores)])
    if not len(scores):
        return 1.0, 0.0
    index = min(len(scores) - 1, max(0, int(np.ceil((len(scores) + 1) * (1.0 - alpha))) - 1))
    quantile = float(scores[index])
    return float(1.0 - quantile), quantile


def _capability_gate(root: Path) -> dict[str, dict[str, Any]]:
    matrix = json.loads((root / "results" / "15_npp_guard_v1_capability_matrix.json").read_text(encoding="utf-8"))
    return {
        row["family"]: {
            "tier": row["tier"],
            "status": row["status"],
            "reason": row["reason"],
            "official_members": row["official_members"],
            "matched_trajectories": row["matched_trajectories"],
        }
        for row in matrix["family_tiers"]
    }


def _protocol_inventory(root: Path) -> pd.DataFrame:
    assignments = pd.read_csv(root / "results" / "13_split_inventory.csv")
    base = assignments.loc[
        assignments["split_type"].eq("random")
        & assignments["split_id"].eq("seed_20260921")
        & assignments["partition"].isin(["train", "validation", "test"]),
        ["sample_id", "accident_class", "severity", "partition"],
    ].drop_duplicates("sample_id")
    inventory = pd.read_csv(root / "results" / "09_multi_accident_class_inventory.csv")
    base = base.merge(inventory[["sample_id", "operation_csv"]], on="sample_id", how="left", validate="one_to_one")
    mapping = family_mapping()
    base["family"] = base["accident_class"].map(dict(zip(mapping["class"], mapping["family"])))
    if len(base) != 505 or base["sample_id"].duplicated().any() or base["operation_csv"].isna().any():
        raise AssertionError("17.2 protocol source must contain 505 unique trajectory/sample_id records")
    if base["family"].isna().any():
        raise AssertionError("17.2 protocol source contains an unmapped accident class")
    return base.sort_values("sample_id").reset_index(drop=True)


def _make_protocol_manifest(root: Path, output: Path) -> dict[str, Any]:
    existing = output / "release_split_manifest.json"
    if existing.exists():
        manifest = _load_artifact_json(output, "release_split_manifest.json")
        expected = manifest.get("content_sha256")
        body = {key: value for key, value in manifest.items() if key != "content_sha256"}
        if expected != _canonical_sha256(body):
            raise AssertionError("Locked release split manifest content hash failed")
        return manifest

    source = _protocol_inventory(root)
    release_source = source.loc[source["partition"].eq("test")].copy()
    development_pool = source.loc[source["partition"].isin(["train", "validation"])].copy()
    rng = np.random.default_rng(RANDOM_SEED)
    assigned: list[dict[str, Any]] = []
    for accident_class, group in development_pool.groupby("accident_class", sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        order = rng.permutation(len(group))
        shuffled = group.iloc[order].reset_index(drop=True)
        n = len(shuffled)
        n_calibration = max(1, round(n * 0.20)) if n >= 3 else 0
        n_validation = max(1, round(n * 0.20)) if n >= 4 else 0
        while n - n_calibration - n_validation < 1:
            if n_validation:
                n_validation -= 1
            elif n_calibration:
                n_calibration -= 1
            else:
                raise AssertionError(f"No development train row remains for {accident_class}")
        partitions = (
            ("calibration", n_calibration),
            ("development_validation", n_validation),
            ("development_train", n - n_calibration - n_validation),
        )
        offset = 0
        for partition, count in partitions:
            for row in shuffled.iloc[offset : offset + count].itertuples(index=False):
                assigned.append({"sample_id": row.sample_id, "accident_class": row.accident_class, "severity": float(row.severity), "family": row.family, "operation_csv": row.operation_csv, "partition": partition})
            offset += count
    for row in release_source.itertuples(index=False):
        assigned.append({"sample_id": row.sample_id, "accident_class": row.accident_class, "severity": float(row.severity), "family": row.family, "operation_csv": row.operation_csv, "partition": "locked_release_test"})
    records = sorted(assigned, key=lambda row: str(row["sample_id"]))
    if len(records) != 505 or len({row["sample_id"] for row in records}) != 505:
        raise AssertionError("17.2 split assignment does not cover 505 unique trajectories")
    counts = pd.Series([row["partition"] for row in records]).value_counts().to_dict()
    if set(counts) != {"development_train", "development_validation", "calibration", "locked_release_test"}:
        raise AssertionError("17.2 split assignment is missing a required partition")
    mapping = family_mapping().to_dict(orient="records")
    feature_schema_hash = _canonical_sha256({"strict_process_features": strict_process_features(), "derived_feature_columns": feature_columns(strict_process_features()), "window_s": WINDOW_S, "injection_s": INJECTION_S})
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "split_rule": "Lock the existing random/seed_20260921 test partition as release test; stratify the remaining development pool by accident_class with a deterministic permutation.",
        "source": "results/13_split_inventory.csv random/seed_20260921 train+validation+test and results/09_multi_accident_class_inventory.csv",
        "locked_release_source": "results/13_split_inventory.csv random/seed_20260921 test",
        "family_mapping_version": "official_local_README_Table_1_mapping_v14",
        "family_mapping_hash": _canonical_sha256(mapping),
        "feature_schema_hash": feature_schema_hash,
        "records": records,
        "partition_counts": counts,
        "sets": {
            "family_classifier_train": {"partition": "development_train", "source": "protocol/development_train"},
            "family_classifier_validation": {"partition": "development_validation", "source": "protocol/development_validation"},
            "probability_calibrator_fit": {"partition": "calibration", "source": "protocol/calibration"},
            "conformal_calibration": {"partition": "calibration", "source": "protocol/calibration"},
            "distance_ood_fit": {"partition": "development_train", "source": "protocol/development_train"},
            "distance_threshold_calibration": {"partition": "calibration", "source": "protocol/calibration"},
            "loca_severity_train": {"partition": "development_train", "accident_class": "LOCA", "source": "protocol/development_train/LOCA"},
            "loca_severity_validation": {"partition": "development_validation", "accident_class": "LOCA", "source": "protocol/development_validation/LOCA"},
            "release_evaluation_test": {"partition": "locked_release_test", "source": "locked release test"},
        },
    }
    body["content_sha256"] = _canonical_sha256(body)
    _write_json(existing, body)
    return body


def _protocol_rows(manifest: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(manifest["records"])


def _set_ids(manifest: dict[str, Any], name: str) -> set[str]:
    spec = manifest["sets"][name]
    rows = _protocol_rows(manifest)
    selected = rows.loc[rows["partition"].eq(spec["partition"])]
    if spec.get("accident_class"):
        selected = selected.loc[selected["accident_class"].eq(spec["accident_class"])]
    return set(selected["sample_id"].astype(str))


def _write_overlap_audit(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    names = list(manifest["sets"])
    sets = {}
    for name in names:
        ids = _set_ids(manifest, name)
        sets[name] = {"sample_count": len(ids), "source": manifest["sets"][name]["source"], "sample_id_sha256": _sample_id_sha256(ids), "sample_ids": sorted(ids)}
    intersections = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(set(sets[left]["sample_ids"]) & set(sets[right]["sample_ids"]))
            intersections.append({"left": left, "right": right, "intersection_count": len(overlap), "sample_ids": overlap})
    release_name = "release_evaluation_test"
    protected = [name for name in names if name != release_name]
    zero_release_overlap = all(
        next(item for item in intersections if item["left"] == min(name, release_name, key=names.index) and item["right"] == max(name, release_name, key=names.index))["intersection_count"] == 0
        for name in protected
    )
    audit = {
        "protocol_version": manifest["protocol_version"],
        "sets": sets,
        "pairwise_intersections": intersections,
        "assertions": {"release_test_disjoint_from_all_fit_sets": bool(zero_release_overlap), "all_records_unique": len(_protocol_rows(manifest)) == _protocol_rows(manifest)["sample_id"].nunique()},
    }
    output = root / "results"
    _write_json(output / "17_v1_overlap_audit.json", audit)
    csv_rows = []
    for name, item in sets.items():
        csv_rows.append({"record_type": "set", "left": name, "right": "", "count": item["sample_count"], "source": item["source"], "sha256": item["sample_id_sha256"]})
    for item in intersections:
        csv_rows.append({"record_type": "intersection", "left": item["left"], "right": item["right"], "count": item["intersection_count"], "source": "", "sha256": _sample_id_sha256(item["sample_ids"])})
    pd.DataFrame(csv_rows).to_csv(output / "17_v1_overlap_audit.csv", index=False)
    if not audit["assertions"]["release_test_disjoint_from_all_fit_sets"]:
        raise AssertionError("Locked release test overlaps a fit/calibration set")
    return audit


def _fit_family_artifacts(root: Path, protocol_manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset, assignments = load_ood_context(project_root=root)
    mapping = family_mapping()
    class_to_family = dict(zip(mapping["class"], mapping["family"]))
    dataset = dataset.loc[dataset["window_s"].eq(WINDOW_S)].copy()
    dataset["family"] = dataset["accident_class"].map(class_to_family)
    protocol = _protocol_rows(protocol_manifest)
    frame = dataset.merge(protocol[["sample_id", "partition"]], on="sample_id", how="inner", validate="many_to_one")
    train = frame.loc[frame["partition"].eq("development_train")].copy()
    validation = frame.loc[frame["partition"].eq("development_validation")].copy()
    calibration = frame.loc[frame["partition"].eq("calibration")].copy()
    release = frame.loc[frame["partition"].eq("locked_release_test")].copy()
    features = strict_process_features()
    columns = feature_columns(features)
    labels = sorted(frame["family"].dropna().unique().tolist())
    base_for_save = classification_model_suite(RANDOM_SEED)[CLASSIFIER_MODEL]
    base_for_save.fit(train[columns], train["family"])
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_for_save), method="sigmoid", cv=None)
    calibrated.fit(calibration[columns], calibration["family"])
    validation_probabilities = _aligned_probabilities(
        calibrated, calibrated.predict_proba(calibration[columns]), labels
    )
    validation_pred = np.asarray(labels)[validation_probabilities.argmax(axis=1)]
    distance_model = _fit_distance(train[columns], train["family"], labels)
    calibration_distances = _distance_values(distance_model, calibration[columns], validation_pred)
    distance_threshold = float(np.quantile(calibration_distances, DISTANCE_TARGET_COVERAGE))
    probability_cutoff, conformal_score_quantile = _calibration_cutoff(
        validation_probabilities, calibration["family"].to_numpy(), labels, CONFORMAL_ALPHA
    )
    if train.empty or validation.empty or calibration.empty or release.empty:
        raise AssertionError("17.2 protocol has an empty classification partition")
    return (
        {
            "base_model": base_for_save,
            "calibrated_model": calibrated,
            "distance_model": distance_model,
        },
        {
            "labels": labels,
            "features": features,
            "feature_columns": columns,
            "window_s": WINDOW_S,
            "injection_s": INJECTION_S,
            "calibration_folds": 0,
            "calibration_status": "frozen_base_sigmoid_calibration_pool",
            "conformal_alpha": CONFORMAL_ALPHA,
            "conformal_score_quantile": conformal_score_quantile,
            "conformal_probability_cutoff": probability_cutoff,
            "distance_warning_threshold": distance_threshold,
            "distance_threshold_source": "protocol/calibration distance 70th percentile",
            "calibration_rows": int(len(calibration)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "release_rows": int(len(release)),
        },
    )


def _fit_loca_severity_artifact(root: Path, protocol_manifest: dict[str, Any]) -> dict[str, Any]:
    features = strict_process_features()
    dataset, _ = build_severity_features(root / "data" / "NuclearPowerPlantAccidentData" / "Operation_csv_data" / "LOCA", windows_s=(WINDOW_S,), features=features)
    dataset["protocol_sample_id"] = dataset["sample_id"].str.replace(r"^LOCA_0+", "LOCA_", regex=True)
    protocol = _protocol_rows(protocol_manifest)
    train_ids = _set_ids(protocol_manifest, "loca_severity_train")
    validation_ids = _set_ids(protocol_manifest, "loca_severity_validation")
    train_frame = dataset.loc[dataset["protocol_sample_id"].isin(train_ids)].copy()
    validation_frame = dataset.loc[dataset["protocol_sample_id"].isin(validation_ids)].copy()
    retained = [feature for feature in features if feature not in set(LOCA_EXCLUDED_BASES)]
    columns = feature_columns(retained)
    model = severity_model_suite(20260920)["random_forest"]
    model.set_params(n_jobs=1)
    model.fit(train_frame[columns], train_frame["severity"])
    validation_pred = model.predict(validation_frame[columns])
    return {
        "model": model,
        "columns": columns,
        "retained_features": retained,
        "excluded_bases": list(LOCA_EXCLUDED_BASES),
        "window_s": WINDOW_S,
        "validation_metrics": {
            "mae": float(mean_absolute_error(validation_frame["severity"], validation_pred)),
            "rmse": float(mean_squared_error(validation_frame["severity"], validation_pred) ** 0.5),
            "r2": float(r2_score(validation_frame["severity"], validation_pred)),
        },
        "train_sample_ids": sorted(train_ids),
        "validation_sample_ids": sorted(validation_ids),
        "release_sample_ids": sorted(_set_ids(protocol_manifest, "release_evaluation_test") & set(protocol.loc[protocol["accident_class"].eq("LOCA"), "sample_id"])),
        "exploratory": True,
        "source": "17.2 protocol/development_train and protocol/development_validation",
    }


def build_v1_artifacts(project_root: str | Path = PROJECT_ROOT, artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """Fit and persist v1 artifacts under the locked 17.2 trajectory protocol."""
    root = Path(project_root)
    output = Path(artifact_root)
    output.mkdir(parents=True, exist_ok=True)
    protocol_manifest = _make_protocol_manifest(root, output)
    overlap_audit = _write_overlap_audit(root, protocol_manifest)
    family_models, family_metadata = _fit_family_artifacts(root, protocol_manifest)
    loca_artifact = _fit_loca_severity_artifact(root, protocol_manifest)
    _save_joblib(output / "family_classifier.joblib", family_models["base_model"])
    _save_joblib(output / "probability_calibrator.joblib", family_models["calibrated_model"])
    _save_joblib(output / "distance_ood_model.joblib", family_models["distance_model"])
    _save_joblib(output / "loca_severity_model.joblib", loca_artifact)

    capability_gate = _capability_gate(root)
    mapping = family_mapping()
    feature_schema = {
        "required_time_column": "TIME",
        "strict_process_features": family_metadata["features"],
        "derived_feature_columns": family_metadata["feature_columns"],
        "summary_statistics": list(SUMMARY_STATISTICS),
        "window_s": WINDOW_S,
        "injection_s": INJECTION_S,
        "expected_sampling_interval_s": EXPECTED_INTERVAL_S,
        "sampling_interval_tolerance_s": INTERVAL_TOLERANCE_S,
        "minimum_window_s": WINDOW_S,
    }
    _write_json(output / "feature_extractor.json", feature_schema)
    _write_json(output / "family_mapping.json", mapping.to_dict(orient="records"))
    _write_json(output / "capability_matrix.json", {"family_gate": capability_gate, "source": "results/15_npp_guard_v1_capability_matrix.json"})
    _write_json(output / "conformal_calibration.json", {
        "method": "split_conformal_probability_threshold",
        "alpha": CONFORMAL_ALPHA,
        "labels": family_metadata["labels"],
        "probability_cutoff": family_metadata["conformal_probability_cutoff"],
        "nonconformity_score_quantile": family_metadata["conformal_score_quantile"],
        "calibration_rows": family_metadata["calibration_rows"],
        "calibration_split": "protocol/calibration only",
        "empirical_milestone_16_set_coverage": 0.689,
        "coverage_claim": "No 90% statistical coverage guarantee; empirical set coverage was 68.9% in milestone 16.",
    })
    _write_json(output / "policy.json", {
        "model_version": MODEL_VERSION,
        "decision_policy": "family classifier -> calibrated probability -> conformal alpha=.10 -> distance OOD check",
        "conformal_alpha": CONFORMAL_ALPHA,
        "distance_warning_threshold": family_metadata["distance_warning_threshold"],
        "distance_threshold_source": family_metadata["distance_threshold_source"],
        "status_rules": {
            "empty_conformal_set": "unknown",
            "singleton_and_distance_normal": "accepted",
            "singleton_and_distance_ood": "requires_review",
            "multi_family_set": "requires_review",
            "tier_B": "requires_review",
            "tier_C": "unknown",
        },
        "assessment_gating": {
            "loca_severity": "status == accepted and accident_family == primary_coolant_boundary_break and capability tier == Tier A",
            "otherwise": "not_run_due_to_family_gating",
        },
        "limitations": LIMITATIONS,
    })
    metadata = {
        "model_version": MODEL_VERSION,
        "random_seed": RANDOM_SEED,
        "family_classifier": {
            "model": CLASSIFIER_MODEL,
            "calibration": family_metadata["calibration_status"],
            "calibration_folds": family_metadata["calibration_folds"],
            "train_rows": family_metadata["train_rows"],
            "validation_rows": family_metadata["validation_rows"],
            "calibration_rows": family_metadata["calibration_rows"],
            "release_rows": family_metadata["release_rows"],
            "train_set": "family_classifier_train",
            "calibration_set": "probability_calibrator_fit",
        },
        "feature_schema": feature_schema,
        "feature_schema_hash": protocol_manifest["feature_schema_hash"],
        "family_mapping_version": "official_local_README_Table_1_mapping_v14",
        "family_mapping_hash": protocol_manifest["family_mapping_hash"],
        "protocol": {
            "version": PROTOCOL_VERSION,
            "split_manifest": "artifacts/v1/release_split_manifest.json",
            "split_manifest_sha256": _sha256(output / "release_split_manifest.json"),
            "overlap_audit": "results/17_v1_overlap_audit.json",
            "release_test_disjoint_from_fit_sets": overlap_audit["assertions"]["release_test_disjoint_from_all_fit_sets"],
        },
        "policy": json.loads((output / "policy.json").read_text(encoding="utf-8")),
        "capability_gate": capability_gate,
        "loca_severity": {
            "model": "random_forest",
            "window_s": WINDOW_S,
            "excluded_bases": list(LOCA_EXCLUDED_BASES),
            "validation_metrics": loca_artifact["validation_metrics"],
            "train_sample_count": len(loca_artifact["train_sample_ids"]),
            "validation_sample_count": len(loca_artifact["validation_sample_ids"]),
            "release_sample_count": len(loca_artifact["release_sample_ids"]),
            "train_set": "loca_severity_train",
            "validation_set": "loca_severity_validation",
            "exploratory": True,
        },
        "protection_time": {
            "status": "not_available_in_v1",
            "exploratory": True,
            "reason": "No methodologically valid chain from the 120 s severity estimate to the sparse protection-time landmark target was established.",
            "reported_limitations": {"test_mae_s": 86.82564467592593, "test_rmse_s": 338.73522013129934, "test_r2": 0.3302622978959968},
        },
        "inference_never_retrains": True,
        "artifact_files": {},
    }
    artifact_names = [
        "family_classifier.joblib",
        "probability_calibrator.joblib",
        "distance_ood_model.joblib",
        "loca_severity_model.joblib",
        "feature_extractor.json",
        "family_mapping.json",
        "capability_matrix.json",
        "conformal_calibration.json",
        "policy.json",
        "release_split_manifest.json",
    ]
    metadata["artifact_files"] = {
        name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)} for name in artifact_names
    }
    _write_json(output / "manifest.json", metadata)
    _load_artifacts_cached.cache_clear()
    return metadata


def _invalid_response(errors: list[str], data_quality: dict[str, Any] | None = None, model_version: str | None = None) -> dict[str, Any]:
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
        "data_quality": {"valid": False, "errors": errors, **(data_quality or {})},
        "limitations": LIMITATIONS,
        "model_version": model_version or MODEL_VERSION,
    }


def _validate_input(frame: pd.DataFrame, features: list[str]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    quality: dict[str, Any] = {"row_count": int(len(frame)), "extra_columns": sorted(set(frame.columns) - {"TIME", *features})}
    missing = [column for column in ["TIME", *features] if column not in frame.columns]
    if missing:
        return {"valid": False, "missing_columns": missing, **quality}, [f"missing_required_columns:{missing}"]
    if frame.columns.duplicated().any():
        errors.append("duplicate_column_names")
    time = pd.to_numeric(frame["TIME"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(time).all():
        errors.append("TIME_contains_NaN_or_Inf")
    if len(time) < 2:
        errors.append("at_least_two_rows_required")
        diffs = np.array([], dtype=float)
    else:
        diffs = np.diff(time)
        if np.any(diffs <= 0):
            errors.append("TIME_not_strictly_increasing_or_duplicate_time_points")
    if len(diffs):
        quality.update({
            "sampling_interval_median_s": float(np.median(diffs)),
            "sampling_interval_max_deviation_s": float(np.max(np.abs(diffs - EXPECTED_INTERVAL_S))),
        })
        if np.any(np.abs(diffs - EXPECTED_INTERVAL_S) > INTERVAL_TOLERANCE_S):
            errors.append("sampling_interval_not_approximately_10_s")
    if len(time) and np.isfinite(time).all():
        quality.update({"time_start_s": float(time[0]), "time_end_s": float(time[-1]), "window_s": float(time.max() - time.min())})
        if float(time.max()) < WINDOW_S:
            errors.append("window_shorter_than_120_s")
        if not np.any(time <= INJECTION_S):
            errors.append("baseline_at_or_before_injection_missing")
        if not np.any((time > INJECTION_S) & (time <= INJECTION_S + WINDOW_S)):
            errors.append("no_samples_in_120_s_window")
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"{feature}_contains_NaN_or_Inf")
    quality["valid"] = not errors
    return quality, errors


def _load_artifact_json(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_artifacts_cached(root_string: str) -> dict[str, Any]:
    root = Path(root_string)
    manifest = _load_artifact_json(root, "manifest.json")
    feature_schema = _load_artifact_json(root, "feature_extractor.json")
    policy = _load_artifact_json(root, "policy.json")
    conformal = _load_artifact_json(root, "conformal_calibration.json")
    capability = _load_artifact_json(root, "capability_matrix.json")["family_gate"]
    return {
        "manifest": manifest,
        "feature_schema": feature_schema,
        "policy": policy,
        "conformal": conformal,
        "capability": capability,
        "base_model": joblib.load(root / "family_classifier.joblib"),
        "calibrator": joblib.load(root / "probability_calibrator.joblib"),
        "distance": joblib.load(root / "distance_ood_model.joblib"),
        "loca_severity": joblib.load(root / "loca_severity_model.joblib"),
    }


def _status_from_policy(conformal_set: list[str], distance_warning: bool, gate: dict[str, Any]) -> str:
    if not conformal_set:
        return "unknown"
    if len(conformal_set) > 1 or distance_warning:
        return "requires_review"
    if gate.get("tier") == "Tier C":
        return "unknown"
    if gate.get("tier") == "Tier B":
        return "requires_review"
    return "accepted"


def _assessment(status: str, family: str, artifacts: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    gate = artifacts["capability"].get(family, {"tier": "Tier C", "status": "not-supported", "reason": "Family absent from the capability matrix."})
    output: dict[str, Any] = {
        "family_capability": {**gate, "research_output": gate["status"] == "supported"},
        "loca_severity": {
            "status": "not_run_due_to_family_gating",
            "estimate": None,
            "exploratory": True,
            "reason": "LOCA severity runs only for an accepted Tier-A LOCA family diagnosis.",
        },
        "protection_time": {
            "status": "not_available_in_v1",
            "exploratory": True,
            "reason": "No valid chained severity-to-protection-time inference path is fixed in v1.",
            "reported_test_mae_s": 86.82564467592593,
            "reported_test_r2": 0.3302622978959968,
        },
    }
    if status == "accepted" and family == "primary_coolant_boundary_break" and gate.get("tier") == "Tier A":
        model_info = artifacts["loca_severity"]
        vector = pd.DataFrame([features])[model_info["columns"]]
        estimate = float(model_info["model"].predict(vector)[0])
        output["loca_severity"] = {
            "status": "available_exploratory",
            "estimate": estimate,
            "unit": "% of 100 cm2 break area",
            "model": "random_forest",
            "window_s": WINDOW_S,
            "excluded_bases": model_info["excluded_bases"],
            "validation_mae": model_info["validation_metrics"]["mae"],
            "exploratory": True,
        }
    return output


def diagnose_dataframe(df: pd.DataFrame, artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """Diagnose one trajectory window using fixed v1 artifacts only."""
    root = Path(artifact_root)
    try:
        artifacts = _load_artifacts_cached(str(root.resolve()))
    except Exception as exc:
        return _invalid_response([f"artifact_load_error:{type(exc).__name__}:{exc}"])
    features = artifacts["feature_schema"]["strict_process_features"]
    quality, errors = _validate_input(df, features)
    if errors:
        return _invalid_response(errors, quality, artifacts["manifest"]["model_version"])
    try:
        extracted = extract_window_features(df, features, WINDOW_S, INJECTION_S)
        vector = pd.DataFrame([extracted])[artifacts["feature_schema"]["derived_feature_columns"]]
        labels = artifacts["conformal"]["labels"]
        probabilities = _aligned_probabilities(
            artifacts["calibrator"], artifacts["calibrator"].predict_proba(vector), labels
        )[0]
        top_index = int(np.argmax(probabilities))
        family = str(labels[top_index])
        confidence = float(probabilities[top_index])
        conformal_set = [label for label, probability in zip(labels, probabilities) if probability >= artifacts["conformal"]["probability_cutoff"] - 1e-12]
        distance_score = float(_distance_values(artifacts["distance"], vector, np.asarray([family]))[0])
        distance_warning = bool(distance_score > artifacts["policy"]["distance_warning_threshold"])
        gate = artifacts["capability"].get(family, {"tier": "Tier C", "status": "not-supported"})
        status = _status_from_policy(conformal_set, distance_warning, gate)
        assessment = _assessment(status, family, artifacts, extracted)
        if status == "unknown" and gate.get("tier") == "Tier C":
            assessment["family_capability"]["research_output"] = False
        quality.update({"valid": True, "used_columns": int(len(features)), "ignored_extra_columns": quality.get("extra_columns", [])})
        return {
            "status": status,
            "accident_family": family if status != "unknown" else None,
            "family_top1": family,
            "family_confidence": confidence,
            "conformal_set": conformal_set,
            "ood_warning": {
                "flag": distance_warning,
                "method": "class_conditional_ledoit_wolf_mahalanobis",
                "threshold": float(artifacts["policy"]["distance_warning_threshold"]),
                "reason": "distance_above_validation_threshold" if distance_warning else "distance_within_validation_threshold",
                "proxy_note": "This is a research severity-shift proxy, not a production OOD label.",
            },
            "distance_score": distance_score,
            "assessment": assessment,
            "severity_estimate": assessment["loca_severity"].get("estimate"),
            "severity_status": assessment["loca_severity"]["status"],
            "data_quality": quality,
            "limitations": LIMITATIONS,
            "model_version": artifacts["manifest"]["model_version"],
        }
    except Exception as exc:
        return _invalid_response([f"inference_error:{type(exc).__name__}:{exc}"], quality, artifacts["manifest"]["model_version"])


def diagnose_csv(path: str | Path, artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """Read a CSV and diagnose it using fixed v1 artifacts."""
    try:
        return diagnose_dataframe(pd.read_csv(path), artifact_root=artifact_root)
    except Exception as exc:
        return _invalid_response([f"csv_read_error:{type(exc).__name__}:{exc}"])


def _sample_frame(root: Path, severity: int) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "NuclearPowerPlantAccidentData" / "Operation_csv_data" / "LOCA" / f"{severity}.csv")


def verify_artifact_manifest(artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Any]:
    root = Path(artifact_root)
    manifest = _load_artifact_json(root, "manifest.json")
    checks = {}
    for name, expected in manifest["artifact_files"].items():
        path = root / name
        checks[name] = {
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else None,
            "expected_sha256": expected["sha256"],
            "passed": path.exists() and _sha256(path) == expected["sha256"],
        }
    return {"all_passed": all(item["passed"] for item in checks.values()), "checks": checks}


def _evaluation_candidates(root: Path) -> pd.DataFrame:
    split_manifest = _load_artifact_json(root / "artifacts" / "v1", "release_split_manifest.json")
    held_out = pd.DataFrame(split_manifest["records"])
    held_out = held_out.loc[
        held_out["partition"].eq("locked_release_test"),
        ["sample_id", "accident_class", "severity", "operation_csv", "partition"],
    ].copy()
    held_out["split_type"] = "locked_release"
    held_out["split_id"] = split_manifest["protocol_version"]
    inventory = pd.read_csv(root / "results" / "09_multi_accident_class_inventory.csv")
    candidates = held_out.merge(inventory[["sample_id", "numeric_case_id"]], on="sample_id", how="left", validate="one_to_one")
    candidates["severity"] = candidates["severity"].astype(float)
    candidates = candidates.sort_values(["sample_id", "split_id"]).reset_index(drop=True)
    if candidates["operation_csv"].isna().any():
        raise AssertionError("Evaluation examples are missing operation_csv paths")
    return candidates


def _compact_example(candidate: Any, response: dict[str, Any], input_kind: str = "full_trajectory") -> dict[str, Any]:
    assessment = response.get("assessment", {})
    loca = assessment.get("loca_severity")
    quality = response.get("data_quality", {})
    return {
        "sample_id": str(candidate.sample_id),
        "source": str(candidate.operation_csv).replace("\\", "/"),
        "input_kind": input_kind,
        "evaluation_cohort": {
            "split_type": str(candidate.split_type),
            "split_id": str(candidate.split_id),
            "partition": str(candidate.partition),
            "reference_accident_class": str(candidate.accident_class),
            "reference_severity": float(candidate.severity),
        },
        "status": response.get("status"),
        "accident_family": response.get("accident_family"),
        "family_confidence": response.get("family_confidence"),
        "conformal_set": response.get("conformal_set", []),
        "ood_warning": response.get("ood_warning"),
        "distance_score": response.get("distance_score"),
        "severity_gating": loca,
        "data_quality": {
            key: quality[key]
            for key in ("row_count", "window_s", "valid", "errors")
            if key in quality
        },
        "model_version": response.get("model_version"),
    }


def find_behavior_examples(project_root: str | Path = PROJECT_ROOT, artifact_root: str | Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """Search the locked release test cohort without using labels for inference."""
    root = Path(project_root)
    found: dict[str, Any] = {}
    candidates = _evaluation_candidates(root)
    for candidate in candidates.itertuples(index=False):
        frame = pd.read_csv(root / Path(candidate.operation_csv))
        response = diagnose_dataframe(frame, artifact_root)
        tier = response.get("assessment", {}).get("family_capability", {}).get("tier")
        family = response.get("accident_family")
        if response.get("status") == "accepted" and tier == "Tier A" and "accepted_tier_a" not in found:
            found["accepted_tier_a"] = _compact_example(candidate, response)
        if response.get("status") == "accepted" and tier == "Tier A" and family != "primary_coolant_boundary_break" and "accepted_non_loca_tier_a" not in found:
            found["accepted_non_loca_tier_a"] = _compact_example(candidate, response)
        if (
            response.get("status") == "accepted"
            and family == "primary_coolant_boundary_break"
            and response.get("assessment", {}).get("loca_severity", {}).get("status") == "available_exploratory"
            and "accepted_loca_severity" not in found
        ):
            found["accepted_loca_severity"] = _compact_example(candidate, response)
        if response.get("status") == "requires_review" and "requires_review" not in found:
            found["requires_review"] = _compact_example(candidate, response)
        if response.get("status") == "unknown" and "unknown" not in found:
            found["unknown"] = _compact_example(candidate, response)
        if {"accepted_tier_a", "accepted_loca_severity", "requires_review", "unknown"}.issubset(found):
            break

    for key in ("accepted_tier_a", "accepted_loca_severity", "requires_review", "unknown"):
        found.setdefault(
            key,
            {
                "status": "not_observed_in_current_evaluation_set",
                "reason": "No matching frozen-artifact inference result in locked release test cohort.",
                "evaluation_cohort_size": int(len(candidates)),
            },
        )
    found.setdefault(
        "accepted_non_loca_tier_a",
        {
            "status": "not_observed_in_current_evaluation_set",
            "reason": "No accepted non-LOCA Tier-A result was observed in the locked release test cohort.",
            "evaluation_cohort_size": int(len(candidates)),
        },
    )

    invalid_candidate = candidates.iloc[0]
    invalid_frame = pd.read_csv(root / Path(invalid_candidate.operation_csv))
    invalid_response = diagnose_dataframe(invalid_frame.loc[invalid_frame["TIME"] <= 50].copy(), artifact_root)
    found["invalid_input"] = _compact_example(invalid_candidate, invalid_response, "controlled_truncation_to_50s")
    found["search_protocol"] = {
        "source": "artifacts/v1/release_split_manifest.json locked_release_test",
        "cohort_size": int(len(candidates)),
        "uses_frozen_artifacts": True,
        "inference_does_not_use_reference_labels": True,
    }
    return found


def run_regression_tests(
    project_root: str | Path = PROJECT_ROOT,
    artifact_root: str | Path = ARTIFACT_ROOT,
    parity_result: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run deterministic API, protocol, gating, and locked-release checks."""
    root = Path(project_root)
    valid = _sample_frame(root, 1)
    ood = _sample_frame(root, 100)
    examples = find_behavior_examples(root, artifact_root)
    tests: list[dict[str, Any]] = []

    def check(name: str, passed: bool, response: dict[str, Any], detail: str) -> None:
        tests.append({"test": name, "passed": bool(passed), "status": response.get("status"), "detail": detail})

    response = diagnose_dataframe(valid, artifact_root)
    check("valid_120s_input", response["status"] != "invalid_input", response, "LOCA/1.csv full trajectory")
    missing = valid.drop(columns=[strict_process_features()[0]])
    response_missing = diagnose_dataframe(missing, artifact_root)
    check("missing_required_column", response_missing["status"] == "invalid_input", response_missing, "removed one strict process variable")
    short = valid.loc[valid["TIME"] <= 50].copy()
    response_short = diagnose_dataframe(short, artifact_root)
    check("window_shorter_than_120s", response_short["status"] == "invalid_input", response_short, "LOCA/1.csv truncated at 50 s")
    nan_frame = valid.copy()
    nan_frame.loc[nan_frame.index[1], strict_process_features()[1]] = np.nan
    response_nan = diagnose_dataframe(nan_frame, artifact_root)
    check("nan_inf_rejected", response_nan["status"] == "invalid_input", response_nan, "inserted NaN in a strict variable")
    inf_frame = valid.copy()
    inf_frame.loc[inf_frame.index[1], strict_process_features()[1]] = np.inf
    response_inf = diagnose_dataframe(inf_frame, artifact_root)
    check("inf_rejected", response_inf["status"] == "invalid_input", response_inf, "inserted Inf in a strict variable")
    extra = valid.copy()
    extra["EXTRA_UNUSED_COLUMN"] = 123.0
    response_extra = diagnose_dataframe(extra, artifact_root)
    check("extra_column_ignored", response_extra["status"] != "invalid_input", response_extra, "extra column is ignored and reported")
    duplicate = valid.iloc[[0, 1]].copy()
    duplicate.loc[duplicate.index[1], "TIME"] = duplicate.loc[duplicate.index[0], "TIME"]
    duplicate = pd.concat([duplicate, valid.iloc[2:]], ignore_index=True)
    response_duplicate = diagnose_dataframe(duplicate, artifact_root)
    check("duplicate_time_rejected", response_duplicate["status"] == "invalid_input", response_duplicate, "duplicated TIME point")
    response_loca = diagnose_dataframe(valid, artifact_root)
    loca_gate = response_loca.get("assessment", {}).get("family_capability", {})
    expected_loca_run = response_loca.get("status") == "accepted" and response_loca.get("family_top1") == "primary_coolant_boundary_break" and loca_gate.get("tier") == "Tier A"
    check("loca_severity_gate_matches_policy", response_loca.get("severity_status") == ("available_exploratory" if expected_loca_run else "not_run_due_to_family_gating"), response_loca, "LOCA severity only runs for accepted LOCA Tier-A decisions")
    response_ood = diagnose_dataframe(ood, artifact_root)
    check("ood_sample_runs", response_ood["status"] in {"accepted", "requires_review", "unknown"}, response_ood, "real LOCA/100.csv severity-shift proxy")
    review = examples["requires_review"]
    check("requires_review_does_not_run_loca_severity", review.get("status") == "not_observed_in_current_evaluation_set" or review.get("severity_gating", {}).get("status") == "not_run_due_to_family_gating", review, "locked release behavior example")
    non_loca = examples["accepted_non_loca_tier_a"]
    check("accepted_non_loca_does_not_run_loca_severity", non_loca.get("status") == "not_observed_in_current_evaluation_set" or non_loca.get("severity_gating", {}).get("status") == "not_run_due_to_family_gating", non_loca, "locked release behavior example")
    loca_example = examples["accepted_loca_severity"]
    check("accepted_loca_runs_severity", loca_example.get("status") == "not_observed_in_current_evaluation_set" or (loca_example.get("status") == "accepted" and loca_example.get("severity_gating", {}).get("status") == "available_exploratory"), loca_example, "locked release accepted LOCA example")
    check("invalid_input_does_not_run_assessment", examples["invalid_input"].get("status") == "invalid_input" and examples["invalid_input"].get("severity_gating") is None, examples["invalid_input"], "controlled 50 s truncation")
    repeat_a = diagnose_dataframe(valid, artifact_root)
    repeat_b = diagnose_dataframe(valid, artifact_root)
    check("repeat_inference_consistent", json.dumps(repeat_a, sort_keys=True, default=_json_default) == json.dumps(repeat_b, sort_keys=True, default=_json_default), repeat_a, "same input and fixed artifacts")
    manifest_check = verify_artifact_manifest(artifact_root)
    check("artifact_sha256_verification", manifest_check["all_passed"], {"status": "verified" if manifest_check["all_passed"] else "failed"}, "manifest-listed artifacts")
    overlap_audit = json.loads((root / "results" / "17_v1_overlap_audit.json").read_text(encoding="utf-8"))
    check("zero_overlap", overlap_audit["assertions"]["release_test_disjoint_from_all_fit_sets"], {"status": "passed" if overlap_audit["assertions"]["release_test_disjoint_from_all_fit_sets"] else "failed"}, "locked release test versus all fit/calibration sets")
    if parity_result is None:
        from .parity import evaluate_pipeline_parity
        parity_result = evaluate_pipeline_parity(root, artifact_root)
    check("exact_pipeline_parity", bool(parity_result["overall_pass"]), {"status": "passed" if parity_result["overall_pass"] else "failed"}, parity_result["gate_reason"])
    from .parity import evaluate_release_benchmark
    benchmark = evaluate_release_benchmark(root, artifact_root)
    check("locked_release_benchmark", int(benchmark["protocol"]["release_sample_count"]) > 0, {"status": "generated"}, "locked release benchmark generated without refit")
    result = pd.DataFrame(tests)
    return result, examples
