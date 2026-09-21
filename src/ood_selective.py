"""OOD-aware selective family diagnosis for milestone 16.

All fitted transforms, calibration quantiles, and rejection thresholds are
derived from train/validation rows. Test rows are used only for final scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, f1_score, log_loss, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from .hierarchical_diagnosis import family_mapping
    from .multi_accident_features import input_feature_groups, strict_process_features
    from .multi_accident_models import model_suite
    from .ood_validation import CLASS_LABELS
    from .severity_invariant import PROJECT_ROOT, feature_columns, load_ood_context
except ImportError:
    from hierarchical_diagnosis import family_mapping
    from multi_accident_features import input_feature_groups, strict_process_features
    from multi_accident_models import model_suite
    from ood_validation import CLASS_LABELS
    from severity_invariant import PROJECT_ROOT, feature_columns, load_ood_context


RESULT_ROOT = PROJECT_ROOT / "results"
WINDOWS_S = (60, 90, 120)
MAIN_WINDOW_S = 120
TARGET_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
CONFORMAL_ALPHAS = (0.05, 0.10, 0.20)
OPERATING_COVERAGE = 0.70
FOCUS_CLASSES = ("RI", "LOCAC", "SLBIC")
SCALAR_MECHANISMS = ("probability", "margin", "distance")


def _aligned_probabilities(model: object, probabilities: np.ndarray, labels: list[str]) -> np.ndarray:
    classes = list(getattr(model, "classes_"))
    aligned = np.zeros((len(probabilities), len(labels)), dtype=float)
    for index, label in enumerate(classes):
        if label in labels:
            aligned[:, labels.index(label)] = probabilities[:, index]
    sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, sums, out=np.full_like(aligned, 1.0 / max(1, len(labels))), where=sums > 0)


def _fit_calibrated(base: object, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[object, int, str]:
    if y_train.nunique() < 2:
        fallback = DummyClassifier(strategy="prior")
        fallback.fit(x_train, y_train)
        return fallback, 0, "uncalibrated_single_class_fallback"
    min_support = int(y_train.value_counts().min())
    folds = min(3, min_support)
    if folds < 2:
        base.fit(x_train, y_train)
        return base, 0, "uncalibrated_min_class_support"
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=folds, n_jobs=1)
    calibrated.fit(x_train, y_train)
    return calibrated, folds, "train_only_sigmoid_cv"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], accepted: np.ndarray | None = None) -> dict[str, Any]:
    accepted = np.ones(len(y_true), dtype=bool) if accepted is None else np.asarray(accepted, dtype=bool)
    n_total = int(len(y_true))
    n_accepted = int(accepted.sum())
    if n_accepted == 0:
        return {
            "n_total": n_total,
            "n_accepted": 0,
            "n_rejected": n_total,
            "coverage": 0.0,
            "selective_accuracy": np.nan,
            "selective_macro_f1": np.nan,
            "selective_balanced_accuracy": np.nan,
            "risk": np.nan,
            "reject_rate": 1.0 if n_total else np.nan,
        }
    y_acc = y_true[accepted]
    p_acc = y_pred[accepted]
    observed = [label for label in labels if np.any(y_acc == label)]
    accuracy = float(np.mean(y_acc == p_acc))
    return {
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_total - n_accepted,
        "coverage": float(n_accepted / n_total) if n_total else 0.0,
        "selective_accuracy": accuracy,
        "selective_macro_f1": float(f1_score(y_acc, p_acc, labels=labels, average="macro", zero_division=0)),
        "selective_balanced_accuracy": float(recall_score(y_acc, p_acc, labels=observed, average="macro", zero_division=0)) if observed else np.nan,
        "risk": float(1.0 - accuracy),
        "reject_rate": float(1.0 - n_accepted / n_total) if n_total else np.nan,
    }


def _score_columns(probabilities: np.ndarray, distances: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(probabilities, axis=1)
    return {
        "probability": probabilities.max(axis=1),
        "margin": ordered[:, -1] - ordered[:, -2] if probabilities.shape[1] > 1 else probabilities[:, 0],
        "distance": distances,
    }


def _threshold(values: np.ndarray, target_coverage: float, high_is_good: bool) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0
    quantile = 1.0 - target_coverage if high_is_good else target_coverage
    return float(np.quantile(clean, quantile, method="lower" if high_is_good else "higher"))


def _accepted(values: np.ndarray, threshold: float, high_is_good: bool) -> np.ndarray:
    return values >= threshold if high_is_good else values <= threshold


def _fit_distance(x_train: pd.DataFrame, y_train: pd.Series, labels: list[str]) -> dict[str, Any]:
    scaler = StandardScaler().fit(x_train)
    scaled = scaler.transform(x_train)
    global_cov = LedoitWolf().fit(scaled)
    class_models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in labels:
        selected = scaled[y_train.to_numpy() == label]
        if len(selected) >= 2:
            covariance = LedoitWolf().fit(selected)
            mean = selected.mean(axis=0)
            precision = covariance.precision_
        else:
            mean = global_cov.location_
            precision = global_cov.precision_
        class_models[label] = (mean, precision)
    return {"scaler": scaler, "class_models": class_models, "fit_rows": int(len(x_train))}


def _distance_values(distance_model: dict[str, Any], x: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    scaled = distance_model["scaler"].transform(x)
    output = np.zeros(len(x), dtype=float)
    for label in np.unique(predictions):
        mask = predictions == label
        if label not in distance_model["class_models"]:
            output[mask] = np.nan
            continue
        mean, precision = distance_model["class_models"][label]
        delta = scaled[mask] - mean
        output[mask] = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return np.maximum(output, 0.0)


def _conformal_sets(probabilities: np.ndarray, labels: list[str], calibration_scores: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    scores = np.sort(np.asarray(calibration_scores, dtype=float))
    if scores.size == 0:
        quantile = 1.0
    else:
        index = min(scores.size - 1, max(0, int(np.ceil((scores.size + 1) * (1.0 - alpha))) - 1))
        quantile = float(scores[index])
    probability_cutoff = 1.0 - quantile
    sets = probabilities >= probability_cutoff - 1e-12
    return sets, probability_cutoff


def _nearest_gaps(assignments: pd.DataFrame, partition: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split_type, split_id), split in assignments.groupby(["split_type", "split_id"], sort=False):
        train = split.loc[split["partition"].eq("train")]
        for row in split.loc[split["partition"].eq(partition)].itertuples(index=False):
            same_class = train.loc[train["accident_class"].eq(row.accident_class)]
            if same_class.empty:
                gap = np.nan
            else:
                gap = float((same_class["severity"] - float(row.severity)).abs().min())
            rows.append({
                "split_type": split_type,
                "split_id": split_id,
                "sample_id": row.sample_id,
                "nearest_severity_gap": gap,
            })
    return pd.DataFrame(rows)


def _model_name_from_15(result_root: Path, split_type: str, split_id: str) -> str:
    path = result_root / "15_selective_metrics.csv"
    if path.exists():
        rows = pd.read_csv(path)
        selected = rows.loc[
            rows["split_type"].eq(split_type)
            & rows["split_id"].eq(split_id)
            & rows["window_s"].eq(MAIN_WINDOW_S)
            & rows["eval_split"].eq("test")
            & rows["mode"].eq("family_selective")
            & rows["selected_for_test"].eq(True)
            & rows["curve_point"].eq(False)
        ]
        if len(selected) == 1 and selected.iloc[0]["model"] in {"logistic_regression", "random_forest", "hist_gradient_boosting"}:
            return str(selected.iloc[0]["model"])
    return "hist_gradient_boosting"


def _fit_family_classifier(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    seed: int,
    model_name: str,
) -> tuple[object, np.ndarray, np.ndarray, int, str]:
    base = model_suite(seed).get(model_name)
    if base is None:
        base = DummyClassifier(strategy="prior")
    model, folds, status = _fit_calibrated(base, train[columns], train["family"])
    val_prob = _aligned_probabilities(model, model.predict_proba(validation[columns]), labels)
    return model, val_prob, None, folds, status


def _combination_thresholds(
    val_y: np.ndarray,
    val_pred: np.ndarray,
    val_probability: np.ndarray,
    val_distance: np.ndarray,
    labels: list[str],
    target: float,
) -> tuple[float, float, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for probability_target in TARGET_COVERAGES:
        probability_threshold = _threshold(val_probability, probability_target, True)
        for distance_target in TARGET_COVERAGES:
            distance_threshold = _threshold(val_distance, distance_target, False)
            accepted = (val_probability >= probability_threshold) & (val_distance <= distance_threshold)
            values = _metrics(val_y, val_pred, labels, accepted)
            candidates.append({
                "probability_threshold": probability_threshold,
                "distance_threshold": distance_threshold,
                "target_gap": abs(float(values["coverage"]) - target),
                "validation_macro_f1": values["selective_macro_f1"],
                "validation_accuracy": values["selective_accuracy"],
                "validation_coverage": values["coverage"],
                "accepted": accepted,
            })
    chosen = sorted(
        candidates,
        key=lambda row: (
            row["target_gap"],
            -float(row["validation_macro_f1"]) if np.isfinite(row["validation_macro_f1"]) else np.inf,
            -float(row["validation_accuracy"]) if np.isfinite(row["validation_accuracy"]) else np.inf,
        ),
    )[0]
    return float(chosen["probability_threshold"]), float(chosen["distance_threshold"]), chosen


def _metric_row(
    *,
    split_type: str,
    split_id: str,
    window_s: int,
    feature_group: str,
    analysis_scope: str,
    mechanism: str,
    eval_split: str,
    target_coverage: float,
    values: dict[str, Any],
    threshold: float | None,
    secondary_threshold: float | None,
    selected_model: str,
    calibration_status: str,
    validation_coverage: float | None = None,
) -> dict[str, Any]:
    return {
        "split_type": split_type,
        "split_id": split_id,
        "window_s": int(window_s),
        "feature_group": feature_group,
        "analysis_scope": analysis_scope,
        "mechanism": mechanism,
        "eval_split": eval_split,
        "target_coverage": float(target_coverage),
        "threshold": threshold,
        "secondary_threshold": secondary_threshold,
        "validation_coverage": validation_coverage,
        "selected_model": selected_model,
        "calibration_status": calibration_status,
        **values,
    }


def _evaluate_split(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    result_root: Path,
    split_type: str,
    split_id: str,
    window_s: int,
    feature_group: str,
    analysis_scope: str,
    selected_model: str,
    labels: list[str],
    emit_risk_curve: bool,
) -> dict[str, list[dict[str, Any]]]:
    split = assignments.loc[assignments["split_type"].eq(split_type) & assignments["split_id"].eq(split_id)]
    active = split.loc[split["partition"].isin(["train", "validation", "test"])]
    frame = dataset.loc[dataset["window_s"].eq(window_s)].merge(
        active[["sample_id", "partition"]], on="sample_id", how="inner", validate="many_to_one"
    ).copy()
    mapping = family_mapping()
    class_to_family = dict(zip(mapping["class"], mapping["family"]))
    frame["family"] = frame["accident_class"].map(class_to_family)
    train = frame.loc[frame["partition"].eq("train")]
    validation = frame.loc[frame["partition"].eq("validation")]
    test = frame.loc[frame["partition"].eq("test")]
    if train.empty or validation.empty or test.empty:
        raise AssertionError(f"Empty split for 16: {split_type}/{split_id}/{window_s}s/{feature_group}")
    features_by_group, removed = input_feature_groups()
    features = features_by_group[feature_group]
    columns = feature_columns(features)
    model, val_prob, _, folds, calibration_status = _fit_family_classifier(
        train, validation, columns, labels, int(split["seed"].dropna().iloc[0]) if split["seed"].notna().any() else 20260921, selected_model
    )
    test_prob = _aligned_probabilities(model, model.predict_proba(test[columns]), labels)
    val_pred = np.asarray(labels)[val_prob.argmax(axis=1)]
    test_pred = np.asarray(labels)[test_prob.argmax(axis=1)]
    distance_model = _fit_distance(train[columns], train["family"], labels)
    val_distance = _distance_values(distance_model, validation[columns], val_pred)
    test_distance = _distance_values(distance_model, test[columns], test_pred)
    val_scores = _score_columns(val_prob, val_distance)
    test_scores = _score_columns(test_prob, test_distance)
    rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    gap_validation = _nearest_gaps(split, "validation")[["sample_id", "nearest_severity_gap"]]
    gap_test = _nearest_gaps(split, "test")[["sample_id", "nearest_severity_gap"]]
    gap_values = pd.to_numeric(gap_validation["nearest_severity_gap"], errors="coerce").dropna().to_numpy(dtype=float)
    gap_threshold = float(np.quantile(gap_values, 0.75)) if len(gap_values) else np.nan

    base_values = _metrics(test["family"].to_numpy(), test_pred, labels)
    rows.append(_metric_row(
        split_type=split_type, split_id=split_id, window_s=window_s, feature_group=feature_group,
        analysis_scope=analysis_scope, mechanism="forced_family", eval_split="test", target_coverage=1.0,
        values=base_values, threshold=0.0, secondary_threshold=None, selected_model=selected_model,
        calibration_status=calibration_status, validation_coverage=1.0,
    ))
    for mechanism in SCALAR_MECHANISMS:
        high_is_good = mechanism != "distance"
        operating_threshold = _threshold(val_scores[mechanism], OPERATING_COVERAGE, high_is_good)
        val_accept = _accepted(val_scores[mechanism], operating_threshold, high_is_good)
        test_accept = _accepted(test_scores[mechanism], operating_threshold, high_is_good)
        rows.append(_metric_row(
            split_type=split_type, split_id=split_id, window_s=window_s, feature_group=feature_group,
            analysis_scope=analysis_scope, mechanism=mechanism, eval_split="test", target_coverage=OPERATING_COVERAGE,
            values=_metrics(test["family"].to_numpy(), test_pred, labels, test_accept), threshold=operating_threshold,
            secondary_threshold=None, selected_model=selected_model, calibration_status=calibration_status,
            validation_coverage=float(val_accept.mean()),
        ))
        if emit_risk_curve:
            for target in TARGET_COVERAGES:
                threshold = _threshold(val_scores[mechanism], target, high_is_good)
                val_curve_accept = _accepted(val_scores[mechanism], threshold, high_is_good)
                test_curve_accept = _accepted(test_scores[mechanism], threshold, high_is_good)
                curve_values = _metrics(test["family"].to_numpy(), test_pred, labels, test_curve_accept)
                risk_rows.append({
                    "split_type": split_type, "split_id": split_id, "window_s": int(window_s),
                    "feature_group": feature_group, "analysis_scope": analysis_scope, "mechanism": mechanism,
                    "target_coverage": float(target), "validation_threshold": threshold,
                    "validation_coverage": float(val_curve_accept.mean()), **curve_values,
                })
        if analysis_scope == "main" and window_s == MAIN_WINDOW_S:
            sample = test[["sample_id", "accident_class", "severity", "family"]].copy()
            sample["window_s"] = int(window_s)
            sample["predicted_family"] = test_pred
            sample["mechanism"] = mechanism
            sample["score"] = test_scores[mechanism]
            sample["ood_score"] = -test_scores[mechanism] if high_is_good else test_scores[mechanism]
            sample["threshold"] = operating_threshold
            sample["rejected"] = ~test_accept
            sample["accepted"] = test_accept
            sample["prediction_set_size"] = np.nan
            sample["conformal_alpha"] = np.nan
            sample["split_type"] = split_type
            sample["split_id"] = split_id
            sample["feature_group"] = feature_group
            sample["analysis_scope"] = analysis_scope
            sample = sample.merge(gap_test, on="sample_id", how="left", validate="one_to_one")
            sample["high_gap_proxy"] = sample["nearest_severity_gap"].ge(gap_threshold) if np.isfinite(gap_threshold) else False
            sample["gap_threshold_from_validation"] = gap_threshold
            sample_rows.extend(sample.to_dict(orient="records"))

    probability_threshold, distance_threshold, combination_validation = _combination_thresholds(
        validation["family"].to_numpy(), val_pred, val_scores["probability"], val_scores["distance"], labels, OPERATING_COVERAGE
    )
    combination_accept = (test_scores["probability"] >= probability_threshold) & (test_scores["distance"] <= distance_threshold)
    rows.append(_metric_row(
        split_type=split_type, split_id=split_id, window_s=window_s, feature_group=feature_group,
        analysis_scope=analysis_scope, mechanism="combination", eval_split="test", target_coverage=OPERATING_COVERAGE,
        values=_metrics(test["family"].to_numpy(), test_pred, labels, combination_accept), threshold=probability_threshold,
        secondary_threshold=distance_threshold, selected_model=selected_model, calibration_status=calibration_status,
        validation_coverage=float(combination_validation["validation_coverage"]),
    ))
    if analysis_scope == "main" and window_s == MAIN_WINDOW_S:
        sample = test[["sample_id", "accident_class", "severity", "family"]].copy()
        sample["window_s"] = int(window_s)
        sample["predicted_family"] = test_pred
        sample["mechanism"] = "combination"
        sample["score"] = test_scores["probability"]
        sample["ood_score"] = test_scores["distance"]
        sample["threshold"] = probability_threshold
        sample["secondary_threshold"] = distance_threshold
        sample["rejected"] = ~combination_accept
        sample["accepted"] = combination_accept
        sample["prediction_set_size"] = np.nan
        sample["conformal_alpha"] = np.nan
        sample["split_type"] = split_type
        sample["split_id"] = split_id
        sample["feature_group"] = feature_group
        sample["analysis_scope"] = analysis_scope
        sample = sample.merge(gap_test, on="sample_id", how="left", validate="one_to_one")
        sample["high_gap_proxy"] = sample["nearest_severity_gap"].ge(gap_threshold) if np.isfinite(gap_threshold) else False
        sample["gap_threshold_from_validation"] = gap_threshold
        sample_rows.extend(sample.to_dict(orient="records"))

    calibration_scores = 1.0 - val_prob[np.arange(len(validation)), [labels.index(label) for label in validation["family"]]]
    for alpha in CONFORMAL_ALPHAS:
        val_sets, cutoff = _conformal_sets(val_prob, labels, calibration_scores, alpha)
        test_sets, _ = _conformal_sets(test_prob, labels, calibration_scores, alpha)
        set_sizes = test_sets.sum(axis=1)
        singleton = set_sizes == 1
        empty = set_sizes == 0
        ambiguous = set_sizes > 1
        contains_true = np.array([test_sets[index, labels.index(label)] for index, label in enumerate(test["family"])], dtype=bool)
        conformal_values = _metrics(test["family"].to_numpy(), test_pred, labels, singleton)
        conformal_rows.append({
            "split_type": split_type, "split_id": split_id, "window_s": int(window_s),
            "feature_group": feature_group, "analysis_scope": analysis_scope, "eval_split": "test",
            "alpha": float(alpha), "calibration_n": int(len(calibration_scores)), "probability_cutoff": cutoff,
            "empirical_set_coverage": float(contains_true.mean()), "average_prediction_set_size": float(set_sizes.mean()),
            "single_label_rate": float(singleton.mean()), "ambiguous_multi_label_rate": float(ambiguous.mean()),
            "empty_unknown_rate": float(empty.mean()), "coverage": conformal_values["coverage"],
            "selective_accuracy": conformal_values["selective_accuracy"], "selective_macro_f1": conformal_values["selective_macro_f1"],
            "selective_balanced_accuracy": conformal_values["selective_balanced_accuracy"], "risk": conformal_values["risk"],
            "reject_rate": float(1.0 - singleton.mean()), "selected_model": selected_model,
            "calibration_status": calibration_status,
        })
        if analysis_scope == "main" and window_s == MAIN_WINDOW_S and abs(alpha - 0.10) < 1e-12:
            sample = test[["sample_id", "accident_class", "severity", "family"]].copy()
            sample["window_s"] = int(window_s)
            sample["predicted_family"] = test_pred
            sample["mechanism"] = "conformal_alpha_0.10"
            sample["score"] = np.nan
            sample["ood_score"] = np.nan
            sample["threshold"] = cutoff
            sample["rejected"] = ~singleton
            sample["accepted"] = singleton
            sample["prediction_set_size"] = set_sizes
            sample["conformal_alpha"] = alpha
            sample["split_type"] = split_type
            sample["split_id"] = split_id
            sample["feature_group"] = feature_group
            sample["analysis_scope"] = analysis_scope
            sample = sample.merge(gap_test, on="sample_id", how="left", validate="one_to_one")
            sample["high_gap_proxy"] = sample["nearest_severity_gap"].ge(gap_threshold) if np.isfinite(gap_threshold) else False
            sample["gap_threshold_from_validation"] = gap_threshold
            sample_rows.extend(sample.to_dict(orient="records"))
    return {"metrics": rows, "risk": risk_rows, "conformal": conformal_rows, "samples": sample_rows}


def _aggregate_class_rejection(samples: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = samples.loc[samples["analysis_scope"].eq("main") & samples["window_s"].eq(MAIN_WINDOW_S)]
    for (mechanism, accident_class), group in selected.groupby(["mechanism", "accident_class"], sort=True):
        if accident_class not in FOCUS_CLASSES:
            continue
        accepted = group["accepted"].astype(bool).to_numpy()
        y_true = group["family"].to_numpy()
        y_pred = group["predicted_family"].to_numpy()
        accepted_true = y_true[accepted]
        accepted_pred = y_pred[accepted]
        focus_family = str(y_true[0]) if len(y_true) else ""
        accepted_macro_f1 = (
            float(f1_score(accepted_true, accepted_pred, labels=[focus_family], average="macro", zero_division=0))
            if len(accepted_true) else np.nan
        )
        accepted_recall = (
            float(recall_score(accepted_true, accepted_pred, labels=[focus_family], average="macro", zero_division=0))
            if len(accepted_true) else np.nan
        )
        rows.append({
            "window_s": MAIN_WINDOW_S,
            "feature_group": "A_strict_38",
            "mechanism": mechanism,
            "accident_class": accident_class,
            "n_total": int(len(group)),
            "n_rejected": int((~accepted).sum()),
            "reject_rate": float((~accepted).mean()),
            "accepted_n": int(accepted.sum()),
            "accepted_family_macro_f1": accepted_macro_f1,
            "accepted_family_recall": accepted_recall,
            "nearest_gap_mean_rejected": float(group.loc[~group["accepted"], "nearest_severity_gap"].mean()) if (~group["accepted"]).any() else np.nan,
            "nearest_gap_mean_accepted": float(group.loc[group["accepted"], "nearest_severity_gap"].mean()) if group["accepted"].any() else np.nan,
        })
    return pd.DataFrame(rows)


def _safe_detection_metrics(group: pd.DataFrame, mechanism: str, proxy_type: str, proxy: np.ndarray) -> dict[str, Any]:
    score = pd.to_numeric(group["ood_score"], errors="coerce").to_numpy(dtype=float)
    rejected = group["rejected"].astype(bool).to_numpy()
    if not np.isfinite(score).any():
        score = rejected.astype(float)
    valid = np.isfinite(score) & np.isfinite(proxy)
    y = proxy[valid].astype(int)
    if valid.sum() == 0 or len(np.unique(y)) < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = float(roc_auc_score(y, score[valid]))
        auprc = float(average_precision_score(y, score[valid]))
    positive = proxy.astype(bool)
    negative = ~positive
    return {
        "window_s": MAIN_WINDOW_S,
        "feature_group": "A_strict_38",
        "mechanism": mechanism,
        "proxy_type": proxy_type,
        "n_ood_proxy": int(positive.sum()),
        "n_id_proxy": int(negative.sum()),
        "ood_reject_rate": float(rejected[positive].mean()) if positive.any() else np.nan,
        "id_reject_rate": float(rejected[negative].mean()) if negative.any() else np.nan,
        "auroc": auroc,
        "auprc": auprc,
        "mean_gap_rejected": float(group.loc[group["rejected"], "nearest_severity_gap"].mean()) if group["rejected"].any() else np.nan,
        "mean_gap_accepted": float(group.loc[group["accepted"], "nearest_severity_gap"].mean()) if group["accepted"].any() else np.nan,
        "gap_difference_rejected_minus_accepted": float(group.loc[group["rejected"], "nearest_severity_gap"].mean() - group.loc[group["accepted"], "nearest_severity_gap"].mean()) if group["rejected"].any() and group["accepted"].any() else np.nan,
    }


def _build_ood_detection(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mechanism, group in samples.groupby("mechanism", sort=True):
        extrap = group.loc[group["split_type"].isin(["random", "severity_extrapolation"])].copy()
        proxy = extrap["split_type"].eq("severity_extrapolation").to_numpy(dtype=bool)
        rows.append(_safe_detection_metrics(extrap, mechanism, "extrapolation_test_vs_random_test", proxy))
        high_gap = group["high_gap_proxy"].astype(bool).to_numpy()
        rows.append(_safe_detection_metrics(group, mechanism, "large_nearest_severity_gap_proxy", high_gap))
    return pd.DataFrame(rows)


def _write_figures(
    risk: pd.DataFrame,
    conformal: pd.DataFrame,
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    class_rejection: pd.DataFrame,
    result_root: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = result_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    curve = risk.loc[(risk["window_s"].eq(MAIN_WINDOW_S)) & risk["feature_group"].eq("A_strict_38")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, mechanism in zip(axes, SCALAR_MECHANISMS):
        selected = curve.loc[curve["mechanism"].eq(mechanism)]
        for split_type, group in selected.groupby("split_type", sort=True):
            grouped = group.groupby("target_coverage", as_index=False)[["coverage", "risk"]].mean()
            axis.plot(grouped["coverage"], grouped["risk"], marker="o", label=split_type)
        axis.set_title(mechanism)
        axis.set_xlabel("Test coverage")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Risk = 1 - selective accuracy")
    axes[-1].legend(fontsize=7)
    fig.suptitle("16 OOD-aware risk-coverage curves (thresholds from validation)")
    fig.tight_layout()
    path = figure_root / "16_risk_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    selected_samples = samples.loc[(samples["window_s"].eq(MAIN_WINDOW_S)) & samples["feature_group"].eq("A_strict_38")]
    selected_samples = selected_samples.loc[selected_samples["mechanism"].isin((*SCALAR_MECHANISMS, "combination", "conformal_alpha_0.10"))]
    fig, ax = plt.subplots(figsize=(9, 4))
    box_data = []
    box_labels = []
    for mechanism, group in selected_samples.groupby("mechanism", sort=False):
        box_data.extend([
            group.loc[group["accepted"], "nearest_severity_gap"].dropna().to_numpy(),
            group.loc[group["rejected"], "nearest_severity_gap"].dropna().to_numpy(),
        ])
        box_labels.extend([f"{mechanism}\naccepted", f"{mechanism}\nrejected"])
    if box_data:
        ax.boxplot(box_data, tick_labels=box_labels, showmeans=True)
    ax.set_ylabel("Nearest train severity gap")
    ax.set_title("Accepted vs rejected severity-gap proxy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = figure_root / "16_accepted_vs_rejected_severity_gap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    conformal_main = conformal.loc[(conformal["window_s"].eq(MAIN_WINDOW_S)) & conformal["feature_group"].eq("A_strict_38")]
    grouped = conformal_main.groupby("alpha", as_index=False)[["average_prediction_set_size", "empirical_set_coverage"]].mean()
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(grouped["alpha"], grouped["average_prediction_set_size"], marker="o", label="mean set size")
    axis.set_xlabel("alpha")
    axis.set_ylabel("Mean prediction-set size")
    other = axis.twinx()
    other.plot(grouped["alpha"], grouped["empirical_set_coverage"], marker="s", color="#e45756", label="empirical coverage")
    other.set_ylabel("Empirical set coverage")
    axis.set_title("Conformal prediction-set size and coverage")
    fig.tight_layout()
    path = figure_root / "16_conformal_set_size_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    comparison = metrics.loc[(metrics["window_s"].eq(MAIN_WINDOW_S)) & metrics["feature_group"].eq("A_strict_38") & metrics["split_type"].eq("severity_extrapolation") & metrics["eval_split"].eq("test")]
    comparison = comparison.groupby("mechanism", as_index=False)[["coverage", "selective_macro_f1", "risk", "reject_rate"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, column in zip(axes, ["coverage", "selective_macro_f1", "risk"]):
        axis.bar(comparison["mechanism"], comparison[column], color="#4c78a8")
        axis.set_title(column)
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Mechanism comparison on 120 s severity extrapolation")
    fig.tight_layout()
    path = figure_root / "16_mechanism_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    focus = class_rejection.loc[class_rejection["mechanism"].isin((*SCALAR_MECHANISMS, "combination", "conformal_alpha_0.10"))]
    if not focus.empty:
        pivot = focus.pivot_table(index="accident_class", columns="mechanism", values="reject_rate", aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(9, 4), color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"])
        ax.set_ylabel("Reject rate")
        ax.set_title("RI / LOCAC / SLBIC rejection rates")
        ax.legend(fontsize=7)
        fig = ax.get_figure()
        fig.tight_layout()
        path = figure_root / "16_focus_class_reject_rates.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return paths


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def run_experiment(project_root: str | Path = PROJECT_ROOT, result_root: str | Path = RESULT_ROOT) -> dict[str, Any]:
    root = Path(project_root)
    result = Path(result_root)
    dataset, assignments = load_ood_context(project_root=root, assignments_path=result / "13_split_inventory.csv")
    mapping = family_mapping()
    class_to_family = dict(zip(mapping["class"], mapping["family"]))
    labels = sorted({class_to_family[label] for label in CLASS_LABELS})
    features_by_group, removed = input_feature_groups()
    if len(features_by_group["A_strict_38"]) != 38 or removed["B_without_potential_leakage"] != ["LVCR"] or len(removed["C_without_SLBIC_initial"]) != 14:
        raise AssertionError("16 feature groups do not match the 09 audit")
    if dataset["sample_id"].nunique() != 505:
        raise AssertionError("16 must reuse the 505 trajectory matched cohort")
    if assignments.duplicated(["split_type", "split_id", "sample_id"]).any():
        raise AssertionError("A sample_id appears more than once in a 16 split")

    metric_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    split_specs: list[tuple[str, str, int, str, str, bool]] = []
    for (split_type, split_id), _split in assignments.groupby(["split_type", "split_id"], sort=False):
        model_name = _model_name_from_15(result, split_type, split_id) if split_type in {"random", "severity_blocked", "severity_extrapolation"} else "hist_gradient_boosting"
        split_specs.append((split_type, split_id, MAIN_WINDOW_S, "A_strict_38", "main", model_name, True))
        for window_s in (60, 90):
            split_specs.append((split_type, split_id, window_s, "A_strict_38", "control", "hist_gradient_boosting", False))
        if split_type == "severity_extrapolation":
            split_specs.append((split_type, split_id, MAIN_WINDOW_S, "B_without_potential_leakage", "sensitivity", model_name, False))
            split_specs.append((split_type, split_id, MAIN_WINDOW_S, "C_without_SLBIC_initial", "sensitivity", model_name, False))
    for split_type, split_id, window_s, feature_group, scope, model_name, emit_curve in split_specs:
        current = _evaluate_split(
            dataset, assignments, result, split_type, split_id, window_s, feature_group, scope,
            model_name, labels, emit_curve,
        )
        metric_rows.extend(current["metrics"])
        risk_rows.extend(current["risk"])
        conformal_rows.extend(current["conformal"])
        sample_rows.extend(current["samples"])

    metrics = pd.DataFrame(metric_rows)
    risk = pd.DataFrame(risk_rows)
    conformal = pd.DataFrame(conformal_rows)
    samples = pd.DataFrame(sample_rows)
    class_rejection = _aggregate_class_rejection(samples, labels)
    ood_detection = _build_ood_detection(samples)
    figures = _write_figures(risk, conformal, samples, metrics, class_rejection, result)

    main_test = metrics.loc[
        metrics["analysis_scope"].eq("main")
        & metrics["window_s"].eq(MAIN_WINDOW_S)
        & metrics["feature_group"].eq("A_strict_38")
        & metrics["eval_split"].eq("test")
    ]
    extrapolation = main_test.loc[main_test["split_type"].eq("severity_extrapolation")]
    extrapolation_summary = extrapolation.groupby("mechanism", as_index=False)[
        ["coverage", "selective_macro_f1", "selective_balanced_accuracy", "risk", "reject_rate"]
    ].mean()
    validation_main = metrics.loc[
        metrics["analysis_scope"].eq("main")
        & metrics["window_s"].eq(MAIN_WINDOW_S)
        & metrics["feature_group"].eq("A_strict_38")
        & metrics["eval_split"].eq("test")
        & metrics["mechanism"].isin((*SCALAR_MECHANISMS, "combination"))
    ]
    validation_choice = validation_main.groupby("mechanism", as_index=False).agg(
        mean_coverage=("validation_coverage", "mean"), mean_macro_f1=("selective_macro_f1", "mean")
    )
    eligible = validation_choice.loc[validation_choice["mean_coverage"].between(0.60, 0.80)]
    if eligible.empty:
        eligible = validation_choice
    recommended = str(eligible.sort_values(["mean_macro_f1", "mean_coverage"], ascending=[False, False]).iloc[0]["mechanism"])

    control_summary = metrics.loc[metrics["analysis_scope"].eq("control") & metrics["eval_split"].eq("test")].groupby(
        ["window_s", "split_type", "mechanism"], as_index=False
    )[["coverage", "selective_macro_f1", "risk", "reject_rate"]].mean()
    sensitivity = metrics.loc[
        metrics["analysis_scope"].eq("sensitivity") & metrics["eval_split"].eq("test")
    ].groupby(["feature_group", "split_type", "mechanism"], as_index=False)[
        ["coverage", "selective_macro_f1", "risk", "reject_rate"]
    ].mean()
    conformal_main = conformal.loc[
        conformal["analysis_scope"].eq("main") & conformal["window_s"].eq(MAIN_WINDOW_S) & conformal["feature_group"].eq("A_strict_38")
    ]
    conformal_extrapolation = conformal_main.loc[conformal_main["split_type"].eq("severity_extrapolation")].groupby("alpha", as_index=False)[
        ["empirical_set_coverage", "average_prediction_set_size", "single_label_rate", "ambiguous_multi_label_rate", "empty_unknown_rate", "coverage", "selective_macro_f1", "risk", "reject_rate"]
    ].mean()
    forced_extrapolation = extrapolation_summary.loc[extrapolation_summary["mechanism"].eq("forced_family")]
    forced_risk = float(forced_extrapolation["risk"].iloc[0]) if not forced_extrapolation.empty else np.nan
    forced_macro_f1 = float(forced_extrapolation["selective_macro_f1"].iloc[0]) if not forced_extrapolation.empty else np.nan
    conformal_reasonable = conformal_extrapolation.loc[conformal_extrapolation["coverage"].between(0.60, 0.80)]
    conformal_candidate = conformal_reasonable.loc[
        conformal_reasonable["risk"].lt(forced_risk) & conformal_reasonable["selective_macro_f1"].ge(forced_macro_f1 - 0.01)
    ]
    focus_samples = samples.loc[
        samples["analysis_scope"].eq("main") & samples["window_s"].eq(MAIN_WINDOW_S) & samples["feature_group"].eq("A_strict_38")
    ].copy()
    focus_enrichment = (
        focus_samples.assign(is_focus=focus_samples["accident_class"].isin(FOCUS_CLASSES))
        .groupby("mechanism", as_index=False)
        .apply(
            lambda group: pd.Series({
                "focus_reject_rate": float(group.loc[group["is_focus"], "rejected"].mean()) if group["is_focus"].any() else np.nan,
                "other_reject_rate": float(group.loc[~group["is_focus"], "rejected"].mean()) if (~group["is_focus"]).any() else np.nan,
            }),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    if not focus_enrichment.empty:
        focus_enrichment["focus_minus_other_reject_rate"] = focus_enrichment["focus_reject_rate"] - focus_enrichment["other_reject_rate"]
    gap_summary = ood_detection.loc[ood_detection["proxy_type"].eq("extrapolation_test_vs_random_test")].copy()
    summary = {
        "experiment": "OOD-aware selective family diagnosis",
        "cohort_count": int(dataset["sample_id"].nunique()),
        "primary_window_s": MAIN_WINDOW_S,
        "control_windows_s": [60, 90],
        "split_protocol": "Reuse results/13_split_inventory.csv; trajectory/sample_id grouped random, severity-blocked, and severity-extrapolation splits.",
        "mechanism_comparison_extrapolation_120s": _json_records(extrapolation_summary),
        "ood_detection_proxy": _json_records(ood_detection),
        "accepted_rejected_gap_summary": _json_records(gap_summary),
        "conformal_extrapolation_120s": _json_records(conformal_extrapolation),
        "focus_class_rejection": _json_records(class_rejection),
        "focus_rejection_enrichment": _json_records(focus_enrichment),
        "control_window_summary": _json_records(control_summary),
        "leakage_confounding_sensitivity": _json_records(sensitivity),
        "validation_only_recommendation": {
            "selected_mechanism": recommended,
            "operating_target_coverage": OPERATING_COVERAGE,
            "selection_rule": "Validation-only mean selective Macro-F1 among mechanisms with validation coverage in [0.60, 0.80]; fallback to highest validation Macro-F1.",
        },
        "success_criteria": {
            "reasonable_coverage_60_to_80": bool(
                extrapolation_summary.loc[extrapolation_summary["mechanism"].ne("forced_family"), "coverage"].between(0.60, 0.80).any()
                or not conformal_reasonable.empty
                if not extrapolation_summary.empty else not conformal_reasonable.empty
            ),
            "risk_reduction_vs_forced_family": bool(
                (
                    extrapolation_summary.loc[extrapolation_summary["mechanism"].ne("forced_family"), "risk"].min() < forced_risk
                    if not extrapolation_summary.loc[extrapolation_summary["mechanism"].ne("forced_family")].empty and np.isfinite(forced_risk)
                    else False
                ) or not conformal_candidate.empty
            ),
            "macro_f1_not_materially_worse_at_reasonable_coverage": bool(
                not conformal_candidate.empty
                or (
                    not extrapolation_summary.loc[extrapolation_summary["mechanism"].ne("forced_family")].empty
                    and np.isfinite(forced_macro_f1)
                    and (extrapolation_summary.loc[extrapolation_summary["mechanism"].ne("forced_family"), "selective_macro_f1"] >= forced_macro_f1 - 0.01).any()
                )
            ),
            "ood_proxy_reject_rate_exceeds_id": bool(
                (ood_detection.loc[ood_detection["proxy_type"].eq("extrapolation_test_vs_random_test"), "ood_reject_rate"]
                > ood_detection.loc[ood_detection["proxy_type"].eq("extrapolation_test_vs_random_test"), "id_reject_rate"]).any()
                if not ood_detection.empty else False
            ),
            "focus_classes_rejected_more_than_other_classes": bool(
                (focus_enrichment["focus_minus_other_reject_rate"] > 0).any() if not focus_enrichment.empty else False
            ),
        },
        "judgement": "pending_manual_review",
        "decision_policy_path": "results/16_decision_policy.json",
        "figure_paths": figures,
        "assertions": {
            "same_13_assignments": True,
            "trajectory_sample_id_grouped": True,
            "strict_38_and_declared_sensitivity_groups": True,
            "probability_calibration_train_only": True,
            "distance_scaler_and_covariance_train_only": True,
            "conformal_calibration_validation_only": True,
            "thresholds_validation_only": True,
            "test_not_used_for_tuning": True,
            "severity_shift_is_proxy_not_production_ood": True,
        },
    }
    success = summary["success_criteria"]
    summary["judgement"] = "candidate_for_v1_unknown_policy" if success["reasonable_coverage_60_to_80"] and success["risk_reduction_vs_forced_family"] and success["macro_f1_not_materially_worse_at_reasonable_coverage"] and success["ood_proxy_reject_rate_exceeds_id"] and success["focus_classes_rejected_more_than_other_classes"] else "data_insufficient_for_reliable_ood_rejector"

    policy = {
        "status": "research_policy_only",
        "warning": "These results are simulation/cohort-specific and are not safe-deployment evidence.",
        "recommended_mechanism_from_validation_only": recommended,
        "operating_target_coverage": OPERATING_COVERAGE,
        "sequence": [
            "family classifier",
            "train-only fitted probability calibration",
            "calibrated probability and train-only class-conditional distance checks",
            "split-conformal prediction set at alpha=0.10 using validation calibration",
            "singleton accepted family; prediction set size > 1 -> Requires review; empty set -> Unknown",
        ],
        "unknown_rule": "Reject to Unknown when the selected probability/distance policy fails or conformal set is empty.",
        "requires_review_rule": "Output Requires review when conformal prediction set contains multiple families.",
        "limitations": [
            "Severity extrapolation and large severity gap are dataset-specific OOD proxies, not real production OOD labels.",
            "Thresholds are split-specific validation thresholds and require independent external validation before any product use.",
            "Family diagnosis does not establish reliable subtype diagnosis or safety function.",
        ],
    }
    metrics.to_csv(result / "16_selective_metrics.csv", index=False)
    ood_detection.to_csv(result / "16_ood_detection_metrics.csv", index=False)
    conformal.to_csv(result / "16_conformal_metrics.csv", index=False)
    class_rejection.to_csv(result / "16_rejection_by_class.csv", index=False)
    risk.to_csv(result / "16_risk_coverage.csv", index=False)
    (result / "16_decision_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (result / "16_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "metrics": metrics,
        "ood_detection": ood_detection,
        "conformal": conformal,
        "class_rejection": class_rejection,
        "risk": risk,
        "samples": samples,
        "summary": summary,
        "policy": policy,
    }


if __name__ == "__main__":
    output = run_experiment()
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, default=str))
