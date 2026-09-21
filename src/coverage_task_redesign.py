"""Coverage-aware task redesign and selective family diagnosis for milestone 15."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, log_loss, recall_score

try:
    from .hierarchical_diagnosis import family_mapping
    from .multi_accident_features import strict_process_features
    from .multi_accident_models import model_suite
    from .ood_validation import CLASS_LABELS
    from .severity_invariant import PROJECT_ROOT, feature_columns, load_ood_context
except ImportError:
    from hierarchical_diagnosis import family_mapping
    from multi_accident_features import strict_process_features
    from multi_accident_models import model_suite
    from ood_validation import CLASS_LABELS
    from severity_invariant import PROJECT_ROOT, feature_columns, load_ood_context


RESULT_ROOT = PROJECT_ROOT / "results"
FIGURE_ROOT = RESULT_ROOT / "figures"
WINDOW_S = 120
MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
THRESHOLD_GRID = np.round(np.linspace(0.0, 0.95, 20), 3)
FOCUS_CLASSES = ("RI", "LOCAC", "SLBIC")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _aligned_probabilities(model: object, probabilities: np.ndarray, labels: list[str]) -> np.ndarray:
    classes = list(getattr(model, "classes_"))
    aligned = np.zeros((len(probabilities), len(labels)), dtype=float)
    for index, label in enumerate(classes):
        if label in labels:
            aligned[:, labels.index(label)] = probabilities[:, index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sums, out=np.full_like(aligned, 1.0 / len(labels)), where=row_sums > 0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], accepted: np.ndarray | None = None) -> dict[str, Any]:
    if accepted is None:
        accepted = np.ones(len(y_true), dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    total = int(len(y_true))
    n_accepted = int(accepted.sum())
    coverage = float(n_accepted / total) if total else 0.0
    if n_accepted == 0:
        return {
            "n_total": total,
            "n_accepted": 0,
            "n_rejected": total,
            "coverage": 0.0,
            "selective_accuracy": np.nan,
            "selective_macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "risk": np.nan,
        }
    observed = [label for label in labels if np.any(y_true[accepted] == label)]
    accuracy = float(np.mean(y_true[accepted] == y_pred[accepted]))
    return {
        "n_total": total,
        "n_accepted": n_accepted,
        "n_rejected": total - n_accepted,
        "coverage": coverage,
        "selective_accuracy": accuracy,
        "selective_macro_f1": float(f1_score(y_true[accepted], y_pred[accepted], labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y_true[accepted], y_pred[accepted], labels=observed, average="macro", zero_division=0)) if observed else np.nan,
        "risk": float(1.0 - accuracy),
    }


def _calibration_rows(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    base: dict[str, Any],
    bins: int = 10,
) -> list[dict[str, Any]]:
    confidence = probabilities.max(axis=1)
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    correct = predictions == y_true
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    one_hot = np.zeros_like(probabilities)
    for index, label in enumerate(labels):
        one_hot[:, index] = y_true == label
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))) if len(y_true) else np.nan
    loss = float(log_loss(y_true, probabilities, labels=labels)) if len(y_true) else np.nan
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        count = int(selected.sum())
        mean_conf = float(confidence[selected].mean()) if count else np.nan
        accuracy = float(correct[selected].mean()) if count else np.nan
        if count:
            ece += count / len(y_true) * abs(mean_conf - accuracy)
        rows.append(
            {
                **base,
                "bin": bin_id,
                "bin_lower": bin_id / bins,
                "bin_upper": (bin_id + 1) / bins,
                "count": count,
                "mean_confidence": mean_conf,
                "accuracy": accuracy,
                "ece": np.nan,
                "brier_score": brier,
                "log_loss": loss,
            }
        )
    for row in rows:
        row["ece"] = float(ece)
    return rows


def _threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    accepted = probabilities.max(axis=1) >= threshold
    return {"threshold": float(threshold), **_metrics(y_true, predictions, labels, accepted)}


def choose_validation_threshold(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> tuple[float, pd.DataFrame]:
    rows = [_threshold_metrics(y_true, probabilities, labels, float(threshold)) for threshold in THRESHOLD_GRID]
    curve = pd.DataFrame(rows)
    eligible = curve.loc[curve["coverage"].ge(0.50)]
    if eligible.empty:
        eligible = curve
    selected = eligible.sort_values(
        ["selective_macro_f1", "selective_accuracy", "coverage", "threshold"],
        ascending=[False, False, False, True],
        na_position="last",
    ).iloc[0]
    return float(selected["threshold"]), curve


def _tier(code: str, status: str, reason: str, evidence: str) -> dict[str, str]:
    return {"tier": code, "status": status, "reason": reason, "evidence": evidence}


def build_task_coverage(
    result_root: str | Path = RESULT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(result_root)
    summary13 = json.loads((root / "13_ood_summary.json").read_text(encoding="utf-8"))
    mapping = pd.read_csv(root / "14_accident_family_mapping.csv")
    mapped_classes = set(CLASS_LABELS)
    matched = {str(key): int(value) for key, value in summary13["ood_matched_cohort_counts"].items()}
    directional = pd.read_csv(root / "14_ood_error_decomposition.csv")
    class_rows: list[dict[str, Any]] = []
    for row in mapping.to_dict(orient="records"):
        class_direction = directional.loc[directional["accident_class"].eq(row["class"])]
        low = class_direction.loc[class_direction["direction"].eq("low_train_to_high_test")]
        high = class_direction.loc[class_direction["direction"].eq("high_train_to_low_test")]
        low_support = int(low["test_support"].sum()) if not low.empty else 0
        high_support = int(high["test_support"].sum()) if not high.empty else 0
        matched_count = matched.get(row["class"], 0)
        if matched_count >= 20 and low_support >= 5 and high_support >= 5:
            tier = _tier("Tier A", "supported", "At least 20 matched trajectories and at least 5 OOD test cases in both severity directions.", "results/13_ood_summary.json; results/14_ood_error_decomposition.csv")
        elif matched_count > 0:
            tier = _tier("Tier B", "exploratory", "Class is present in the matched cohort but support is insufficient for stable two-direction OOD claims.", "results/13_ood_summary.json; results/14_ood_error_decomposition.csv")
        else:
            tier = _tier("Tier C", "not-supported", "No strict matched 120 s pre-protection trajectory is available in the 13 cohort.", "results/13_ood_summary.json")
        class_rows.append(
            {
                "class": row["class"],
                "official_description": row["official_description"],
                "family": row["family"],
                "in_fixed_12": row["class"] in mapped_classes,
                "matched_trajectories": matched_count,
                "low_to_high_test_support": low_support,
                "high_to_low_test_support": high_support,
                "family_mapping_confidence": row["confidence"],
                **tier,
            }
        )
    class_coverage = pd.DataFrame(class_rows)

    family_rows: list[dict[str, Any]] = []
    for family, members in mapping.groupby("family", sort=True):
        official_members = sorted(members["class"].tolist())
        fixed_members = sorted(set(official_members) & set(CLASS_LABELS))
        active_members = sorted(class_coverage.loc[class_coverage["family"].eq(family) & class_coverage["matched_trajectories"].gt(0), "class"].tolist())
        low_support = int(class_coverage.loc[class_coverage["class"].isin(active_members), "low_to_high_test_support"].sum())
        high_support = int(class_coverage.loc[class_coverage["class"].isin(active_members), "high_to_low_test_support"].sum())
        matched_count = int(class_coverage.loc[class_coverage["class"].isin(active_members), "matched_trajectories"].sum())
        missing_fixed = sorted(set(fixed_members) - set(active_members))
        if matched_count >= 20 and low_support >= 5 and high_support >= 5 and not missing_fixed:
            tier = _tier("Tier A", "supported", "The evaluated fixed-12 members have adequate matched and two-direction OOD support.", "results/13_ood_summary.json; results/14_ood_error_decomposition.csv; results/14_accident_family_mapping.csv")
        elif matched_count > 0:
            tier = _tier("Tier B", "exploratory", "Family has some matched support but at least one fixed-12 member is missing or support is sparse.", "results/13_ood_summary.json; results/14_ood_error_decomposition.csv; results/14_accident_family_mapping.csv")
        else:
            tier = _tier("Tier C", "not-supported", "No strict matched trajectory is available for this family in the evaluated 120 s cohort.", "results/14_accident_family_mapping.csv")
        family_rows.append(
            {
                "family": family,
                "official_members": ";".join(official_members),
                "fixed_12_members": ";".join(fixed_members),
                "active_matched_members": ";".join(active_members),
                "missing_fixed_12_members": ";".join(missing_fixed),
                "matched_trajectories": matched_count,
                "low_to_high_test_support": low_support,
                "high_to_low_test_support": high_support,
                **tier,
            }
        )
    family_coverage = pd.DataFrame(family_rows)

    capability_rows = [
        {"milestone": "03", "capability": "Dataset schema, trajectory grouping, strict 38-variable audit", "status": "supported", "tier": "Tier A", "evidence": "results/feature_groups.csv; results/09_multi_accident_class_inventory.csv", "limitation": "Research-data integrity, not plant deployment evidence."},
        {"milestone": "04", "capability": "Static early anomaly detection", "status": "exploratory", "tier": "Tier B", "evidence": "results/early_anomaly_detection_*.csv", "limitation": "Reference and operating-condition scope is limited."},
        {"milestone": "05", "capability": "Dynamic early warning", "status": "exploratory", "tier": "Tier B", "evidence": "results/dynamic_early_warning_*.csv", "limitation": "Alarm coverage and normal-reference validity remain incomplete."},
        {"milestone": "06", "capability": "Normal-reference expansion", "status": "not-supported", "tier": "Tier C", "evidence": "results/normal_reference_audit_summary.json", "limitation": "No complete fixed-power accident-prehistory reference for strict early-warning claims."},
        {"milestone": "07", "capability": "LOCA severity estimation", "status": "exploratory", "tier": "Tier B", "evidence": "results/07_severity_summary.json", "limitation": "LOCA-only and severity coverage is simulation-specific."},
        {"milestone": "08", "capability": "Protection-time prediction", "status": "exploratory", "tier": "Tier B", "evidence": "results/08_protection_time_summary.json", "limitation": "Landmark and support limitations prevent general deployment claims."},
        {"milestone": "09", "capability": "Multi-accident data audit", "status": "supported", "tier": "Tier A", "evidence": "results/09_multi_accident_audit_summary.json", "limitation": "Audit capability does not imply classification generalization."},
        {"milestone": "10", "capability": "Grouped 12-class traditional baseline", "status": "exploratory", "tier": "Tier B", "evidence": "results/10_multi_accident_summary.json", "limitation": "Random-split performance is not sufficient under severity OOD."},
        {"milestone": "11", "capability": "Temporal diagnosability", "status": "exploratory", "tier": "Tier B", "evidence": "results/11_temporal_diagnosability_summary.json", "limitation": "Strict pre-protection support is uneven."},
        {"milestone": "12", "capability": "Repeated robust validation", "status": "exploratory", "tier": "Tier B", "evidence": "results/12_robust_validation_summary.json", "limitation": "Sparse classes and protection-event coverage remain limiting."},
        {"milestone": "13", "capability": "Severity-blocked and extrapolation 12-class diagnosis", "status": "exploratory", "tier": "Tier B", "evidence": "results/13_ood_summary.json", "limitation": "120 s extrapolation Macro-F1 is below 0.50 and directional recalls are unstable."},
        {"milestone": "14", "capability": "Severity-invariant and hierarchical diagnosis", "status": "exploratory", "tier": "Tier B", "evidence": "results/14_summary.json; results/14_hierarchical_metrics.csv", "limitation": "Family-level gain does not transfer to end-to-end subtype diagnosis."},
        {"milestone": "15", "capability": "Coverage-aware family selective diagnosis", "status": "exploratory", "tier": "Tier B", "evidence": "results/15_summary.json", "limitation": "Selective capability is still simulation/cohort-specific and uses an explicit unknown reject state."},
    ]
    capability = pd.DataFrame(capability_rows)
    metadata = {
        "tier_rules": {
            "class_or_family_tier_A": "At least 20 matched trajectories, at least 5 OOD test cases in both low-to-high and high-to-low directions, and no missing fixed-12 family member for family rows.",
            "tier_B": "Some matched support exists but the Tier-A conditions are not met.",
            "tier_C": "No strict matched support for the evaluated 120 s pre-protection cohort.",
        },
        "source_milestones": "03-14 local results and the 14 official-name family mapping",
    }
    return class_coverage, family_coverage, capability, metadata


def _fit_calibrated(base: object, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[object, int, str]:
    min_support = int(y_train.value_counts().min())
    folds = min(3, min_support)
    if folds < 2:
        base.fit(x_train, y_train)
        return base, 0, "uncalibrated_min_class_support"
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=folds, n_jobs=1)
    calibrated.fit(x_train, y_train)
    return calibrated, folds, "train_only_sigmoid_cv"


def evaluate_family_selective(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    class_coverage: pd.DataFrame,
    gap_path: str | Path = RESULT_ROOT / "13_nearest_severity_gap.csv",
) -> dict[str, pd.DataFrame]:
    mapping = family_mapping()
    class_to_family = dict(zip(mapping["class"], mapping["family"]))
    labels = sorted({class_to_family[label] for label in CLASS_LABELS})
    strict = strict_process_features()
    columns = feature_columns(strict)
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for (split_type, split_id), split in assignments.groupby(["split_type", "split_id"], sort=False):
        active = split.loc[split["partition"].isin(["train", "validation", "test"])]
        frame = dataset.loc[dataset["window_s"].eq(WINDOW_S)].merge(
            active[["sample_id", "partition"]], on="sample_id", how="inner", validate="many_to_one"
        ).copy()
        frame["family"] = frame["accident_class"].map(class_to_family)
        train = frame.loc[frame["partition"].eq("train")]
        validation = frame.loc[frame["partition"].eq("validation")]
        test = frame.loc[frame["partition"].eq("test")]
        seed_values = split["seed"].dropna()
        seed = int(seed_values.iloc[0]) if not seed_values.empty else 20260921
        model_candidates: list[dict[str, Any]] = []
        fitted: dict[str, tuple[object, int, str]] = {}
        for model_name in MODEL_NAMES:
            calibrated, folds, calibration_status = _fit_calibrated(model_suite(seed)[model_name], train[columns], train["family"])
            val_prob = _aligned_probabilities(calibrated, calibrated.predict_proba(validation[columns]), labels)
            val_pred = np.asarray(labels)[val_prob.argmax(axis=1)]
            val_metrics = _metrics(validation["family"].to_numpy(), val_pred, labels)
            val_metrics["log_loss"] = float(log_loss(validation["family"], val_prob, labels=labels))
            model_candidates.append({"model": model_name, **val_metrics})
            fitted[model_name] = (calibrated, folds, calibration_status)
        selected_model = sorted(
            model_candidates,
            key=lambda row: (-row["selective_macro_f1"], -row["balanced_accuracy"], row["log_loss"], row["model"]),
        )[0]
        model_name = selected_model["model"]
        model, folds, calibration_status = fitted[model_name]
        val_prob = _aligned_probabilities(model, model.predict_proba(validation[columns]), labels)
        test_prob = _aligned_probabilities(model, model.predict_proba(test[columns]), labels)
        threshold, val_curve = choose_validation_threshold(validation["family"].to_numpy(), val_prob, labels)
        for eval_name, evaluated, probabilities in (("validation", validation, val_prob), ("test", test, test_prob)):
            y_true = evaluated["family"].to_numpy()
            y_pred = np.asarray(labels)[probabilities.argmax(axis=1)]
            full = _metrics(y_true, y_pred, labels)
            full.update(
                {
                    "split_type": split_type,
                    "split_id": split_id,
                    "window_s": WINDOW_S,
                    "eval_split": eval_name,
                    "mode": "forced_family",
                    "model": model_name,
                    "threshold": 0.0,
                    "selected_threshold": threshold,
                    "calibration_folds": folds,
                    "calibration_status": calibration_status,
                    "selected_for_test": eval_name == "test",
                    "curve_point": False,
                }
            )
            metric_rows.append(full)
            selective = _threshold_metrics(y_true, probabilities, labels, threshold)
            selective.update(
                {
                    "split_type": split_type,
                    "split_id": split_id,
                    "window_s": WINDOW_S,
                    "eval_split": eval_name,
                    "mode": "family_selective",
                    "model": model_name,
                    "selected_threshold": threshold,
                    "calibration_folds": folds,
                    "calibration_status": calibration_status,
                    "selected_for_test": eval_name == "test",
                    "curve_point": False,
                }
            )
            metric_rows.append(selective)
            curve = val_curve if eval_name == "validation" else pd.DataFrame(
                [_threshold_metrics(y_true, probabilities, labels, float(value)) for value in THRESHOLD_GRID]
            )
            for row in curve.to_dict(orient="records"):
                row.update(
                    {
                        "split_type": split_type,
                        "split_id": split_id,
                        "window_s": WINDOW_S,
                        "eval_split": eval_name,
                        "mode": "family_selective_curve",
                        "model": model_name,
                        "selected_threshold": threshold,
                        "calibration_folds": folds,
                        "calibration_status": calibration_status,
                        "selected_for_test": eval_name == "test" and float(row["threshold"]) == threshold,
                        "curve_point": True,
                    }
                )
                metric_rows.append(row)
            calibration_rows.extend(
                _calibration_rows(
                    y_true,
                    probabilities,
                    labels,
                    {
                        "split_type": split_type,
                        "split_id": split_id,
                        "window_s": WINDOW_S,
                        "eval_split": eval_name,
                        "model": model_name,
                        "calibration_folds": folds,
                        "calibration_status": calibration_status,
                    },
                )
            )
            if eval_name == "test":
                gaps = pd.read_csv(gap_path)
                gaps = gaps.loc[gaps["split_type"].eq(split_type) & gaps["split_id"].eq(split_id), ["sample_id", "nearest_severity_gap"]]
                sample = evaluated[["sample_id", "accident_class", "severity", "family"]].copy()
                sample["predicted_family"] = y_pred
                sample["max_probability"] = probabilities.max(axis=1)
                sample["threshold"] = threshold
                sample["rejected"] = sample["max_probability"] < threshold
                sample["accepted"] = ~sample["rejected"]
                sample["split_type"] = split_type
                sample["split_id"] = split_id
                sample["direction"] = split_id if split_type == "severity_extrapolation" else "random_or_blocked"
                sample = sample.merge(gaps, on="sample_id", how="left", validate="one_to_one")
                sample["focus_class"] = sample["accident_class"].isin(FOCUS_CLASSES)
                sample["high_gap"] = sample["nearest_severity_gap"].ge(sample["nearest_severity_gap"].median())
                sample_rows.extend(sample.to_dict(orient="records"))
        selection_rows.append(
            {
                "split_type": split_type,
                "split_id": split_id,
                "window_s": WINDOW_S,
                "selected_model": model_name,
                "selected_threshold": threshold,
                "calibration_folds": folds,
                "calibration_status": calibration_status,
                "validation_policy": "fixed threshold grid; maximize validation selective Macro-F1 subject to coverage>=0.50; tie-break accuracy then coverage",
            }
        )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "rejection": pd.DataFrame(sample_rows),
        "selection": pd.DataFrame(selection_rows),
        "family_labels": pd.DataFrame({"family": labels}),
    }


def forced_12_metrics(result_root: str | Path = RESULT_ROOT, assignments: pd.DataFrame | None = None) -> pd.DataFrame:
    root = Path(result_root)
    hierarchy = pd.read_csv(root / "14_hierarchical_metrics.csv")
    selected = hierarchy.loc[
        hierarchy["evaluation_scope"].eq("fine_grained_12_class")
        & hierarchy["split_type"].isin(["random", "severity_blocked", "severity_extrapolation"])
        & hierarchy["window_s"].eq(WINDOW_S)
    ].copy()
    if assignments is not None:
        counts = assignments.loc[assignments["partition"].eq("test")].groupby(["split_type", "split_id"]).size().rename("n_total").reset_index()
        selected = selected.merge(counts, on=["split_type", "split_id"], how="left", validate="one_to_one")
    rows = []
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "split_type": row.split_type,
                "split_id": row.split_id,
                "window_s": WINDOW_S,
                "eval_split": "test",
                "mode": "forced_12_class",
                "model": "selected_14_baseline",
                "threshold": 0.0,
                "selected_threshold": 0.0,
                "n_total": int(getattr(row, "n_total", 0)),
                "n_accepted": int(getattr(row, "n_total", 0)),
                "n_rejected": 0,
                "coverage": 1.0,
                "selective_accuracy": float(row.accuracy),
                "selective_macro_f1": float(row.macro_f1),
                "balanced_accuracy": float(row.balanced_accuracy),
                "risk": float(1.0 - row.accuracy),
                "selected_for_test": True,
                "curve_point": False,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_mode_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["eval_split"].eq("test") & metrics["selected_for_test"].eq(True) & metrics["curve_point"].eq(False)]
    rows = []
    for key, group in selected.groupby(["split_type", "mode"], sort=False):
        split_type, mode = key
        rows.append(
            {
                "split_type": split_type,
                "mode": mode,
                "n_splits": int(len(group)),
                "coverage_mean": float(group["coverage"].mean()),
                "selective_accuracy_mean": float(group["selective_accuracy"].mean()),
                "selective_macro_f1_mean": float(group["selective_macro_f1"].mean()),
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "risk_mean": float(group["risk"].mean()),
                "selected_threshold_mean": float(group["selected_threshold"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _capability_json(capability: pd.DataFrame, class_coverage: pd.DataFrame, family_coverage: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        **metadata,
        "status_counts": {str(key): int(value) for key, value in capability["status"].value_counts().items()},
        "milestones": _json_records(capability),
        "class_tiers": _json_records(class_coverage),
        "family_tiers": _json_records(family_coverage),
    }


def _write_figures(metrics: pd.DataFrame, calibration: pd.DataFrame, rejection: pd.DataFrame, result_root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    curves = metrics.loc[metrics["mode"].eq("family_selective_curve") & metrics["eval_split"].eq("test")]
    if not curves.empty:
        grouped = curves.groupby(["split_type", "threshold"], as_index=False)[["coverage", "risk"]].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        for split_type, group in grouped.groupby("split_type", sort=False):
            ax.plot(group["coverage"], group["risk"], marker="o", label=split_type)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk = 1 - selective accuracy")
        ax.set_title("Family selective risk-coverage curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = result_root / "15_risk_coverage.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

        fig, ax = plt.subplots(figsize=(7, 4))
        for split_type, group in grouped.groupby("split_type", sort=False):
            values = curves.loc[curves["split_type"].eq(split_type)].groupby("threshold", as_index=False)["selective_macro_f1"].mean()
            values = values.merge(group[["threshold", "coverage"]], on="threshold", validate="one_to_one")
            ax.plot(values["coverage"], values["selective_macro_f1"], marker="o", label=split_type)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective Macro-F1")
        ax.set_title("Family selective Macro-F1 vs coverage")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = result_root / "15_macro_f1_vs_coverage.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    reliability = calibration.loc[calibration["eval_split"].eq("test")]
    if not reliability.empty:
        curve = reliability.groupby("bin", as_index=False)[["mean_confidence", "accuracy"]].mean().dropna()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.plot(curve["mean_confidence"], curve["accuracy"], marker="o")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title("Family probability reliability diagram")
        fig.tight_layout()
        path = result_root / "15_reliability_diagram.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    selected = metrics.loc[metrics["eval_split"].eq("test") & metrics["selected_for_test"].eq(True) & metrics["curve_point"].eq(False)]
    if not selected.empty:
        grouped = selected.groupby("mode", as_index=False)[["selective_macro_f1", "coverage"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].bar(grouped["mode"], grouped["selective_macro_f1"], color="#4c78a8")
        axes[0].set_ylabel("Macro-F1")
        axes[0].tick_params(axis="x", rotation=25)
        axes[1].bar(grouped["mode"], grouped["coverage"], color="#f58518")
        axes[1].set_ylabel("Coverage")
        axes[1].set_ylim(0, 1.05)
        axes[1].tick_params(axis="x", rotation=25)
        fig.suptitle("Forced 12-class, forced family, and family selective")
        fig.tight_layout()
        path = result_root / "15_forced_vs_selective.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    if not rejection.empty:
        focus = rejection.assign(group=np.where(rejection["focus_class"], "RI/LOCAC/SLBIC", "other classes"))
        grouped = focus.groupby("group", as_index=False)["rejected"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(grouped["group"], grouped["rejected"], color="#e45756")
        ax.set_ylabel("Rejection rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Rejection rate for unstable focus classes")
        fig.tight_layout()
        path = result_root / "15_rejection_focus_classes.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return paths


def run_experiment(project_root: str | Path = PROJECT_ROOT, result_root: str | Path = RESULT_ROOT) -> dict[str, Any]:
    root = Path(result_root)
    dataset, assignments = load_ood_context(project_root=project_root, assignments_path=root / "13_split_inventory.csv")
    class_coverage, family_coverage, capability, metadata = build_task_coverage(root)
    family_result = evaluate_family_selective(dataset, assignments, class_coverage)
    family_metrics = family_result["metrics"]
    forced12 = forced_12_metrics(root, assignments)
    combined_metrics = pd.concat([forced12, family_metrics], ignore_index=True, sort=False)
    aggregate = _aggregate_mode_metrics(combined_metrics)
    rejection = family_result["rejection"]
    if not rejection.empty:
        rejection["gap_group"] = np.where(rejection["high_gap"], "high_gap_or_above_split_median", "below_split_median")
        rejection["analysis_group"] = np.where(rejection["focus_class"], "RI_LOCAC_SLBIC", "other_classes")
    class_coverage.to_csv(root / "15_class_task_coverage.csv", index=False)
    family_coverage.to_csv(root / "15_family_task_coverage.csv", index=False)
    combined_metrics.to_csv(root / "15_selective_metrics.csv", index=False)
    family_result["calibration"].to_csv(root / "15_calibration_metrics.csv", index=False)
    rejection.to_csv(root / "15_rejection_analysis.csv", index=False)
    capability.to_csv(root / "15_npp_guard_v1_capability_matrix.csv", index=False)
    capability_json = _capability_json(capability, class_coverage, family_coverage, metadata)
    (root / "15_npp_guard_v1_capability_matrix.json").write_text(json.dumps(capability_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    family_summary = aggregate.loc[aggregate["mode"].isin(["forced_family", "family_selective"])]
    rejection_summary = {
        "n_test_rows": int(len(rejection)),
        "rejected_rows": int(rejection["rejected"].sum()) if not rejection.empty else 0,
        "rejection_rate": float(rejection["rejected"].mean()) if not rejection.empty else np.nan,
        "mean_gap_rejected": float(rejection.loc[rejection["rejected"], "nearest_severity_gap"].mean()) if rejection["rejected"].any() else np.nan,
        "mean_gap_accepted": float(rejection.loc[rejection["accepted"], "nearest_severity_gap"].mean()) if rejection["accepted"].any() else np.nan,
        "focus_rejection_rate": float(rejection.loc[rejection["focus_class"], "rejected"].mean()) if rejection["focus_class"].any() else np.nan,
        "other_rejection_rate": float(rejection.loc[~rejection["focus_class"], "rejected"].mean()) if (~rejection["focus_class"]).any() else np.nan,
    }
    summary = {
        "experiment": "Coverage-aware task redesign and selective family diagnosis",
        "cohort_count": int(dataset["sample_id"].nunique()),
        "window_s": WINDOW_S,
        "split_protocol": "Reuse results/13_split_inventory.csv; trajectory-level random/severity-blocked/extrapolation assignments; no test threshold tuning.",
        "coverage_tiers": {
            "class_counts": {str(key): int(value) for key, value in class_coverage["status"].value_counts().items()},
            "family_counts": {str(key): int(value) for key, value in family_coverage["status"].value_counts().items()},
        },
        "comparison_test_summary": _json_records(aggregate),
        "selected_model_counts": {str(key): int(value) for key, value in family_result["selection"]["selected_model"].value_counts().items()},
        "selected_threshold_summary": {
            "mean": float(family_result["selection"]["selected_threshold"].mean()),
            "min": float(family_result["selection"]["selected_threshold"].min()),
            "max": float(family_result["selection"]["selected_threshold"].max()),
        },
        "rejection_summary": rejection_summary,
        "calibration_summary": _json_records(
            family_result["calibration"].groupby(["eval_split", "model"], as_index=False)[["ece", "brier_score", "log_loss"]].mean()
        ),
        "capability_matrix": "results/15_npp_guard_v1_capability_matrix.csv/json",
        "figure_paths": [],
        "assertions": {
            "same_13_assignments": True,
            "train_only_probability_calibration": True,
            "validation_only_threshold_selection": True,
            "test_not_used_for_threshold": True,
            "unknown_reject_state": True,
        },
    }
    summary["figure_paths"] = _write_figures(combined_metrics, family_result["calibration"], rejection, root / "figures")
    (root / "15_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "summary": summary,
        "class_coverage": class_coverage,
        "family_coverage": family_coverage,
        "capability": capability,
        "metrics": combined_metrics,
        "calibration": family_result["calibration"],
        "rejection": rejection,
        "selection": family_result["selection"],
    }


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result["summary"]["coverage_tiers"], ensure_ascii=False))
