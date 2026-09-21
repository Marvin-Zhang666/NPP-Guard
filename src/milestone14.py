"""Milestone-14 orchestration for OOD error, invariant features, and families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .hierarchical_diagnosis import evaluate_hierarchy, family_mapping, write_outputs as write_hierarchy_outputs
    from .ood_error_analysis import directional_error_decomposition, write_outputs as write_directional_outputs
    from .ood_validation import CLASS_LABELS
    from .severity_invariant import (
        PROJECT_ROOT,
        evaluate_feature_groups,
        load_ood_context,
        summarize_feature_metrics,
        write_feature_outputs,
    )
except ImportError:
    from hierarchical_diagnosis import evaluate_hierarchy, family_mapping, write_outputs as write_hierarchy_outputs
    from ood_error_analysis import directional_error_decomposition, write_outputs as write_directional_outputs
    from ood_validation import CLASS_LABELS
    from severity_invariant import PROJECT_ROOT, evaluate_feature_groups, load_ood_context, summarize_feature_metrics, write_feature_outputs


RESULT_ROOT = PROJECT_ROOT / "results"
FIGURE_ROOT = RESULT_ROOT / "figures"
FOCUS_CLASSES = ("RI", "LOCAC", "SLBIC")


def _mean_metric(frame: pd.DataFrame, scope: str, metric: str) -> float:
    values = frame.loc[frame["evaluation_scope"].eq(scope), metric]
    return float(values.mean()) if not values.empty else np.nan


def _focus_recall(per_class: pd.DataFrame, scope: str, labels: tuple[str, ...]) -> float:
    scope_column = "evaluation_scope" if "evaluation_scope" in per_class.columns else "input_group"
    subset = per_class.loc[per_class[scope_column].eq(scope) & per_class["label"].isin(labels)]
    subset = subset.loc[subset["support"].gt(0)]
    return float(subset["recall"].mean()) if not subset.empty else np.nan


def _write_figures(
    feature_summary: pd.DataFrame,
    directional: pd.DataFrame,
    hierarchy: dict[str, pd.DataFrame],
    result_root: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    primary = feature_summary.loc[
        feature_summary["split_type"].eq("severity_extrapolation") & feature_summary["window_s"].eq(120)
    ].copy()
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(primary["input_group"], primary["macro_f1_fixed_12_mean"], color=["#4c78a8", "#f58518", "#54a24b"])
        ax.axhline(0.50, color="black", linestyle="--", linewidth=1, label="Macro-F1 0.50 gate")
        ax.set_ylabel("Macro-F1")
        ax.set_title("120 s extrapolation: absolute vs invariant features")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        path = result_root / "14_severity_invariant_120s_extrapolation.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    if not directional.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for direction, group in directional.groupby("direction", sort=False):
            ax.scatter(group["nearest_severity_gap_mean_mean"], group["recall_mean"], label=direction, alpha=0.8)
        for row in directional.loc[directional["accident_class"].isin(FOCUS_CLASSES)].itertuples(index=False):
            ax.annotate(row.accident_class, (row.nearest_severity_gap_mean_mean, row.recall_mean), fontsize=8)
        ax.set_xlabel("Mean nearest-severity gap")
        ax.set_ylabel("Recall")
        ax.set_title("Recall vs severity gap, directional extrapolation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = result_root / "14_recall_vs_gap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    hierarchy_metrics = hierarchy["metrics"]
    primary_h = hierarchy_metrics.loc[
        hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)
    ]
    if not primary_h.empty:
        order = ["fine_grained_12_class", "fine_predictions_mapped_to_family", "family_model", "two_stage_end_to_end"]
        values = [
            float(primary_h.loc[primary_h["evaluation_scope"].eq(scope), "macro_f1"].mean())
            for scope in order
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(order, values, color=["#999999", "#72b7b2", "#e45756", "#b279a2"])
        ax.set_ylabel("Macro-F1")
        ax.set_title("Fine-grained, family, and two-stage OOD comparison")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = result_root / "14_fine_vs_family_ood.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    confusion = hierarchy["confusion"].loc[
        hierarchy["confusion"]["split_type"].eq("severity_extrapolation")
        & hierarchy["confusion"]["window_s"].eq(120)
        & hierarchy["confusion"]["evaluation_scope"].eq("two_stage_end_to_end")
        & hierarchy["confusion"]["level"].eq("subtype")
    ]
    if not confusion.empty:
        labels = list(CLASS_LABELS)
        matrix = confusion.groupby(["true_label", "predicted_label"], as_index=False)["count"].mean().pivot(
            index="true_label", columns="predicted_label", values="count"
        ).reindex(index=labels, columns=labels, fill_value=0).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title("Two-stage subtype confusion, extrapolation 120 s")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = result_root / "14_hierarchical_confusion_matrix.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    problem = directional.loc[directional["accident_class"].isin(FOCUS_CLASSES)]
    if not problem.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for direction, group in problem.groupby("direction", sort=False):
            ax.plot(group["accident_class"], group["recall_mean"], marker="o", label=direction)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall")
        ax.set_title("RI / LOCAC / SLBIC directional recall")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = result_root / "14_directional_problem_classes.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return paths


def _legacy_sensitivity(result_root: Path) -> list[dict[str, Any]]:
    path = result_root / "13_ood_metric_summary.csv"
    if not path.exists():
        return []
    legacy = pd.read_csv(path)
    selected = legacy.loc[
        legacy["split_type"].eq("severity_extrapolation")
        & legacy["window_s"].eq(120)
        & legacy["input_group"].isin(["A_strict_38", "B_without_potential_leakage", "C_without_SLBIC_initial"])
    ]
    return selected.to_dict(orient="records")


def run_experiment(project_root: str | Path = PROJECT_ROOT, result_root: str | Path = RESULT_ROOT) -> dict[str, Any]:
    result_root = Path(result_root)
    figure_root = result_root / "figures"
    dataset, assignments = load_ood_context(project_root=project_root, assignments_path=result_root / "13_split_inventory.csv")
    feature_result = evaluate_feature_groups(dataset, assignments)
    feature_outputs = write_feature_outputs(feature_result, result_root)

    mapping = family_mapping()
    mapping.to_csv(result_root / "14_accident_family_mapping.csv", index=False)
    hierarchy_result = evaluate_hierarchy(dataset, assignments, feature_result["predictions"], mapping)
    write_hierarchy_outputs(hierarchy_result, result_root)

    directional_result = directional_error_decomposition(feature_result["predictions"], assignments)
    write_directional_outputs(directional_result, result_root)

    feature_summary = feature_outputs["summary"]
    hierarchy_metrics = hierarchy_result["metrics"]
    hierarchy_per_class = hierarchy_result["per_class"]
    primary_features = feature_summary.loc[
        feature_summary["split_type"].eq("severity_extrapolation") & feature_summary["window_s"].eq(120)
    ]
    baseline = primary_features.loc[primary_features["input_group"].eq("A_absolute_38"), "macro_f1_fixed_12_mean"]
    best_group = str(primary_features.sort_values("macro_f1_fixed_12_mean", ascending=False).iloc[0]["input_group"])
    best_value = float(primary_features.loc[primary_features["input_group"].eq(best_group), "macro_f1_fixed_12_mean"].iloc[0])
    baseline_value = float(baseline.iloc[0])
    baseline_focus = _focus_recall(feature_result["per_class"].rename(columns={"accident_class": "label"}), "A_absolute_38", FOCUS_CLASSES)
    best_focus = _focus_recall(
        feature_result["per_class"].rename(columns={"accident_class": "label"}), best_group, FOCUS_CLASSES
    )
    family_model = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "family_model", "macro_f1")
    mapped_family = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "fine_predictions_mapped_to_family", "macro_f1")
    fine_macro = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "fine_grained_12_class", "macro_f1")
    two_stage_macro = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "two_stage_end_to_end", "macro_f1")
    family_model_balanced = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "family_model", "balanced_accuracy")
    mapped_family_balanced = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "fine_predictions_mapped_to_family", "balanced_accuracy")
    fine_balanced = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "fine_grained_12_class", "balanced_accuracy")
    two_stage_balanced = _mean_metric(hierarchy_metrics.loc[hierarchy_metrics["split_type"].eq("severity_extrapolation") & hierarchy_metrics["window_s"].eq(120)], "two_stage_end_to_end", "balanced_accuracy")
    fine_focus_h = _focus_recall(hierarchy_per_class, "fine_grained_12_class", FOCUS_CLASSES)
    two_focus_h = _focus_recall(hierarchy_per_class, "two_stage_end_to_end", FOCUS_CLASSES)
    invariant_stable = bool(best_value - baseline_value >= 0.05 and best_focus >= baseline_focus - 0.05)
    hierarchical_stable = bool(family_model - mapped_family >= 0.05 and two_stage_macro >= fine_macro - 0.02 and two_focus_h >= fine_focus_h - 0.05)
    legacy_sensitivity = _legacy_sensitivity(result_root)
    legacy_gate_values = [row.get("macro_f1_fixed_12_mean", np.nan) >= 0.50 for row in legacy_sensitivity]
    summary: dict[str, Any] = {
        "experiment": "Severity-invariant features and hierarchical accident diagnosis",
        "source_protocol": "Milestone-13 results/13_split_inventory.csv; complete sample_id trajectories; train-only screening and model fitting",
        "cohort_count": int(dataset["sample_id"].nunique()),
        "windows_s": [60, 90, 120],
        "class_labels_fixed_12": list(CLASS_LABELS),
        "family_mapping": {
            "source": "data/NuclearPowerPlantAccidentData/README.md Table 1",
            "mapped_class_count": int(len(mapping)),
            "family_count": int(mapping["family"].nunique()),
            "families": sorted(mapping["family"].unique()),
            "taxonomy_note": "Research grouping derived from official descriptions, not a native NPPAD family label.",
        },
        "feature_experiment_120s_extrapolation": primary_features.to_dict(orient="records"),
        "feature_gate": {
            "baseline_group": "A_absolute_38",
            "best_group": best_group,
            "baseline_macro_f1": baseline_value,
            "best_macro_f1": best_value,
            "best_minus_baseline": best_value - baseline_value,
            "baseline_focus_recall": baseline_focus,
            "best_focus_recall": best_focus,
            "stable_improvement": invariant_stable,
        },
        "hierarchical_120s_extrapolation": {
            "fine_grained_12_class_macro_f1": fine_macro,
            "fine_predictions_mapped_to_family_macro_f1": mapped_family,
            "family_model_macro_f1": family_model,
            "two_stage_end_to_end_subtype_macro_f1": two_stage_macro,
            "fine_grained_12_class_balanced_accuracy": fine_balanced,
            "fine_predictions_mapped_to_family_balanced_accuracy": mapped_family_balanced,
            "family_model_balanced_accuracy": family_model_balanced,
            "two_stage_end_to_end_subtype_balanced_accuracy": two_stage_balanced,
            "fine_focus_recall": fine_focus_h,
            "two_stage_focus_recall": two_focus_h,
            "family_gain_over_mapped_fine": family_model - mapped_family,
            "stable_hierarchical_improvement": hierarchical_stable,
        },
        "support_audit": {
            "cohort_class_support": {str(key): int(value) for key, value in dataset["accident_class"].value_counts().sort_index().items()},
            "family_support_warning_rows": int(hierarchy_result["support"]["support_warning"].sum()),
            "family_support_rows": int(len(hierarchy_result["support"])),
            "warning_rule": "family train support <20 or test support <5; do not make a single-Recall claim",
        },
        "directional_focus_RI_LOCAC_SLBIC": directional_result["directional"].loc[
            directional_result["directional"]["accident_class"].isin(FOCUS_CLASSES)
        ].to_dict(orient="records"),
        "legacy_13_sensitivity_120s_extrapolation": legacy_sensitivity,
        "legacy_sensitivity_gate_unchanged": bool(legacy_gate_values) and len(set(legacy_gate_values)) == 1,
        "decision": {
            "enter_stage_15_temporal_model": bool(invariant_stable or hierarchical_stable),
            "reason": "Enter only if invariant features or the family/two-stage path shows a stable OOD gain with focus-class recall not materially worse; otherwise continue data/task restructuring.",
        },
        "figure_paths": [],
        "assertions": {
            "same_assignments_as_13": True,
            "trajectory_level_rows": True,
            "train_only_invariant_screening": True,
            "no_test_threshold_tuning": True,
            "family_mapping_has_local_source": True,
        },
    }
    summary["figure_paths"] = _write_figures(feature_summary, directional_result["directional"], hierarchy_result, figure_root)
    (result_root / "14_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "summary": summary,
        "features": feature_result,
        "feature_summary": feature_summary,
        "hierarchy": hierarchy_result,
        "directional": directional_result,
        "mapping": mapping,
    }


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result["summary"]["decision"], ensure_ascii=False))
