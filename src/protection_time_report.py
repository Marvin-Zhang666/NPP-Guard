"""Run and save the leakage-safe LOCA protection-time experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from protection_time_features import LANDMARKS_S, build_protection_time_dataset
from protection_time_models import INPUT_GROUPS, process_feature_columns, run_experiment
from severity_features import candidate_process_features


HIGH_COUPLING_BASES = {"P", "LVPZ", "TSAT", "VOL"}


def _severity_coverage(values: pd.Series) -> str:
    ids = sorted(int(value) for value in values)
    missing = [value for value in range(1, 101) if value not in ids]
    span = f"{ids[0]}-{ids[-1]}"
    return span if not missing else f"{span}; missing={','.join(map(str, missing))}"


def run_report(project_root: str | Path) -> dict[str, object]:
    """Build data, run both modes, save artifacts, and assert invariants."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_root = Path(project_root)
    results_root = project_root / "results"
    figures_root = results_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    candidate_features = candidate_process_features(results_root / "feature_groups.csv")
    feature_audit = pd.read_csv(results_root / "feature_groups.csv")
    selected_audit = feature_audit.loc[feature_audit["feature"].isin(candidate_features)]
    forbidden_groups = {
        "direct_accident_or_leak",
        "protection_or_control_action",
        "radiological_or_dose",
        "safety_margin_or_consequence",
    }
    assert len(candidate_features) == 38
    assert not selected_audit["group"].isin(forbidden_groups).any()

    dataset, point_audit = build_protection_time_dataset(
        loca_dir=project_root / "data/NuclearPowerPlantAccidentData/Operation_csv_data/LOCA",
        event_table_path=results_root / "loca_event_table.csv",
        landmarks_s=LANDMARKS_S,
        features=candidate_features,
    )
    process_columns = process_feature_columns(dataset)
    assert dataset["sample_id"].nunique() == 100
    assert len(process_columns) == 38 * 5
    assert (dataset["first_protection_s"] > dataset["landmark_s"]).all()
    assert (point_audit["last_sample_s"] <= point_audit["landmark_s"]).all()
    expected_counts = {30: 100, 60: 99, 120: 76, 300: 6}

    landmark_summary = (
        dataset.groupby("landmark_s")
        .agg(
            valid_samples=("sample_id", "nunique"),
            severity_min=("severity", "min"),
            severity_max=("severity", "max"),
            severity_unique=("severity", "nunique"),
            first_protection_min_s=("first_protection_s", "min"),
            first_protection_max_s=("first_protection_s", "max"),
        )
        .reindex(LANDMARKS_S)
        .reset_index()
    )
    actual_counts = dict(
        zip(
            landmark_summary["landmark_s"],
            landmark_summary["valid_samples"],
            strict=True,
        )
    )
    assert actual_counts == expected_counts
    coverage = (
        dataset.groupby("landmark_s")["severity"]
        .apply(_severity_coverage)
        .rename("severity_coverage")
        .reset_index()
    )
    landmark_summary = landmark_summary.merge(coverage, on="landmark_s")

    strict_metrics, strict_predictions, strict_splits, strict_audit = run_experiment(
        dataset, mode="strict"
    )
    sensitivity_metrics, sensitivity_predictions, sensitivity_splits, sensitivity_audit = run_experiment(
        dataset, mode="sensitivity", excluded_bases=HIGH_COUPLING_BASES
    )
    metrics = pd.concat([strict_metrics, sensitivity_metrics], ignore_index=True)
    predictions = pd.concat([strict_predictions, sensitivity_predictions], ignore_index=True)
    split_manifest = pd.concat([strict_splits, sensitivity_splits], ignore_index=True)
    leakage_audit = pd.concat([strict_audit, sensitivity_audit], ignore_index=True)
    assert set(metrics["input_group"]) == set(INPUT_GROUPS)
    assert set(metrics["model"]) == {
        "naive_mean", "ridge", "random_forest", "gradient_boosting"
    }
    assert not metrics[["mae", "rmse", "r2"]].isna().any().any()

    strict_validation = metrics.loc[
        (metrics["mode"] == "strict") & (metrics["split"] == "validation")
    ].sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
    selected_by_group = (
        strict_validation.groupby(["landmark_s", "input_group"], as_index=False)
        .head(1)
        .sort_values(["landmark_s", "input_group"])
    )
    selected_keys = ["mode", "landmark_s", "input_group", "model"]
    selected_group_test = selected_by_group[selected_keys].merge(
        metrics.loc[(metrics["mode"] == "strict") & (metrics["split"] == "test")],
        on=selected_keys,
        how="left",
    )
    best_validation = strict_validation.iloc[0]
    best_test = metrics.loc[
        (metrics["mode"] == "strict")
        & (metrics["split"] == "test")
        & metrics["landmark_s"].eq(best_validation["landmark_s"])
        & metrics["input_group"].eq(best_validation["input_group"])
        & metrics["model"].eq(best_validation["model"])
    ].iloc[0]

    sensitivity_validation = metrics.loc[
        (metrics["mode"] == "sensitivity") & (metrics["split"] == "validation")
    ].sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
    sensitivity_best_validation = sensitivity_validation.iloc[0]
    sensitivity_best_test = metrics.loc[
        (metrics["mode"] == "sensitivity")
        & (metrics["split"] == "test")
        & metrics["landmark_s"].eq(sensitivity_best_validation["landmark_s"])
        & metrics["input_group"].eq(sensitivity_best_validation["input_group"])
        & metrics["model"].eq(sensitivity_best_validation["model"])
    ].iloc[0]

    landmark_summary.to_csv(results_root / "08_landmark_summary.csv", index=False)
    point_audit.to_csv(results_root / "08_window_sample_points.csv", index=False)
    metrics.to_csv(results_root / "08_protection_time_metrics.csv", index=False)
    predictions.to_csv(results_root / "08_protection_time_predictions.csv", index=False)
    split_manifest.to_csv(results_root / "08_protection_time_split.csv", index=False)
    leakage_audit.to_csv(results_root / "08_leakage_audit.csv", index=False)
    selected_group_test.to_csv(results_root / "08_input_group_comparison.csv", index=False)

    group_conclusions = selected_group_test.pivot(
        index="landmark_s", columns="input_group", values="mae"
    ).reindex(columns=INPUT_GROUPS)
    summary = {
        "experiment": "LOCA landmark protection-time prediction",
        "target": "remaining_time_s = first_protection_s - landmark_s",
        "landmarks_s": list(LANDMARKS_S),
        "trajectory_count": 100,
        "candidate_process_feature_count": len(candidate_features),
        "derived_process_feature_count": len(process_columns),
        "landmark_valid_sample_counts": {
            str(row.landmark_s): int(row.valid_samples)
            for row in landmark_summary.itertuples()
        },
        "severity_coverage": {
            str(row.landmark_s): row.severity_coverage
            for row in landmark_summary.itertuples()
        },
        "strict_validation_selected_best": {
            "landmark_s": int(best_validation["landmark_s"]),
            "input_group": best_validation["input_group"],
            "model": best_validation["model"],
            "validation_mae": float(best_validation["mae"]),
            "validation_rmse": float(best_validation["rmse"]),
            "validation_r2": float(best_validation["r2"]),
        },
        "strict_test_metrics": {
            "mae": float(best_test["mae"]),
            "rmse": float(best_test["rmse"]),
            "r2": float(best_test["r2"]),
        },
        "sensitivity_excluded_bases": sorted(HIGH_COUPLING_BASES),
        "sensitivity_validation_selected_best": {
            "landmark_s": int(sensitivity_best_validation["landmark_s"]),
            "input_group": sensitivity_best_validation["input_group"],
            "model": sensitivity_best_validation["model"],
            "validation_mae": float(sensitivity_best_validation["mae"]),
            "test_mae": float(sensitivity_best_test["mae"]),
            "test_rmse": float(sensitivity_best_test["rmse"]),
            "test_r2": float(sensitivity_best_test["r2"]),
        },
        "selected_test_mae_by_input_group": {
            str(index): {
                str(key): float(value)
                for key, value in row.items()
                if pd.notna(value)
            }
            for index, row in group_conclusions.iterrows()
        },
    }
    (results_root / "08_protection_time_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_comparison = selected_group_test.sort_values(["landmark_s", "input_group"])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for input_group in INPUT_GROUPS:
        values = test_comparison.loc[test_comparison["input_group"] == input_group]
        ax.plot(values["landmark_s"], values["mae"], marker="o", label=input_group)
    ax.set(
        xlabel="Landmark (s)",
        ylabel="Test MAE (remaining time, s)",
        title="Landmark vs MAE: validation-selected models",
    )
    ax.set_xticks(list(LANDMARKS_S))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_root / "08_landmark_vs_mae.png", dpi=160)
    plt.close(fig)

    pivot = test_comparison.pivot(
        index="landmark_s", columns="input_group", values="mae"
    ).reindex(LANDMARKS_S)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set(
        xlabel="Landmark (s)",
        ylabel="Test MAE (remaining time, s)",
        title="Severity-only vs process-only vs combined",
    )
    ax.legend(title="input group")
    fig.tight_layout()
    fig.savefig(figures_root / "08_input_group_comparison.png", dpi=160)
    plt.close(fig)

    selected_predictions = predictions.loc[
        (predictions["mode"] == "strict")
        & (predictions["split"] == "test")
        & predictions["landmark_s"].eq(int(best_validation["landmark_s"]))
        & predictions["input_group"].eq(best_validation["input_group"])
        & predictions["model"].eq(best_validation["model"])
    ]
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.scatter(
        selected_predictions["remaining_time_s"],
        selected_predictions["prediction"],
        c=selected_predictions["severity"],
        cmap="viridis",
        s=45,
    )
    low = float(
        min(selected_predictions["remaining_time_s"].min(), selected_predictions["prediction"].min())
    )
    high = float(
        max(selected_predictions["remaining_time_s"].max(), selected_predictions["prediction"].max())
    )
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="ideal")
    ax.set(
        xlabel="Actual remaining time (s)",
        ylabel="Predicted remaining time (s)",
        title="Selected grouped test set",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_root / "08_predicted_vs_actual_remaining_time.png", dpi=160)
    plt.close(fig)

    assert all(
        (results_root / name).exists()
        for name in (
            "08_landmark_summary.csv",
            "08_window_sample_points.csv",
            "08_protection_time_metrics.csv",
            "08_protection_time_predictions.csv",
            "08_protection_time_split.csv",
            "08_leakage_audit.csv",
            "08_input_group_comparison.csv",
            "08_protection_time_summary.json",
        )
    )
    assert all(
        (figures_root / name).exists()
        for name in (
            "08_landmark_vs_mae.png",
            "08_input_group_comparison.png",
            "08_predicted_vs_actual_remaining_time.png",
        )
    )
    return {
        "dataset": dataset,
        "landmark_summary": landmark_summary,
        "metrics": metrics,
        "predictions": predictions,
        "leakage_audit": leakage_audit,
        "selected_group_test": selected_group_test,
        "summary": summary,
        "best_validation": best_validation,
        "best_test": best_test,
        "sensitivity_best_validation": sensitivity_best_validation,
        "sensitivity_best_test": sensitivity_best_test,
    }
