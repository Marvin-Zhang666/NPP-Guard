"""Normal-reference early anomaly detection baselines for the LOCA study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


METHODS = (
    "robust_z_score",
    "mahalanobis",
    "pca_reconstruction_error",
)
THRESHOLD_QUANTILE = 0.995
MIN_CONSECUTIVE = 3
NEAR_ZERO_MAD = 1e-10
MAD_SCALE = 1.4826


@dataclass
class ReferenceModel:
    candidate_features: list[str]
    active_features: list[str]
    inactive_features: list[str]
    calibration_rows: int
    calibration_end_s: float
    center: pd.Series
    scale: pd.Series
    precision: np.ndarray
    covariance_condition_number: float
    covariance_shrinkage: float
    pca: PCA
    thresholds: dict[str, float]
    normal_scores: pd.DataFrame


def _numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Reference or case data contains non-finite candidate values")
    return values


def _standardize(values: np.ndarray, model: ReferenceModel) -> np.ndarray:
    center = model.center.to_numpy(float)
    scale = model.scale.to_numpy(float)
    return (values - center) / scale


def score_frame(frame: pd.DataFrame, model: ReferenceModel) -> pd.DataFrame:
    """Score one trajectory using a model fitted only on Normal reference data."""
    values = _numeric(frame, model.active_features)
    z = _standardize(values, model)
    robust_z = np.median(np.abs(z), axis=1)
    mahalanobis = np.sqrt(
        np.maximum(0.0, np.einsum("ij,jk,ik->i", z, model.precision, z))
    )
    reconstruction = model.pca.inverse_transform(model.pca.transform(z))
    pca_error = np.sqrt(np.mean((z - reconstruction) ** 2, axis=1))
    return pd.DataFrame(
        {
            "TIME": pd.to_numeric(frame["TIME"], errors="coerce").to_numpy(float),
            "robust_z_score": robust_z,
            "mahalanobis": mahalanobis,
            "pca_reconstruction_error": pca_error,
        }
    )


def fit_reference_model(
    normal: pd.DataFrame,
    candidate_features: list[str],
    calibration_end_s: float = 100.0,
    threshold_quantile: float = THRESHOLD_QUANTILE,
) -> ReferenceModel:
    """Fit robust baselines and derive all thresholds from Normal only."""
    calibration = normal.loc[normal["TIME"] <= calibration_end_s, candidate_features]
    if len(calibration) < 3:
        raise ValueError("Need at least three Normal calibration rows")
    calibration = calibration.apply(pd.to_numeric, errors="coerce")
    if calibration.isna().any().any():
        raise ValueError("Normal calibration contains missing candidate values")

    medians = calibration.median()
    mad = (calibration - medians).abs().median()
    active_features = [feature for feature in candidate_features if mad[feature] > NEAR_ZERO_MAD]
    inactive_features = [feature for feature in candidate_features if feature not in active_features]
    if len(active_features) < 2:
        raise ValueError("Too few non-degenerate Normal candidate features")

    calibration = calibration[active_features]
    medians = calibration.median()
    mad = (calibration - medians).abs().median()
    scale = pd.concat(
        [MAD_SCALE * mad, 0.01 * calibration.abs().median()],
        axis=1,
    ).max(axis=1).clip(lower=1e-12)
    z_calibration = (calibration - medians) / scale
    z_values = z_calibration.to_numpy(float)

    covariance = LedoitWolf(assume_centered=True).fit(z_values)
    pca = PCA(n_components=0.95, svd_solver="full").fit(z_values)
    model = ReferenceModel(
        candidate_features=list(candidate_features),
        active_features=active_features,
        inactive_features=inactive_features,
        calibration_rows=len(calibration),
        calibration_end_s=calibration_end_s,
        center=medians,
        scale=scale,
        precision=covariance.precision_,
        covariance_condition_number=float(np.linalg.cond(covariance.covariance_)),
        covariance_shrinkage=float(covariance.shrinkage_),
        pca=pca,
        thresholds={},
        normal_scores=pd.DataFrame(),
    )
    normal_scores = score_frame(normal, model)
    model.normal_scores = normal_scores
    model.thresholds = {
        method: float(normal_scores[method].quantile(threshold_quantile))
        for method in METHODS
    }
    return model


def first_persistent_alarm(
    times: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    start_after_s: float,
    min_consecutive: int = MIN_CONSECUTIVE,
) -> float:
    """Return the third confirming sample time, or NaN if no run is found."""
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    above = np.isfinite(scores) & (times > start_after_s) & (scores > threshold)
    run = 0
    for index, is_above in enumerate(above):
        run = run + 1 if is_above else 0
        if run >= min_consecutive:
            return float(times[index])
    return float("nan")


def evaluate_reference(
    normal: pd.DataFrame,
    loca_dir: str | Path,
    events: pd.DataFrame,
    candidate_features: list[str],
    calibration_end_s: float = 100.0,
    threshold_quantile: float = THRESHOLD_QUANTILE,
    min_consecutive: int = MIN_CONSECUTIVE,
) -> tuple[ReferenceModel, pd.DataFrame, pd.DataFrame]:
    """Evaluate all LOCA trajectories and return model, case rows, and summary."""
    model = fit_reference_model(
        normal,
        candidate_features,
        calibration_end_s=calibration_end_s,
        threshold_quantile=threshold_quantile,
    )
    normal_rows = []
    for method in METHODS:
        alarm = first_persistent_alarm(
            model.normal_scores["TIME"].to_numpy(),
            model.normal_scores[method].to_numpy(),
            model.thresholds[method],
            start_after_s=0.0,
            min_consecutive=min_consecutive,
        )
        normal_rows.append(
            {
                "method": method,
                "normal_reference_alarm_time_s": alarm,
                "normal_reference_false_alarm": bool(np.isfinite(alarm)),
                "normal_points_above_threshold": int(
                    (model.normal_scores[method] > model.thresholds[method]).sum()
                ),
            }
        )

    rows: list[dict] = []
    for severity in range(1, 101):
        frame = pd.read_csv(Path(loca_dir) / f"{severity}.csv")
        event = events.loc[events["severity"].eq(severity)].iloc[0]
        scores = score_frame(frame, model)
        for method in METHODS:
            detection_time = first_persistent_alarm(
                scores["TIME"].to_numpy(),
                scores[method].to_numpy(),
                model.thresholds[method],
                start_after_s=float(event["injection_s"]),
                min_consecutive=min_consecutive,
            )
            protection_time = float(event["first_protection_s"])
            lead_time = (
                protection_time - detection_time
                if np.isfinite(detection_time)
                else float("nan")
            )
            rows.append(
                {
                    "sample_id": event["sample_id"],
                    "severity": severity,
                    "method": method,
                    "threshold": model.thresholds[method],
                    "detection_time_s": detection_time,
                    "first_protection_time_s": protection_time,
                    "lead_time_s": lead_time,
                    "detected": bool(np.isfinite(detection_time)),
                    "detected_before_protection": bool(
                        np.isfinite(detection_time) and detection_time <= protection_time
                    ),
                }
            )

    results = pd.DataFrame(rows)
    summaries = []
    for method in METHODS:
        method_rows = results.loc[results["method"].eq(method)]
        normal = next(row for row in normal_rows if row["method"] == method)
        detected = method_rows.loc[method_rows["detected"]]
        before = method_rows.loc[method_rows["detected_before_protection"]]
        summaries.append(
            {
                "method": method,
                "candidate_feature_count": len(candidate_features),
                "active_feature_count": len(model.active_features),
                "inactive_near_zero_feature_count": len(model.inactive_features),
                "calibration_rows": model.calibration_rows,
                "threshold_quantile": threshold_quantile,
                "threshold": model.thresholds[method],
                "pca_components": model.pca.n_components_,
                "covariance_condition_number": model.covariance_condition_number,
                "covariance_shrinkage": model.covariance_shrinkage,
                "loca_count": len(method_rows),
                "detected_count": len(detected),
                "protection_before_detected_count": len(before),
                "protection_before_detected_rate": len(before) / len(method_rows),
                "detection_time_median_s": detected["detection_time_s"].median(),
                "detection_time_min_s": detected["detection_time_s"].min(),
                "detection_time_max_s": detected["detection_time_s"].max(),
                "lead_time_median_s": detected["lead_time_s"].median(),
                "lead_time_min_s": detected["lead_time_s"].min(),
                "lead_time_max_s": detected["lead_time_s"].max(),
                **normal,
            }
        )
    return model, results, pd.DataFrame(summaries)


def write_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the two light-weight result tables used by the Notebook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "early_anomaly_detection_results.csv",
        "summary": output_dir / "early_anomaly_detection_method_summary.csv",
    }
    results.to_csv(paths["results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths
