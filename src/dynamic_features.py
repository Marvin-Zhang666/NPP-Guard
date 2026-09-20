"""Normal-reference dynamic features and Mahalanobis early-warning evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

try:
    from .early_anomaly_detection import first_persistent_alarm
except ImportError:
    from early_anomaly_detection import first_persistent_alarm


GROUP_STATIC = "A_static"
GROUP_DELTA = "B_static_delta"
GROUP_SLOPE = "C_static_delta_slope"
FEATURE_GROUPS = (GROUP_STATIC, GROUP_DELTA, GROUP_SLOPE)
ROLLING_WINDOWS_S = (30.0, 60.0, 120.0)
THRESHOLD_QUANTILE = 0.995
MIN_CONSECUTIVE = 3
MAD_SCALE = 1.4826
NEAR_ZERO_MAD = 1e-10


@dataclass
class DynamicReferenceModel:
    feature_group: str
    candidate_features: list[str]
    feature_names: list[str]
    active_features: list[str]
    inactive_features: list[str]
    calibration_rows: int
    calibration_start_s: float
    calibration_end_s: float
    feature_warmup_s: float
    sample_interval_s: float
    rolling_windows_s: tuple[float, ...]
    center: pd.Series
    scale: pd.Series
    precision: np.ndarray
    covariance_condition_number: float
    covariance_shrinkage: float
    threshold: float
    normal_scores: pd.DataFrame


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing candidate features: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Trajectory contains non-finite candidate values")
    return values


def _validate_interval(time: pd.Series, sample_interval_s: float) -> None:
    values = pd.to_numeric(time, errors="coerce").to_numpy(float)
    differences = np.diff(values)
    if not np.isfinite(values).all() or len(differences) == 0:
        raise ValueError("TIME must contain a finite increasing grid")
    if not np.all(differences > 0) or not np.allclose(
        differences, sample_interval_s
    ):
        raise ValueError(
            f"Expected a strictly increasing {sample_interval_s:g}s TIME grid"
        )


def _rolling_slope(
    values: pd.Series,
    window_s: float,
    sample_interval_s: float,
) -> pd.Series:
    steps = int(round(window_s / sample_interval_s))
    window = steps + 1
    x = np.arange(window, dtype=float) * sample_interval_s
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))

    def slope(window_values: np.ndarray) -> float:
        centered_y = window_values - window_values.mean()
        return float(np.dot(centered_x, centered_y) / denominator)

    return values.rolling(window, min_periods=window).apply(slope, raw=True)


def feature_warmup_s(
    feature_group: str,
    sample_interval_s: float = 10.0,
    rolling_windows_s: tuple[float, ...] = ROLLING_WINDOWS_S,
) -> float:
    if feature_group == GROUP_STATIC:
        return 0.0
    if feature_group == GROUP_DELTA:
        return sample_interval_s
    if feature_group == GROUP_SLOPE:
        return max(rolling_windows_s)
    raise ValueError(f"Unknown feature group: {feature_group}")


def build_feature_frame(
    frame: pd.DataFrame,
    candidate_features: list[str],
    feature_group: str,
    sample_interval_s: float = 10.0,
    rolling_windows_s: tuple[float, ...] = ROLLING_WINDOWS_S,
) -> pd.DataFrame:
    """Build static, delta-rate, and full-window slope features."""
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {feature_group}")
    _validate_interval(frame["TIME"], sample_interval_s)
    values = _numeric(frame, candidate_features)
    features = pd.DataFrame({"TIME": pd.to_numeric(frame["TIME"])})
    for feature in candidate_features:
        features[feature] = values[feature].to_numpy(float)

    if feature_group in (GROUP_DELTA, GROUP_SLOPE):
        for feature in candidate_features:
            features[f"delta10s_rate__{feature}"] = (
                values[feature].diff() / sample_interval_s
            ).to_numpy(float)

    if feature_group == GROUP_SLOPE:
        for window_s in rolling_windows_s:
            label = int(window_s)
            for feature in candidate_features:
                features[f"slope{label}s__{feature}"] = _rolling_slope(
                    values[feature], window_s, sample_interval_s
                ).to_numpy(float)
    return features


def _standardize(values: np.ndarray, model: DynamicReferenceModel) -> np.ndarray:
    return (values - model.center.to_numpy(float)) / model.scale.to_numpy(float)


def score_frame(
    frame: pd.DataFrame,
    model: DynamicReferenceModel,
) -> pd.DataFrame:
    """Score one trajectory using a model calibrated only on Normal data."""
    features = build_feature_frame(
        frame,
        model.candidate_features,
        model.feature_group,
        sample_interval_s=model.sample_interval_s,
        rolling_windows_s=model.rolling_windows_s,
    )
    values = features[model.active_features].to_numpy(float)
    valid = np.isfinite(values).all(axis=1)
    scores = np.full(len(features), np.nan, dtype=float)
    if valid.any():
        z = _standardize(values[valid], model)
        scores[valid] = np.sqrt(
            np.maximum(
                0.0,
                np.einsum("ij,jk,ik->i", z, model.precision, z),
            )
        )
    return pd.DataFrame(
        {
            "TIME": features["TIME"].to_numpy(float),
            "mahalanobis": scores,
        }
    )


def fit_reference_model(
    normal: pd.DataFrame,
    candidate_features: list[str],
    feature_group: str,
    threshold_quantile: float = THRESHOLD_QUANTILE,
    sample_interval_s: float = 10.0,
    rolling_windows_s: tuple[float, ...] = ROLLING_WINDOWS_S,
) -> DynamicReferenceModel:
    """Fit all dynamic-model parameters from the Normal trajectory only."""
    feature_frame = build_feature_frame(
        normal,
        candidate_features,
        feature_group,
        sample_interval_s=sample_interval_s,
        rolling_windows_s=rolling_windows_s,
    )
    warmup_s = feature_warmup_s(
        feature_group, sample_interval_s, rolling_windows_s
    )
    feature_names = [column for column in feature_frame if column != "TIME"]
    valid = feature_frame["TIME"].ge(warmup_s) & feature_frame[feature_names].notna().all(axis=1)
    calibration = feature_frame.loc[valid, feature_names].apply(
        pd.to_numeric, errors="coerce"
    )
    if len(calibration) < 3:
        raise ValueError("Need at least three complete Normal calibration rows")

    medians = calibration.median()
    mad = (calibration - medians).abs().median()
    active_features = [feature for feature in feature_names if mad[feature] > NEAR_ZERO_MAD]
    inactive_features = [feature for feature in feature_names if feature not in active_features]
    if len(active_features) < 2:
        raise ValueError("Too few non-degenerate Normal dynamic features")

    calibration = calibration[active_features]
    medians = calibration.median()
    mad = (calibration - medians).abs().median()
    scale = pd.concat(
        [MAD_SCALE * mad, 0.01 * calibration.abs().median()],
        axis=1,
    ).max(axis=1).clip(lower=1e-12)
    z_calibration = (calibration - medians) / scale
    covariance = LedoitWolf(assume_centered=True).fit(z_calibration.to_numpy(float))
    calibration_end_s = float(feature_frame["TIME"].max())
    model = DynamicReferenceModel(
        feature_group=feature_group,
        candidate_features=list(candidate_features),
        feature_names=feature_names,
        active_features=active_features,
        inactive_features=inactive_features,
        calibration_rows=len(calibration),
        calibration_start_s=warmup_s,
        calibration_end_s=calibration_end_s,
        feature_warmup_s=warmup_s,
        sample_interval_s=sample_interval_s,
        rolling_windows_s=rolling_windows_s,
        center=medians,
        scale=scale,
        precision=covariance.precision_,
        covariance_condition_number=float(np.linalg.cond(covariance.covariance_)),
        covariance_shrinkage=float(covariance.shrinkage_),
        threshold=float("nan"),
        normal_scores=pd.DataFrame(),
    )
    normal_scores = score_frame(normal, model)
    model.normal_scores = normal_scores
    model.threshold = float(
        normal_scores["mahalanobis"].dropna().quantile(threshold_quantile)
    )
    return model


def evaluate_dynamic_group(
    normal: pd.DataFrame,
    loca_dir: str | Path,
    events: pd.DataFrame,
    candidate_features: list[str],
    feature_group: str,
    threshold_quantile: float = THRESHOLD_QUANTILE,
    min_consecutive: int = MIN_CONSECUTIVE,
    sample_interval_s: float = 10.0,
    rolling_windows_s: tuple[float, ...] = ROLLING_WINDOWS_S,
) -> tuple[DynamicReferenceModel, pd.DataFrame, pd.DataFrame]:
    """Evaluate one dynamic feature group over all 100 LOCA trajectories."""
    model = fit_reference_model(
        normal,
        candidate_features,
        feature_group,
        threshold_quantile=threshold_quantile,
        sample_interval_s=sample_interval_s,
        rolling_windows_s=rolling_windows_s,
    )
    normal_alarm = first_persistent_alarm(
        model.normal_scores["TIME"].to_numpy(float),
        model.normal_scores["mahalanobis"].to_numpy(float),
        model.threshold,
        start_after_s=0.0,
        min_consecutive=min_consecutive,
    )

    rows: list[dict] = []
    for severity in range(1, 101):
        frame = pd.read_csv(Path(loca_dir) / f"{severity}.csv")
        event = events.loc[events["severity"].eq(severity)].iloc[0]
        scores = score_frame(frame, model)
        detection_time = first_persistent_alarm(
            scores["TIME"].to_numpy(float),
            scores["mahalanobis"].to_numpy(float),
            model.threshold,
            start_after_s=float(event["injection_s"]),
            min_consecutive=min_consecutive,
        )
        protection_time = float(event["first_protection_s"])
        rows.append(
            {
                "sample_id": event["sample_id"],
                "severity": severity,
                "feature_group": feature_group,
                "method": "mahalanobis",
                "threshold": model.threshold,
                "earliest_valid_feature_time_s": model.feature_warmup_s,
                "earliest_possible_confirmed_alarm_s": model.feature_warmup_s
                + (min_consecutive - 1) * sample_interval_s,
                "detection_time_s": detection_time,
                "first_protection_time_s": protection_time,
                "lead_time_s": protection_time - detection_time
                if np.isfinite(detection_time)
                else float("nan"),
                "detected": bool(np.isfinite(detection_time)),
                "detected_before_protection": bool(
                    np.isfinite(detection_time) and detection_time <= protection_time
                ),
            }
        )

    results = pd.DataFrame(rows)
    detected = results.loc[results["detected"]]
    before = results.loc[results["detected_before_protection"]]
    summary = pd.DataFrame(
        [
            {
                "feature_group": feature_group,
                "method": "mahalanobis",
                "base_candidate_feature_count": len(candidate_features),
                "candidate_feature_count": len(model.feature_names),
                "active_feature_count": len(model.active_features),
                "inactive_near_zero_feature_count": len(model.inactive_features),
                "calibration_rows": model.calibration_rows,
                "calibration_start_s": model.calibration_start_s,
                "calibration_end_s": model.calibration_end_s,
                "feature_warmup_s": model.feature_warmup_s,
                "earliest_possible_confirmed_alarm_s": model.feature_warmup_s
                + (min_consecutive - 1) * sample_interval_s,
                "threshold_quantile": threshold_quantile,
                "threshold": model.threshold,
                "covariance_condition_number": model.covariance_condition_number,
                "covariance_shrinkage": model.covariance_shrinkage,
                "loca_count": len(results),
                "detected_count": len(detected),
                "protection_before_detected_count": len(before),
                "protection_before_detected_rate": len(before) / len(results),
                "detection_time_median_s": detected["detection_time_s"].median(),
                "detection_time_min_s": detected["detection_time_s"].min(),
                "detection_time_max_s": detected["detection_time_s"].max(),
                "lead_time_median_s": detected["lead_time_s"].median(),
                "lead_time_min_s": detected["lead_time_s"].min(),
                "lead_time_max_s": detected["lead_time_s"].max(),
                "normal_reference_alarm_time_s": normal_alarm,
                "normal_reference_false_alarm": bool(np.isfinite(normal_alarm)),
                "normal_points_above_threshold": int(
                    model.normal_scores["mahalanobis"].gt(model.threshold).sum()
                ),
            }
        ]
    )
    return model, results, summary


if __name__ == "__main__":
    demo = pd.DataFrame({"TIME": np.arange(0.0, 141.0, 10.0), "P": 1.0})
    demo.loc[demo["TIME"] >= 120.0, "P"] += demo["TIME"] / 1000.0
    assert feature_warmup_s(GROUP_SLOPE) == 120.0
    assert build_feature_frame(demo, ["P"], GROUP_SLOPE)["slope120s__P"].iloc[:12].isna().all()
    print("dynamic_features self-check passed")
