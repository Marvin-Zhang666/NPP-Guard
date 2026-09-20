"""Feature extraction for grouped LOCA severity estimation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loader import DEFAULT_LOCA_DIR, loca_paths
except ImportError:
    from data_loader import DEFAULT_LOCA_DIR, loca_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_GROUPS = PROJECT_ROOT / "results" / "feature_groups.csv"
WINDOWS_S = (30, 60, 120, 300)
INJECTION_S = 0.5
SUMMARY_STATISTICS = ("last", "delta_t0", "mean", "std", "slope")


def candidate_process_features(
    feature_groups_path: str | Path = DEFAULT_FEATURE_GROUPS,
) -> list[str]:
    """Load the strict process-only candidate list from the 03 audit."""
    table = pd.read_csv(feature_groups_path)
    candidates = table.loc[table["ml_policy"].eq("candidate"), "feature"].tolist()
    if len(candidates) != 38 or table["feature"].duplicated().any():
        raise ValueError("feature_groups.csv is not the expected 38-feature audit")
    return candidates


def _window_rows(
    frame: pd.DataFrame,
    window_s: int,
    injection_s: float = INJECTION_S,
) -> pd.DataFrame:
    time = pd.to_numeric(frame["TIME"], errors="coerce")
    rows = frame.loc[time.gt(injection_s) & time.le(injection_s + window_s)].copy()
    if rows.empty:
        raise ValueError(f"No post-injection samples in {window_s}s window")
    return rows


def window_sample_points(
    frame: pd.DataFrame,
    window_s: int,
    injection_s: float = INJECTION_S,
) -> np.ndarray:
    """Return the actual CSV timestamps included in one observation window."""
    return pd.to_numeric(
        _window_rows(frame, window_s, injection_s)["TIME"], errors="coerce"
    ).to_numpy(dtype=float)


def _linear_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    centered_time = time_s - time_s.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator == 0.0:
        return 0.0
    centered_values = values - values.mean()
    return float(np.dot(centered_time, centered_values) / denominator)


def extract_window_features(
    frame: pd.DataFrame,
    features: list[str],
    window_s: int,
    injection_s: float = INJECTION_S,
) -> dict[str, float]:
    """Extract interpretable endpoint, baseline-change, distribution and slope features."""
    missing = [feature for feature in ["TIME", *features] if feature not in frame]
    if missing:
        raise ValueError(f"Missing LOCA columns: {missing}")

    baseline_rows = frame.loc[
        pd.to_numeric(frame["TIME"], errors="coerce").le(injection_s)
    ]
    if baseline_rows.empty:
        raise ValueError("LOCA frame has no t=0 baseline row")
    baseline = baseline_rows.iloc[-1]
    rows = _window_rows(frame, window_s, injection_s)
    time = pd.to_numeric(rows["TIME"], errors="coerce").to_numpy(dtype=float)
    relative_time = time - injection_s
    result: dict[str, float] = {}
    for feature in features:
        baseline_value = float(pd.to_numeric(pd.Series([baseline[feature]]), errors="coerce").iloc[0])
        values = pd.to_numeric(rows[feature], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(baseline_value):
            raise ValueError(f"Non-finite values in {feature}, window={window_s}")
        result[f"{feature}__last"] = float(values[-1])
        result[f"{feature}__delta_t0"] = float(values[-1] - baseline_value)
        result[f"{feature}__mean"] = float(values.mean())
        result[f"{feature}__std"] = float(values.std(ddof=0))
        result[f"{feature}__slope"] = _linear_slope(relative_time, values)
    return result


def build_severity_features(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    windows_s: tuple[int, ...] = WINDOWS_S,
    injection_s: float = INJECTION_S,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one feature row per LOCA trajectory and window plus a point audit."""
    features = features or candidate_process_features()
    rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for path in loca_paths(loca_dir):
        severity = int(path.stem)
        frame = pd.read_csv(path)
        for window_s in windows_s:
            points = window_sample_points(frame, window_s, injection_s)
            extracted = extract_window_features(
                frame, features, window_s, injection_s
            )
            rows.append(
                {
                    "sample_id": f"LOCA_{severity:03d}",
                    "severity": severity,
                    "window_s": window_s,
                    **extracted,
                }
            )
            point_rows.append(
                {
                    "window_s": window_s,
                    "sample_id": f"LOCA_{severity:03d}",
                    "sample_count": len(points),
                    "first_sample_s": float(points[0]),
                    "last_sample_s": float(points[-1]),
                    "sample_points_s": ",".join(f"{point:g}" for point in points),
                }
            )

    dataset = pd.DataFrame(rows).sort_values(["window_s", "severity"]).reset_index(drop=True)
    points = pd.DataFrame(point_rows).sort_values(["window_s", "sample_id"]).reset_index(drop=True)
    return dataset, points
