"""Leakage-safe landmark features for LOCA protection-time prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .data_loader import DEFAULT_LOCA_DIR, loca_paths
    from .severity_features import (
        INJECTION_S,
        candidate_process_features,
        extract_window_features,
        window_sample_points,
    )
except ImportError:
    from data_loader import DEFAULT_LOCA_DIR, loca_paths
    from severity_features import (
        INJECTION_S,
        candidate_process_features,
        extract_window_features,
        window_sample_points,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_TABLE = PROJECT_ROOT / "results" / "loca_event_table.csv"
LANDMARKS_S = (30, 60, 120, 300)


def build_protection_time_dataset(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    event_table_path: str | Path = DEFAULT_EVENT_TABLE,
    landmarks_s: tuple[int, ...] = LANDMARKS_S,
    injection_s: float = INJECTION_S,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per eligible trajectory and landmark.

    A trajectory is eligible only when its first protection time is strictly
    after the landmark. The event time is retained as metadata and the target,
    never as an input feature.
    """
    features = features or candidate_process_features()
    events = pd.read_csv(event_table_path)
    required = {"sample_id", "severity", "first_protection_s"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event table is missing columns: {sorted(missing)}")
    if events["sample_id"].duplicated().any():
        raise ValueError("Event table has duplicate sample_id values")
    events = events.set_index("sample_id")
    rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    for path in loca_paths(loca_dir):
        sample_id = f"LOCA_{int(path.stem):03d}"
        if sample_id not in events.index:
            raise ValueError(f"Missing event row for {sample_id}")
        event = events.loc[sample_id]
        first_protection = float(event["first_protection_s"])
        severity = int(event["severity"])
        frame = pd.read_csv(path)
        for landmark_s in landmarks_s:
            if not first_protection > landmark_s:
                continue
            points = window_sample_points(frame, landmark_s, injection_s)
            extracted = extract_window_features(
                frame, features, landmark_s, injection_s
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "severity": severity,
                    "landmark_s": int(landmark_s),
                    "first_protection_s": first_protection,
                    "remaining_time_s": first_protection - landmark_s,
                    **extracted,
                }
            )
            point_rows.append(
                {
                    "landmark_s": int(landmark_s),
                    "sample_id": sample_id,
                    "severity": severity,
                    "first_protection_s": first_protection,
                    "sample_count": len(points),
                    "first_sample_s": float(points[0]),
                    "last_sample_s": float(points[-1]),
                    "sample_points_s": ",".join(f"{point:g}" for point in points),
                }
            )

    dataset = pd.DataFrame(rows).sort_values(
        ["landmark_s", "severity"]
    ).reset_index(drop=True)
    points = pd.DataFrame(point_rows).sort_values(
        ["landmark_s", "sample_id"]
    ).reset_index(drop=True)
    if dataset.empty or dataset["sample_id"].duplicated().all():
        raise ValueError("No valid landmark rows were constructed")
    return dataset, points

