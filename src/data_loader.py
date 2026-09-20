"""Reusable loading and early-warning summaries for NPPAD LOCA CSV cases."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPERATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "NuclearPowerPlantAccidentData"
    / "Operation_csv_data"
)
DEFAULT_LOCA_DIR = OPERATION_ROOT / "LOCA"
DEFAULT_NORMAL_FILE = OPERATION_ROOT / "Normal" / "1.csv"
DEFAULT_TRANSIENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "NuclearPowerPlantAccidentData"
    / "NPPAD"
    / "LOCA"
)

LOCA_SEVERITY_UNIT = "% of 100 cm2 break area"
DIRECT_ACCIDENT_FEATURES = ("WLR", "WBK", "MBK")
EARLY_WARNING_FEATURES = (
    "P",
    "TAVG",
    "LVPZ",
    "WECS",
    "PWR",
    "SCMA",
    "TRB",
    "PRB",
    "LWRB",
    "PRBA",
    "DNBR",
)


def _sample_number(path: Path) -> int:
    if not path.stem.isdigit():
        raise ValueError(f"Expected a numeric LOCA filename, got {path.name}")
    return int(path.stem)


def _sample_id(number: int) -> str:
    return f"LOCA_{number:03d}"


def loca_paths(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    expected_ids: range | list[int] = range(1, 101),
) -> list[Path]:
    """Return expected LOCA CSV paths in numeric sample order."""
    directory = Path(loca_dir)
    by_number = {
        _sample_number(path): path
        for path in directory.glob("*.csv")
        if path.stem.isdigit()
    }
    expected = [int(number) for number in expected_ids]
    missing = [number for number in expected if number not in by_number]
    if missing:
        raise FileNotFoundError(
            f"Missing LOCA CSV files in {directory}: {missing}"
        )
    return [by_number[number] for number in expected]


def validate_loca_files(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    expected_ids: range | list[int] = range(1, 101),
) -> pd.DataFrame:
    """Validate file presence, schema, missing values, and TIME axes."""
    directory = Path(loca_dir)
    expected = [int(number) for number in expected_ids]
    numeric_files = {
        _sample_number(path): path
        for path in directory.glob("*.csv")
        if path.stem.isdigit()
    }
    extra_files = sorted(set(numeric_files) - set(expected))
    reference_columns: list[str] | None = None
    rows: list[dict] = []

    for number in expected:
        path = numeric_files.get(number)
        row = {
            "sample_id": _sample_id(number),
            "severity": number,
            "severity_unit": LOCA_SEVERITY_UNIT,
            "file": path.name if path else f"{number}.csv",
            "exists": path is not None,
            "error": None,
            "rows": 0,
            "columns": 0,
            "missing_cells": 0,
            "missing_columns": 0,
            "column_match": False,
            "time_numeric": False,
            "time_strictly_increasing": False,
            "time_duplicate_count": 0,
            "time_start_s": np.nan,
            "time_end_s": np.nan,
            "time_step_s": np.nan,
            "time_step_min_s": np.nan,
            "time_step_max_s": np.nan,
            "time_grid_10s": False,
        }
        if path is None:
            rows.append(row)
            continue

        try:
            frame = pd.read_csv(path)
            if reference_columns is None:
                reference_columns = frame.columns.tolist()
            time = pd.to_numeric(frame["TIME"], errors="coerce")
            differences = time.diff().dropna()
            row.update(
                {
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "missing_cells": int(frame.isna().sum().sum()),
                    "missing_columns": int(frame.isna().any().sum()),
                    "column_match": frame.columns.tolist() == reference_columns,
                    "time_numeric": bool(time.notna().all()),
                    "time_strictly_increasing": bool(differences.gt(0).all()),
                    "time_duplicate_count": int(time.duplicated().sum()),
                    "time_start_s": float(time.iloc[0]),
                    "time_end_s": float(time.iloc[-1]),
                    "time_step_s": float(differences.median()),
                    "time_step_min_s": float(differences.min()),
                    "time_step_max_s": float(differences.max()),
                    "time_grid_10s": bool(
                        differences.size > 0
                        and np.allclose(differences.to_numpy(), 10.0)
                    ),
                }
            )
        except Exception as exc:  # Report the file problem instead of hiding it.
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["extra_numeric_files"] = extra_files
    result.attrs["reference_columns"] = reference_columns or []
    return result


def load_loca_sample(
    path: str | Path,
    add_metadata: bool = True,
) -> pd.DataFrame:
    """Load one LOCA CSV and optionally add sample-level metadata columns."""
    csv_path = Path(path)
    number = _sample_number(csv_path)
    frame = pd.read_csv(csv_path)
    if not add_metadata:
        return frame

    metadata = pd.DataFrame(
        {
            "sample_id": [_sample_id(number)] * len(frame),
            "severity": [number] * len(frame),
            "severity_unit": [LOCA_SEVERITY_UNIT] * len(frame),
            "accident": ["LOCA"] * len(frame),
            "source_file": [csv_path.name] * len(frame),
        },
        index=frame.index,
    )
    return pd.concat([metadata, frame], axis=1)


def iter_loca_samples(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    expected_ids: range | list[int] = range(1, 101),
) -> Iterator[pd.DataFrame]:
    """Yield each metadata-enriched LOCA sample in numeric order."""
    for path in loca_paths(loca_dir, expected_ids):
        yield load_loca_sample(path)


def load_loca_dataset(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    expected_ids: range | list[int] = range(1, 101),
) -> pd.DataFrame:
    """Load all expected LOCA records into one in-memory analysis frame."""
    return pd.concat(
        iter_loca_samples(loca_dir, expected_ids),
        ignore_index=True,
    )


def _first_persistent_time(
    series: pd.Series,
    threshold: float = 3.0,
    min_consecutive: int = 3,
) -> float | None:
    series = series.dropna().sort_index()
    above = series.gt(threshold)
    run_length = above.groupby((~above).cumsum()).cumcount() + 1
    alarm = above & run_length.ge(min_consecutive)
    times = series.index[alarm.to_numpy()]
    return float(times[0]) if len(times) else None


def _baseline_scale(
    reference: pd.DataFrame,
    features: list[str],
    baseline_end_s: float,
) -> pd.Series:
    baseline = reference.loc[reference["TIME"] <= baseline_end_s, features]
    scale = pd.concat(
        [baseline.std(), 0.01 * baseline.abs().median()],
        axis=1,
    ).max(axis=1)
    return scale.mask(scale < 1e-6)


def _event_time(report: Path, pattern: str) -> float | None:
    if not report.exists():
        return None
    text = report.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        rf"([0-9]+(?:\.[0-9]+)?) sec, {pattern}",
        text,
    )
    return float(match.group(1)) if match else None


def build_loca_summary(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    reference_path: str | Path = DEFAULT_NORMAL_FILE,
    transient_dir: str | Path = DEFAULT_TRANSIENT_DIR,
    features: tuple[str, ...] = EARLY_WARNING_FEATURES,
    baseline_end_s: float = 100.0,
    early_limit_s: float = 2000.0,
    score_threshold: float = 3.0,
    min_consecutive: int = 3,
) -> pd.DataFrame:
    """Build one small row per case for severity and early-warning analysis."""
    reference = pd.read_csv(reference_path)
    features = list(features)
    missing = [feature for feature in features if feature not in reference]
    if missing:
        raise ValueError(f"Features missing from reference CSV: {missing}")
    scale = _baseline_scale(reference, features, baseline_end_s)
    rows: list[dict] = []

    for path in loca_paths(loca_dir):
        number = _sample_number(path)
        frame = pd.read_csv(path)
        missing = [feature for feature in features if feature not in frame]
        if missing:
            raise ValueError(f"Features missing from {path.name}: {missing}")

        merged = reference[["TIME", *features]].merge(
            frame[["TIME", *features]],
            on="TIME",
            how="inner",
            suffixes=("_normal", "_loca"),
        )
        merged = merged.loc[merged["TIME"] <= early_limit_s]
        scores = pd.DataFrame({"TIME": merged["TIME"]}).set_index("TIME")
        for feature in features:
            scores[feature] = (
                merged[f"{feature}_loca"]
                - merged[f"{feature}_normal"]
            ).abs() / float(scale[feature])

        row = {
            "sample_id": _sample_id(number),
            "severity": number,
            "severity_unit": LOCA_SEVERITY_UNIT,
            "source_file": path.name,
            "rows": len(frame),
            "time_end_s": float(frame["TIME"].iloc[-1]),
        }
        for feature in features:
            row[f"{feature}_first_deviation_s"] = _first_persistent_time(
                scores[feature], score_threshold, min_consecutive
            )
            row[f"{feature}_peak_score_early"] = float(scores[feature].max())

        composite = scores[features].median(axis=1)
        row["indirect_composite_first_deviation_s"] = _first_persistent_time(
            composite, score_threshold, min_consecutive
        )
        row["indirect_features"] = ",".join(features)

        report = Path(transient_dir) / f"{number}Transient Report.txt"
        row["hpi_pump_start_s"] = _event_time(
            report, r"HPI Pump #1 Position Change: 100%"
        )
        row["scram_s"] = _event_time(report, r"Reactor Scram")
        if "WHPI" in frame:
            whpi = pd.to_numeric(frame["WHPI"], errors="coerce")
            whpi_times = frame.loc[whpi.gt(1e-9), "TIME"]
        else:
            whpi_times = pd.Series(dtype=float)
        row["whpi_first_positive_s"] = (
            float(whpi_times.iloc[0]) if len(whpi_times) else None
        )
        warning_time = row["indirect_composite_first_deviation_s"]
        hpi_time = row["hpi_pump_start_s"]
        row["lead_time_to_hpi_s"] = (
            hpi_time - warning_time
            if warning_time is not None and hpi_time is not None
            else None
        )
        row["lead_time_to_scram_s"] = (
            row["scram_s"] - warning_time
            if warning_time is not None and row["scram_s"] is not None
            else None
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("severity").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "loca_severity_summary.csv",
    )
    args = parser.parse_args()

    validation = validate_loca_files()
    invalid = validation[
        (~validation["exists"])
        | validation["error"].notna()
        | (~validation["column_match"])
        | (~validation["time_grid_10s"])
        | (validation["missing_cells"] > 0)
    ]
    if len(invalid):
        raise RuntimeError(
            "LOCA validation failed:\n" + invalid.to_string(index=False)
        )

    summary = build_loca_summary()
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    print(f"Validated {len(validation)} LOCA files")
    print(f"Total raw rows: {int(validation['rows'].sum())}")
    print(f"Columns: {int(validation['columns'].iloc[0])}")
    print(
        "Time range: "
        f"{validation['time_end_s'].min():.0f}"
        f"-{validation['time_end_s'].max():.0f} s; "
        "10 s grid consistent: "
        f"{bool(validation['time_grid_10s'].all())}"
    )
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
