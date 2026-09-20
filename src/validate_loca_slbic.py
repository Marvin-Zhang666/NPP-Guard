"""Leakage-aware early validation for separating LOCA and SLBIC."""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from validate_npp_guard import build_scale, score_against_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
CSV_ROOT = DATA_ROOT / "Operation_csv_data"
REPORT_ROOT = DATA_ROOT / "NPPAD"
RESULT_ROOT = PROJECT_ROOT / "results"

# Deliberately excludes direct break indicators and ECCS/safety-injection flow.
EARLY_FEATURES = [
    "P", "TSAT", "HUP", "LVCR", "QMWT", "WSTB", "WSTA", "WFWB",
    "WFWA", "QMGA", "QMGB", "TFSB", "TFPK", "TF", "TBLD", "DNBR",
    "SCMA", "SCMB", "VOL", "TRB", "PRB", "RRCB", "WRCB", "RRCO",
    "WRCA", "RRCA",
]
ACCIDENTS = ["LOCA", "SLBIC"]
FOLDS = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]


def protection_time(accident: str, case: int) -> float | None:
    report = REPORT_ROOT / accident / f"{case}Transient Report.txt"
    if not report.exists():
        return None
    pattern = r"([0-9]+(?:\.[0-9]+)?)\s*sec,\s*(?:HPSI|HPI|Reactor Scram|Scram)"
    times = [float(match.group(1)) for match in re.finditer(
        pattern, report.read_text(errors="ignore")
    )]
    return min(times) if times else None


def case_vector(
    reference: pd.DataFrame,
    scale: pd.Series,
    accident: str,
    case: int,
    cutoff: float,
) -> np.ndarray:
    test = pd.read_csv(CSV_ROOT / accident / f"{case}.csv")
    scores = score_against_reference(
        reference, test, EARLY_FEATURES, scale, limit=None
    )
    scores = scores.loc[scores.index.to_numpy(dtype=float) <= cutoff]
    values = []
    for feature in EARLY_FEATURES:
        series = scores[feature].to_numpy(dtype=float)
        values.extend([
            np.nanmedian(series),
            np.nanmean(series),
            np.nanmax(series),
            series[-1],
        ])
    return np.asarray(values, dtype=float)


def build_dataset(
    reference: pd.DataFrame,
    scale: pd.Series,
    cutoff: float,
) -> pd.DataFrame:
    rows = []
    for accident in ACCIDENTS:
        paths = sorted(
            (CSV_ROOT / accident).glob("*.csv"),
            key=lambda path: int(path.stem),
        )
        for path in paths:
            case = int(path.stem)
            rows.append({
                "accident": accident,
                "case": case,
                "protection_s": protection_time(accident, case),
                "vector": case_vector(
                    reference, scale, accident, case, cutoff
                ),
            })
    return pd.DataFrame(rows)


def evaluate_cutoff(
    dataset: pd.DataFrame,
    cutoff: float,
) -> dict:
    vectors = np.vstack(dataset["vector"])
    fill = np.nanmedian(vectors, axis=0)
    vectors = np.where(np.isfinite(vectors), vectors, fill)
    folds = []
    for low, high in FOLDS:
        test = dataset[dataset["case"].between(low, high)]
        train = ~dataset.index.isin(test.index)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        )
        model.fit(vectors[train], dataset.loc[train, "accident"])
        predicted = model.predict(vectors[test.index])
        actual = dataset.loc[test.index, "accident"]
        folds.append({
            "fold": f"{low}-{high}",
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ),
        })

    audit = {}
    for accident in ACCIDENTS:
        subset = dataset[dataset["accident"] == accident]
        known_subset = subset["protection_s"].notna()
        audit[accident] = {
            "reports": int(len(subset)),
            "known_protection_time": int(known_subset.sum()),
            "protection_at_or_before_cutoff": int(
                (known_subset & (subset["protection_s"] <= cutoff)).sum()
            ),
            "fraction_protection_at_or_before_cutoff": float(
                (known_subset & (subset["protection_s"] <= cutoff)).sum()
                / known_subset.sum()
            ) if known_subset.any() else None,
        }
    return {
        "cutoff_s": cutoff,
        "features": EARLY_FEATURES,
        "folds": folds,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in folds])),
        "mean_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in folds])
        ),
        "protection_audit": audit,
    }


def main() -> None:
    reference = pd.read_csv(CSV_ROOT / "Normal" / "1.csv")
    features = [feature for feature in EARLY_FEATURES if feature in reference]
    scale = build_scale(reference, features)
    reports = []
    for cutoff in [50.0, 100.0, 150.0]:
        reports.append(evaluate_cutoff(
            build_dataset(reference, scale, cutoff), cutoff
        ))

    report = {
        "task": "LOCA_vs_SLBIC",
        "model": "standardized logistic regression",
        "selection": "contiguous case folds",
        "reports": reports,
    }
    print(pd.DataFrame([
        {
            "cutoff_s": row["cutoff_s"],
            "mean_accuracy": row["mean_accuracy"],
            "mean_balanced_accuracy": row["mean_balanced_accuracy"],
        }
        for row in reports
    ]).to_string(index=False))
    output = RESULT_ROOT / "loca_slbic_early_validation.json"
    RESULT_ROOT.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
