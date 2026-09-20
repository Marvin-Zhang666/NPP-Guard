"""Validate a conservative three-class classifier with an Unknown outcome."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from validate_loca_slbic import (
    ACCIDENTS,
    EARLY_FEATURES,
    FOLDS,
    case_vector,
    protection_time,
)
from validate_npp_guard import build_scale


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData" / "Operation_csv_data"
RESULT_ROOT = PROJECT_ROOT / "results"
CLASSES = ["LOCA", "SLBIC", "FLB"]
CUTOFF_S = 100.0
CONFIDENCE_THRESHOLD = 0.60


def build_dataset(reference: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    rows = []
    for accident in CLASSES:
        paths = sorted(
            (CSV_ROOT / accident).glob("*.csv"),
            key=lambda path: int(path.stem),
        )
        for path in paths:
            case = int(path.stem)
            if case > 100:
                continue
            rows.append({
                "accident": accident,
                "case": case,
                "protection_s": protection_time(accident, case),
                "vector": case_vector(
                    reference, scale, accident, case, CUTOFF_S
                ),
            })
    return pd.DataFrame(rows)


def fit_model(x_train: np.ndarray, y_train: pd.Series):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(x_train, y_train)
    return model


def evaluate(dataset: pd.DataFrame) -> dict:
    vectors = np.vstack(dataset["vector"])
    fill = np.nanmedian(vectors, axis=0)
    vectors = np.where(np.isfinite(vectors), vectors, fill)
    fold_rows = []
    for low, high in FOLDS:
        test_mask = dataset["case"].between(low, high)
        train_mask = ~test_mask
        model = fit_model(
            vectors[train_mask], dataset.loc[train_mask, "accident"]
        )
        probabilities = model.predict_proba(vectors[test_mask])
        predicted = model.classes_[probabilities.argmax(axis=1)]
        confidence = probabilities.max(axis=1)
        accepted = confidence >= CONFIDENCE_THRESHOLD
        actual = dataset.loc[test_mask, "accident"].to_numpy()
        accepted_accuracy = (
            float(accuracy_score(actual[accepted], predicted[accepted]))
            if accepted.any()
            else None
        )
        fold_rows.append({
            "fold": f"{low}-{high}",
            "test_cases": int(len(actual)),
            "plain_accuracy": float(accuracy_score(actual, predicted)),
            "accepted_cases": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": accepted_accuracy,
            "accepted_errors": int(
                np.sum(accepted & (predicted != actual))
            ),
            "accepted_by_class": {
                label: int(np.sum(accepted & (actual == label)))
                for label in CLASSES
            },
            "rejected_by_class": {
                label: int(np.sum(~accepted & (actual == label)))
                for label in CLASSES
            },
        })

    known = dataset["protection_s"].notna()
    protection_audit = {
        accident: {
            "known_reports": int(
                (known & (dataset["accident"] == accident)).sum()
            ),
            "protection_at_or_before_cutoff": int(
                (
                    known
                    & (dataset["accident"] == accident)
                    & (dataset["protection_s"] <= CUTOFF_S)
                ).sum()
            ),
        }
        for accident in CLASSES
    }
    return {
        "cutoff_s": CUTOFF_S,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classes": CLASSES,
        "features": EARLY_FEATURES,
        "folds": fold_rows,
        "mean_plain_accuracy": float(
            np.mean([row["plain_accuracy"] for row in fold_rows])
        ),
        "mean_coverage": float(
            np.mean([row["coverage"] for row in fold_rows])
        ),
        "accepted_errors_total": int(
            sum(row["accepted_errors"] for row in fold_rows)
        ),
        "accepted_accuracy_when_available": float(
            sum(
                row["accepted_cases"] * (row["accepted_accuracy"] or 0.0)
                for row in fold_rows
            ) / sum(row["accepted_cases"] for row in fold_rows)
        ),
        "protection_audit": protection_audit,
    }


def main() -> None:
    reference = pd.read_csv(CSV_ROOT / "Normal" / "1.csv")
    scale = build_scale(reference, EARLY_FEATURES)
    report = {
        "task": "gated_LOCA_SLBIC_FLB_classifier",
        "unknown_policy": "max_probability_below_threshold",
        "validation": evaluate(build_dataset(reference, scale)),
    }
    validation = report["validation"]
    print(pd.DataFrame([
        {
            "plain_accuracy": validation["mean_plain_accuracy"],
            "coverage": validation["mean_coverage"],
            "accepted_accuracy": validation[
                "accepted_accuracy_when_available"
            ],
            "accepted_errors": validation["accepted_errors_total"],
        }
    ]).to_string(index=False))
    output = RESULT_ROOT / "gated_fault_classifier_validation.json"
    RESULT_ROOT.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
