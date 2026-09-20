"""Cross-direction validation for the available variable-power faults."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from validate_loca_slbic import EARLY_FEATURES
from validate_npp_guard import read_mdb, score_against_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
VARIABLE_ROOT = DATA_ROOT / "Variable_Power_Data"
RESULT_ROOT = PROJECT_ROOT / "results"
FAULTS = ["LOCA", "RW", "SGATR", "SGBTR"]
WINDOW_S = 500.0
CONFIDENCE_THRESHOLD = 0.60


def vector_for_fault(
    normal_name: str,
    scenario_name: str,
    injection_s: float,
) -> np.ndarray:
    normal = read_mdb(normal_name)
    scenario = read_mdb(scenario_name)
    scale = np.maximum(
        normal[EARLY_FEATURES].std(),
        0.01 * normal[EARLY_FEATURES].abs().median(),
    )
    scale[scale < 1e-6] = np.nan
    scores = score_against_reference(
        normal, scenario, EARLY_FEATURES, scale, limit=None
    )
    time = scores.index.to_numpy(dtype=float)
    scores = scores.loc[
        (time >= injection_s)
        & (time <= injection_s + WINDOW_S)
    ]
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


def build_dataset() -> pd.DataFrame:
    manifest = pd.read_csv(VARIABLE_ROOT / "manifest.csv")
    rows = []
    for record in manifest.itertuples(index=False):
        if record.fault_code not in FAULTS:
            continue
        direction = record.transition_direction
        normal_name = (
            f"NORM_{int(record.start_power_pct)}_to_"
            f"{int(record.end_power_pct)}"
        )
        rows.append({
            "direction": direction,
            "fault": record.fault_code,
            "scenario": record.scenario_id,
            "vector": vector_for_fault(
                normal_name,
                record.scenario_id,
                float(record.injection_time_s),
            ),
        })
    return pd.DataFrame(rows)


def evaluate(dataset: pd.DataFrame) -> dict:
    values = np.vstack(dataset["vector"])
    fill = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, fill)
    results = []
    for train_direction, test_direction in [
        ("increase", "decrease"),
        ("decrease", "increase"),
    ]:
        train = dataset["direction"] == train_direction
        test = dataset["direction"] == test_direction
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        )
        model.fit(values[train], dataset.loc[train, "fault"])
        probabilities = model.predict_proba(values[test])
        predicted = model.classes_[probabilities.argmax(axis=1)]
        confidence = probabilities.max(axis=1)
        actual = dataset.loc[test, "fault"].to_numpy()
        accepted = confidence >= CONFIDENCE_THRESHOLD
        results.append({
            "train_direction": train_direction,
            "test_direction": test_direction,
            "plain_accuracy": float(accuracy_score(actual, predicted)),
            "plain_balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ),
            "accepted_cases": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float(
                accuracy_score(actual[accepted], predicted[accepted])
            ),
            "accepted_errors": int(
                np.sum(accepted & (predicted != actual))
            ),
            "predictions": [
                {
                    "scenario": row.scenario,
                    "actual": row.fault,
                    "predicted": label,
                    "confidence": float(conf),
                    "accepted": bool(ok),
                }
                for row, label, conf, ok in zip(
                    dataset.loc[test].itertuples(),
                    predicted,
                    confidence,
                    accepted,
                )
            ],
        })
    return {
        "window_after_injection_s": WINDOW_S,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "faults": FAULTS,
        "direction_splits": results,
    }


def main() -> None:
    report = {
        "task": "variable_power_cross_direction_classifier",
        "validation": evaluate(build_dataset()),
        "limitation": (
            "No variable-power SLBIC scenario is present; this report does not validate "
            "LOCA versus SLBIC under changing power."
        ),
    }
    for row in report["validation"]["direction_splits"]:
        print({
            key: row[key]
            for key in [
                "train_direction", "test_direction", "plain_accuracy",
                "accepted_cases", "coverage", "accepted_accuracy",
                "accepted_errors",
            ]
        })
    output = RESULT_ROOT / "variable_power_classifier_validation.json"
    RESULT_ROOT.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
