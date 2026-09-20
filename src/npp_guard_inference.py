"""Unified, scope-aware inference for the NPP-Guard prototype."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from validate_gated_fault_classifier import CUTOFF_S
from validate_loca_slbic import EARLY_FEATURES
from validate_npp_guard import (
    EARLY_LIMIT,
    RULE_FEATURES,
    THRESHOLD,
    build_scale,
    composite,
    first_persistent_alarm,
    read_mdb,
    score_against_reference,
)
from validate_variable_power_gate import condition_scale


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
CSV_ROOT = DATA_ROOT / "Operation_csv_data"
VARIABLE_ROOT = DATA_ROOT / "Variable_Power_Data"
MODEL_PATH = PROJECT_ROOT / "models" / "npp_guard_full_power_early.joblib"


def _summary_vector(scores: pd.DataFrame) -> np.ndarray:
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


def _classify_early(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    artifact: dict,
) -> dict:
    scores = score_against_reference(
        reference,
        test,
        EARLY_FEATURES,
        artifact["reference_scale"],
        limit=CUTOFF_S,
    )
    vector = _summary_vector(scores)
    vector = np.where(
        np.isfinite(vector), vector, artifact["fill_values"]
    )
    probabilities = artifact["model"].predict_proba(vector.reshape(1, -1))[0]
    index = int(np.argmax(probabilities))
    predicted = str(artifact["model"].classes_[index])
    confidence = float(probabilities[index])
    accepted = confidence >= artifact["confidence_threshold"]
    return {
        "predicted": predicted if accepted else "Unknown",
        "raw_prediction": predicted,
        "confidence": confidence,
        "accepted": bool(accepted),
        "cutoff_s": CUTOFF_S,
    }


def _gate(
    normal: pd.DataFrame,
    test: pd.DataFrame,
    scale: pd.Series,
    limit: float | None = None,
) -> dict:
    scores = score_against_reference(
        normal, test, RULE_FEATURES, scale, limit=limit
    )
    alarm_score = composite(scores, RULE_FEATURES)
    first = first_persistent_alarm(alarm_score)
    return {
        "first_alarm_s": first,
        "peak_score": float(alarm_score.max()),
        "threshold": THRESHOLD,
    }


def infer_full_power(csv_path: str | Path) -> dict:
    reference = pd.read_csv(CSV_ROOT / "Normal" / "1.csv")
    test = pd.read_csv(csv_path)
    artifact = joblib.load(MODEL_PATH)
    gate = _gate(
        reference,
        test,
        build_scale(reference, RULE_FEATURES),
        limit=EARLY_LIMIT,
    )
    classification = _classify_early(reference, test, artifact)
    if gate["first_alarm_s"] is None:
        decision = "NoAlarm"
    else:
        decision = classification["predicted"]
    return {
        "scope": "full_power",
        "anomaly_gate": gate,
        "classification": classification,
        "decision": decision,
    }


def infer_variable_power(
    normal_name: str,
    scenario_name: str,
) -> dict:
    normal = read_mdb(normal_name)
    test = read_mdb(scenario_name)
    gate = _gate(normal, test, condition_scale(normal))
    if gate["first_alarm_s"] is None:
        decision = "NoAlarm"
        classification = "NotApplicable"
    else:
        decision = "Unknown"
        classification = "NotValidatedAcrossPowerConditions"
    return {
        "scope": "variable_power",
        "normal_reference": normal_name,
        "scenario": scenario_name,
        "anomaly_gate": gate,
        "classification": classification,
        "decision": decision,
    }


def main() -> None:
    examples = {
        "full_power_loca": infer_full_power(
            CSV_ROOT / "LOCA" / "51.csv"
        ),
        "full_power_slbic": infer_full_power(
            CSV_ROOT / "SLBIC" / "51.csv"
        ),
        "variable_normal": infer_variable_power(
            "NORM_100_to_80", "NORM_100_to_80"
        ),
        "variable_loca": infer_variable_power(
            "NORM_100_to_80", "LOCA_100_to_80"
        ),
    }
    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
