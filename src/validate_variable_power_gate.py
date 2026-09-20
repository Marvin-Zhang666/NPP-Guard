"""Validate the condition-aware anomaly gate on variable-power scenarios."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from validate_npp_guard import (
    RULE_FEATURES,
    THRESHOLD,
    MIN_CONSECUTIVE,
    first_persistent_alarm,
    score_against_reference,
    read_mdb,
    composite,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
VARIABLE_ROOT = DATA_ROOT / "Variable_Power_Data"
RESULT_ROOT = PROJECT_ROOT / "results"
FAULTS = ["LOCA", "RW", "SGATR", "SGBTR"]


def matching_normal(start: float, end: float) -> str:
    return f"NORM_{int(start)}_to_{int(end)}"


def condition_scale(normal: pd.DataFrame) -> pd.Series:
    # The 80-to-100 normal file has only one sample before 100 s.
    # Use the complete normal transition as a fixed scenario template scale.
    scale = np.maximum(
        normal[RULE_FEATURES].std(),
        0.01 * normal[RULE_FEATURES].abs().median(),
    )
    scale[scale < 1e-6] = np.nan
    return scale


def evaluate_pair(
    normal_name: str,
    scenario_name: str,
    fault: str,
    injection_s: float | None,
) -> dict:
    normal = read_mdb(normal_name)
    scenario = read_mdb(scenario_name)
    scale = condition_scale(normal)
    scores = score_against_reference(
        normal, scenario, RULE_FEATURES, scale, limit=None
    )
    alarm_score = composite(scores, RULE_FEATURES)
    first = first_persistent_alarm(alarm_score)
    result = {
        "normal_reference": normal_name,
        "scenario": scenario_name,
        "fault": fault,
        "first_alarm_s": first,
        "peak_score": float(alarm_score.max()),
    }
    if injection_s is not None:
        result.update({
            "injection_s": injection_s,
            "detection_delay_s": (
                first - injection_s if first is not None else None
            ),
            "pre_injection_alarm": bool(
                first is not None and first < injection_s
            ),
        })
    return result


def main() -> None:
    manifest = pd.read_csv(VARIABLE_ROOT / "manifest.csv")
    rows = []
    for record in manifest.itertuples(index=False):
        if record.fault_code not in FAULTS:
            continue
        normal_name = matching_normal(
            record.start_power_pct, record.end_power_pct
        )
        rows.append(evaluate_pair(
            normal_name,
            record.scenario_id,
            record.fault_code,
            float(record.injection_time_s),
        ))

    # Self-comparison verifies that the normal condition template is quiet.
    for normal_name in ["NORM_80_to_100", "NORM_100_to_80"]:
        rows.append(evaluate_pair(
            normal_name, normal_name, "Normal", None
        ))

    report = {
        "stage": "condition_aware_anomaly_gate",
        "rule_features": RULE_FEATURES,
        "threshold": THRESHOLD,
        "min_consecutive": MIN_CONSECUTIVE,
        "scale_source": "complete matching normal transition template",
        "classification_stage": (
            "not transferred across power conditions; requires separate validation"
        ),
        "rows": rows,
    }
    frame = pd.DataFrame(rows)
    print(frame[
        ["scenario", "fault", "first_alarm_s", "peak_score",
         "injection_s", "detection_delay_s", "pre_injection_alarm"]
    ].to_string(index=False))
    output = RESULT_ROOT / "variable_power_gate_validation.json"
    RESULT_ROOT.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
