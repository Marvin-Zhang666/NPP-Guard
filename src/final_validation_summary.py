"""Run the scoped NPP-Guard pipeline over all available validation cases."""

from pathlib import Path
import json

import pandas as pd

from npp_guard_inference import infer_full_power, infer_variable_power


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
CSV_ROOT = DATA_ROOT / "Operation_csv_data"
VARIABLE_ROOT = DATA_ROOT / "Variable_Power_Data"
RESULT_ROOT = PROJECT_ROOT / "results"
FULL_POWER_ACCIDENTS = [
    "LOCA", "LOCAC", "FLB", "SLBIC", "RW", "SGATR", "SGBTR"
]


def full_power_rows() -> list[dict]:
    rows = []
    for accident in FULL_POWER_ACCIDENTS:
        paths = sorted(
            (CSV_ROOT / accident).glob("*.csv"),
            key=lambda path: int(path.stem),
        )
        for path in paths:
            result = infer_full_power(path)
            rows.append({
                "accident": accident,
                "case": int(path.stem),
                **result,
            })
    return rows


def summarize_full_power(rows: list[dict]) -> dict:
    frame = pd.DataFrame([
        {
            "accident": row["accident"],
            "case": row["case"],
            "first_alarm_s": row["anomaly_gate"]["first_alarm_s"],
            "decision": row["decision"],
            "raw_prediction": row["classification"]["raw_prediction"],
            "confidence": row["classification"]["confidence"],
            "accepted": row["classification"]["accepted"],
        }
        for row in rows
    ])
    summary = {}
    for accident, group in frame.groupby("accident"):
        alarmed = group[group.first_alarm_s.notna()]
        summary[accident] = {
            "cases": int(len(group)),
            "gate_alarm_rate": float(group.first_alarm_s.notna().mean()),
            "median_first_alarm_s": (
                float(group.first_alarm_s.median())
                if group.first_alarm_s.notna().any() else None
            ),
            "accepted_classification_rate_when_alarm": float(
                alarmed.accepted.mean()
            ) if len(alarmed) else None,
            "decision_counts": {
                str(key): int(value)
                for key, value in group.decision.value_counts().items()
            },
        }
    target = frame[frame.accident.isin(["LOCA", "SLBIC", "FLB"])]
    confusion = {}
    for accident, group in target.groupby("accident"):
        confusion[accident] = {
            str(key): int(value)
            for key, value in group.decision.value_counts().items()
        }
    return {
        "by_accident": summary,
        "target_decision_counts": confusion,
    }


def variable_power_rows() -> list[dict]:
    manifest = pd.read_csv(VARIABLE_ROOT / "manifest.csv")
    rows = []
    for record in manifest.itertuples(index=False):
        if record.fault_code == "NA":
            result = infer_variable_power(
                record.scenario_id, record.scenario_id
            )
        else:
            normal_name = (
                f"NORM_{int(record.start_power_pct)}_to_"
                f"{int(record.end_power_pct)}"
            )
            result = infer_variable_power(
                normal_name, record.scenario_id
            )
        rows.append(result)
    return rows


def main() -> None:
    full_rows = full_power_rows()
    variable_rows = variable_power_rows()
    report = {
        "scope": {
            "full_power_classifier": "LOCA/SLBIC/FLB, 100 s window",
            "variable_power": "condition-aware anomaly gate only",
        },
        "full_power_summary": summarize_full_power(full_rows),
        "full_power_rows": full_rows,
        "variable_power_rows": variable_rows,
    }
    print(pd.DataFrame([
        {
            "accident": accident,
            **values,
        }
        for accident, values in report["full_power_summary"]["by_accident"].items()
    ]).to_string(index=False))
    print("\nVariable-power decisions:")
    print(pd.DataFrame([
        {
            "scenario": row.get("scenario"),
            "decision": row["decision"],
            "first_alarm_s": row["anomaly_gate"]["first_alarm_s"],
        }
        for row in variable_rows
    ]).to_string(index=False))
    output = RESULT_ROOT / "final_validation_summary.json"
    RESULT_ROOT.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
