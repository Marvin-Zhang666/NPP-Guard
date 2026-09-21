"""Command-line dashboard validation using real v1/18 examples."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard.view_model import build_export, export_json, inspect_input, invalid_response, prepare_view_model
from src.explainability.core import _load_context, explain_result
from src.inference.v1 import diagnose_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "v1"
DEMO_FILES = {
    "accepted": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/FLB/10.csv",
    "accepted_loca": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/LOCAC/19.csv",
    "requires_review": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/LOCAC/12.csv",
    "unknown": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/LR/93.csv",
    "invalid_input": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/FLB/10.csv",
}
EXPECTED_STATUS = {"accepted": "accepted", "accepted_loca": "accepted", "requires_review": "requires_review", "unknown": "unknown", "invalid_input": "invalid_input"}


def _load_frames() -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_csv(path) for name, path in DEMO_FILES.items()}
    frames["invalid_input"] = frames["invalid_input"].loc[lambda frame: pd.to_numeric(frame["TIME"], errors="coerce") <= 50].copy()
    return frames


def run_smoke_tests(project_root: str | Path = PROJECT_ROOT) -> dict:
    root = Path(project_root)
    artifact_root = root / "artifacts" / "v1"
    schema = json.loads((artifact_root / "feature_extractor.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    context = _load_context(root, artifact_root)
    global_path = root / "results" / "18_global_feature_importance.csv"
    context["global_feature_rows"] = pd.read_csv(global_path).to_dict(orient="records")
    rows = []
    exports = {}
    for name, frame in _load_frames().items():
        quality = inspect_input(frame, schema["strict_process_features"], schema["minimum_window_s"])
        explanation = None
        if quality["valid"]:
            response = diagnose_dataframe(frame, artifact_root)
            explanation = explain_result(frame, response, context, name)
        else:
            response = invalid_response(quality, manifest["model_version"])
        view = prepare_view_model(frame, response, explanation, quality)
        payload = build_export(view, response, explanation, name)
        export_text = export_json(payload)
        actual = response["status"]
        severity_visible = bool(view["loca_assessment"]["run"])
        expected_severity = name == "accepted_loca"
        row = {
            "example": name,
            "expected_status": EXPECTED_STATUS[name],
            "actual_status": actual,
            "status_pass": actual == EXPECTED_STATUS[name],
            "explanation_available": bool(view["explanation"]["available"]),
            "severity_visible": severity_visible,
            "severity_gate_pass": severity_visible == expected_severity,
            "export_json_pass": bool(json.loads(export_text).get("schema_version") == "npp_guard_dashboard_export_v1" and "raw_data" not in export_text),
            "input_quality_pass": bool(quality["valid"] == (name != "invalid_input")),
        }
        row["passed"] = all(value for key, value in row.items() if key.endswith("_pass"))
        rows.append(row)
        exports[name] = payload
    output = root / "results"
    output.mkdir(exist_ok=True)
    smoke = pd.DataFrame(rows)
    smoke.to_csv(output / "19_dashboard_smoke_tests.csv", index=False)
    (output / "19_example_exports.json").write_text(json.dumps(exports, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    summary = {
        "milestone": "19 Dashboard",
        "status": "passed" if bool(smoke["passed"].all()) else "failed",
        "technology": "Streamlit",
        "model_version": manifest["model_version"],
        "examples": rows,
        "all_smoke_tests_passed": bool(smoke["passed"].all()),
        "export_excludes_raw_csv": all("raw_data" not in json.dumps(payload, default=str) for payload in exports.values()),
        "frozen_artifacts_only": True,
    }
    (output / "19_dashboard_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_smoke_tests()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_smoke_tests_passed"] else 1)
