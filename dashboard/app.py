"""Streamlit UI for the NPP-Guard v1 research prototype."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard.view_model import (
    build_export,
    export_json,
    inspect_input,
    invalid_response,
    prepare_view_model,
)
from src.explainability.core import _load_context, explain_result
from src.inference.v1 import diagnose_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "v1"
DEMO_FILES = {
    "accepted · FLB_10": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/FLB/10.csv",
    "accepted LOCA · LOCAC_19": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/LOCAC/19.csv",
    "requires_review · LOCAC_12": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/LOCAC/12.csv",
    "unknown · LR_93": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/LR/93.csv",
    "invalid_input · FLB_10 truncated to 50 s": PROJECT_ROOT / "data/NuclearPowerPlantAccidentData/Operation_csv_data/FLB/10.csv",
}


def _load_demo(label: str) -> pd.DataFrame:
    frame = pd.read_csv(DEMO_FILES[label])
    if label.startswith("invalid_input"):
        frame = frame.loc[pd.to_numeric(frame["TIME"], errors="coerce") <= 50].copy()
    return frame


def _context() -> dict:
    context = _load_context(PROJECT_ROOT, ARTIFACT_ROOT)
    path = PROJECT_ROOT / "results" / "18_global_feature_importance.csv"
    context["global_feature_rows"] = pd.read_csv(path).to_dict(orient="records") if path.exists() else []
    return context


def _status_message(st, status: str) -> None:
    messages = {
        "accepted": st.success,
        "requires_review": st.warning,
        "unknown": st.info,
        "invalid_input": st.error,
    }
    messages.get(status, st.info)(status.replace("_", " ").upper() + " · " + {
        "accepted": "research model accepted",
        "requires_review": "human review required",
        "unknown": "model has insufficient evidence",
        "invalid_input": "input rejected before prediction",
    }.get(status, "diagnostic state"))


def _metric_grid(st, values: list[tuple[str, object]]) -> None:
    columns = st.columns(min(4, len(values)))
    for column, (label, value) in zip(columns, values):
        column.metric(label, "—" if value is None else str(value))


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="NPP-Guard", page_icon="⚛", layout="wide")
    st.title("NPP-Guard")
    st.caption("Frozen v1 research diagnostic dashboard · model attribution, not physical causality")
    st.sidebar.header("Data input")
    source_mode = st.sidebar.radio("Input source", ["Demo sample", "Upload CSV"])
    source = ""
    frame: pd.DataFrame | None = None
    if source_mode == "Demo sample":
        choice = st.sidebar.selectbox("Example", list(DEMO_FILES))
        source = str(DEMO_FILES[choice].relative_to(PROJECT_ROOT))
        if DEMO_FILES[choice].exists():
            frame = _load_demo(choice)
        else:
            st.sidebar.error(f"Demo file not found: {source}")
    else:
        upload = st.sidebar.file_uploader("Upload one trajectory CSV", type=["csv"])
        if upload is not None:
            source = upload.name
            frame = pd.read_csv(upload)

    st.sidebar.divider()
    st.sidebar.caption("Policy parameters are frozen in artifacts/v1 and are not editable here.")
    st.sidebar.caption("Model: npp_guard_v1_17_2_20260921")
    if frame is None:
        st.info("Choose a demo sample or upload a CSV to begin.")
        return

    schema = json.loads((ARTIFACT_ROOT / "feature_extractor.json").read_text(encoding="utf-8"))
    quality = inspect_input(frame, schema["strict_process_features"], schema["minimum_window_s"])
    response = None
    explanation = None
    context = None
    if quality["valid"]:
        response = diagnose_dataframe(frame, ARTIFACT_ROOT)
        context = _context()
        explanation = explain_result(frame, response, context, Path(source).stem)
    else:
        response = invalid_response(quality, json.loads((ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8"))["model_version"])
    view = prepare_view_model(frame, response, explanation, quality)

    st.header("Data quality")
    _metric_grid(st, [
        ("TIME", "present" if quality["time_present"] else "missing"),
        ("Process variables", f"{quality['required_variable_count']}/{quality['required_variable_total']}"),
        ("Coverage", f"{quality['window_s']:.1f} s" if quality["window_s"] is not None else "—"),
        ("Sampling median", f"{quality['sampling_interval_median_s']:.1f} s" if quality["sampling_interval_median_s"] is not None else "—"),
    ])
    checks = pd.DataFrame([
        {"check": "TIME exists", "value": str(quality["time_present"])},
        {"check": "38 strict process variables complete", "value": str(quality["required_variable_count"] == quality["required_variable_total"])},
        {"check": "At least 120 s", "value": str(quality["coverage_ge_120s"])},
        {"check": "NaN/Inf absent", "value": str(not quality["nan_inf_columns"])},
        {"check": "No repeated/non-increasing time points", "value": str(not any("TIME_not" in error for error in quality["errors"]))},
        {"check": "Extra columns", "value": ", ".join(quality["extra_columns"]) if quality["extra_columns"] else "none"},
    ])
    st.dataframe(checks, hide_index=True, width="stretch")
    if quality["errors"]:
        st.error("Input quality errors: " + "; ".join(quality["errors"]))
        st.stop()

    st.header("Diagnostic status")
    _status_message(st, view["status"])
    diagnostic = view["diagnostic"]
    _metric_grid(st, [
        ("Accident family", diagnostic["accident_family"] or "not reported"),
        ("Family confidence", f"{diagnostic['family_confidence']:.4f}" if diagnostic["family_confidence"] is not None else "—"),
        ("Family tier", diagnostic["family_tier"] or "—"),
        ("Distance score", f"{diagnostic['distance_score']:.4f}" if diagnostic["distance_score"] is not None else "—"),
    ])
    st.json({
        "conformal_set": diagnostic["conformal_set"],
        "ood_warning": diagnostic["ood_warning"],
        "model_version": diagnostic["model_version"],
        "top1_candidate": diagnostic["top1_candidate"] if view["status"] == "unknown" else None,
    })

    st.header("120 s process trends")
    trend = view["trends"]
    if trend["variables"]:
        if trend["fallback"]:
            st.caption("Explanation unavailable; showing fallback core variables.")
        trend_frame = frame.loc[pd.to_numeric(frame["TIME"], errors="coerce") <= 120.5, ["TIME", *trend["variables"]]].copy().set_index("TIME")
        st.line_chart(trend_frame, width="stretch")
    else:
        st.info("No plottable process variables were found.")

    st.header("Explainability")
    if view["explanation"]["available"]:
        explanation_view = view["explanation"]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top variables")
            st.dataframe(pd.DataFrame(explanation_view["top_variables"]), hide_index=True, width="stretch")
            st.subheader("Top engineered features")
            st.dataframe(pd.DataFrame(explanation_view["top_features"]), hide_index=True, width="stretch")
        with col2:
            st.subheader("OOD drivers")
            st.dataframe(pd.DataFrame(explanation_view["ood_drivers"]), hide_index=True, width="stretch")
            if view["uncertainty"]:
                st.subheader("Uncertainty / conformal")
                st.json(view["uncertainty"])
        st.caption("All contributions are frozen-model attributions and diagnostic assistance, not physical causal effects.")
    else:
        st.info("No explanation was produced because the input was invalid.")

    st.header("LOCA assessment")
    if view["loca_assessment"]["run"]:
        st.metric("Exploratory severity estimate", f"{view['loca_assessment']['estimate']:.2f}%")
        st.dataframe(pd.DataFrame(view["explanation"]["severity_drivers"]), hide_index=True, width="stretch")
    else:
        st.info("Not run due to family gating.")
    st.caption("Protection time: not_available_in_v1 (exploratory research result is not chained into this API).")

    st.header("Export")
    payload = build_export(view, response, explanation, source)
    st.download_button("Export diagnosis JSON", export_json(payload), file_name="npp_guard_diagnosis.json", mime="application/json")

    st.divider()
    st.warning("Research prototype. Not for safety-critical deployment or real nuclear-plant operation/control. Family-level capability is affected by severity OOD; subtype OOD and conformal coverage guarantees remain unresolved.")


if __name__ == "__main__":
    main()
