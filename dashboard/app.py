"""Streamlit UI for the NPP-Guard v1 research prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.view_model import (
    build_export,
    export_json,
    inspect_input,
    invalid_response,
    prepare_view_model,
)
from src.explainability.core import _load_context, explain_result
from src.inference.v1 import diagnose_dataframe


ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "v1"
SEVERITY_REFERENCE_PATH = PROJECT_ROOT / "dashboard" / "fixtures" / "loca_severity_reference.json"


def _load_schema() -> dict:
    return json.loads((ARTIFACT_ROOT / "feature_extractor.json").read_text(encoding="utf-8"))


def _template_csv(features: list[str]) -> bytes:
    frame = pd.DataFrame({"TIME": list(range(0, 121, 10))})
    for feature in features:
        frame[feature] = 0.0
    return frame.to_csv(index=False).encode("utf-8")


def _context() -> dict:
    context = _load_context(PROJECT_ROOT, ARTIFACT_ROOT)
    path = PROJECT_ROOT / "results" / "18_global_feature_importance.csv"
    context["global_feature_rows"] = pd.read_csv(path).to_dict(orient="records") if path.exists() else []
    reference = json.loads(SEVERITY_REFERENCE_PATH.read_text(encoding="utf-8"))
    model_version = context["artifacts"]["manifest"]["model_version"]
    if reference.get("model_version") != model_version:
        raise AssertionError("Web severity reference does not match the frozen model version")
    columns = list(context["artifacts"]["loca_severity"]["columns"])
    values = reference.get("values")
    if not isinstance(values, list) or len(values) != len(columns):
        raise AssertionError("Web severity reference does not match the frozen severity schema")
    context["severity_reference"] = pd.Series(
        [float(value) for value in values], index=columns, dtype=float
    )
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
    with st.container(horizontal=True):
        for label, value in values:
            st.metric(label, "—" if value is None else str(value), border=True)


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="NPP-Guard | Nuclear Accident Diagnosis Research Prototype",
        page_icon="⚛",
        layout="wide",
    )
    st.title("NPP-Guard")
    st.caption("Nuclear accident diagnosis research prototype · frozen v1.2 inference")
    st.warning("Research prototype. Not for safety-critical deployment or real nuclear-plant operation/control.")
    st.info("Explainability ≠ causality. Results are diagnostic assistance for simulation research, not operating instructions.")

    schema = _load_schema()
    features = list(schema["strict_process_features"])
    with st.expander("使用说明 / CSV example format", expanded=True):
        st.markdown(
            "Upload one trajectory CSV with `TIME` plus all 38 strict process variables. "
            "The file must cover at least 120 s, use approximately 10 s sampling, keep time strictly increasing, "
            "and contain no NaN/Inf values. Only the frozen 120 s inference window is used."
        )
        st.code("TIME," + ",".join(features), language="text")
        st.download_button(
            "Download synthetic CSV template",
            _template_csv(features),
            file_name="npp_guard_web_template.csv",
            mime="text/csv",
            help="13 rows from 0 to 120 s with zero-valued synthetic process columns; replace them with a permitted trajectory.",
        )

    st.sidebar.header("CSV input")
    st.sidebar.caption("Web mode is upload-only; raw NPPAD trajectories are not bundled or uploaded by the app.")
    upload = st.sidebar.file_uploader("Upload one trajectory CSV", type=["csv"])
    source = ""
    frame: pd.DataFrame | None = None
    if upload is not None:
        source = upload.name
        try:
            frame = pd.read_csv(upload)
        except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
            st.sidebar.error(f"CSV could not be parsed: {type(exc).__name__}")

    st.sidebar.divider()
    st.sidebar.caption("Policy parameters are frozen in artifacts/v1 and are not editable here.")
    st.sidebar.caption("Model: npp_guard_v1_17_2_20260921")
    if frame is None:
        st.info("Upload a trajectory CSV to begin. The downloadable template is synthetic and contains no NPPAD data.")
        return

    quality = inspect_input(frame, features, schema["minimum_window_s"])
    response = None
    explanation = None
    context = None
    if quality["valid"]:
        try:
            response = diagnose_dataframe(frame, ARTIFACT_ROOT)
            context = _context()
            explanation = explain_result(frame, response, context, Path(source).stem)
        except (AssertionError, OSError, ValueError) as exc:
            st.error(f"Frozen web assets could not be loaded: {type(exc).__name__}")
            return
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
        severity_drivers = view["explanation"].get("severity_drivers")
        if severity_drivers:
            st.dataframe(pd.DataFrame(severity_drivers), hide_index=True, width="stretch")
        else:
            st.info("The frozen severity estimate is available; detailed attribution was not produced for this input.")
    else:
        st.info("Not run due to family gating.")
    st.caption("Protection time: not_available_in_v1 (exploratory research result is not chained into this API).")

    st.header("Export")
    payload = build_export(view, response, explanation, source)
    st.download_button("Export diagnosis JSON", export_json(payload), file_name="npp_guard_diagnosis.json", mime="application/json")

    st.divider()
    st.warning("Research prototype. Not for safety-critical deployment. Family-level capability is affected by severity OOD; subtype OOD and conformal coverage guarantees remain unresolved.")


if __name__ == "__main__":
    main()
