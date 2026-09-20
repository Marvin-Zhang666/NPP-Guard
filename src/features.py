"""Reusable ML-dataset audit helpers for the NPPAD LOCA study."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loader import DEFAULT_LOCA_DIR, DEFAULT_TRANSIENT_DIR, loca_paths
except ImportError:
    from data_loader import DEFAULT_LOCA_DIR, DEFAULT_TRANSIENT_DIR, loca_paths


FEATURE_GROUP_BASIS = {
    "time_index": "TIME is the sampling coordinate, not a plant sensor.",
    "direct_accident_or_leak": (
        "The README definition names a break, leak, tube leak, or integrated "
        "break quantity; treat it as a direct fault label or near-label."
    ),
    "protection_or_control_action": (
        "The README definition names ECCS, HPI, accumulator, charging, "
        "relief, spray, heater, turbine, letdown, or another control/action "
        "state; exclude from a before-actuation detector."
    ),
    "observable_process": (
        "The README definition is a conventional temperature, pressure, "
        "level, flow, power, reactivity, or inventory process measurement."
    ),
    "safety_margin_or_consequence": (
        "The README definition is a safety margin, containment response, "
        "fuel/clad damage, hydrogen, debris, oxidation, or severe consequence."
    ),
    "radiological_or_dose": (
        "The README definition is radiation, activity, release, leakage, "
        "or external dose."
    ),
    "unknown": "The local README does not provide enough evidence for a safe role assignment.",
}


_DEFINITIONS = [
    ("TIME", "Time (sec)", "time_index", "exclude"),
    ("P", "Press RCS (bar)", "observable_process", "candidate"),
    ("TAVG", "Temp RCS average (°C)", "observable_process", "candidate"),
    ("THA", "Temp Hot leg A (°C)", "observable_process", "candidate"),
    ("THB", "Temp Hot leg B (°C)", "observable_process", "candidate"),
    ("TCA", "Temp Cold leg A (°C)", "observable_process", "candidate"),
    ("TCB", "Temp Cold leg B (°C)", "observable_process", "candidate"),
    ("WRCA", "Flow Reactor coolant loop A (t/hr)", "observable_process", "candidate"),
    ("WRCB", "Flow Reactor coolant loop B (t/hr)", "observable_process", "candidate"),
    ("PSGA", "Pressure Steam generator A (bar)", "observable_process", "candidate"),
    ("PSGB", "Pressure Steam generator B (bar)", "observable_process", "candidate"),
    ("WFWA", "Flow SG A feedwater (t/hr)", "observable_process", "candidate"),
    ("WFWB", "Flow SG B feedwater (t/hr)", "observable_process", "candidate"),
    ("WSTA", "Flow SG A steam (t/hr)", "observable_process", "candidate"),
    ("WSTB", "Flow SG B steam (t/hr)", "observable_process", "candidate"),
    ("VOL", "Volume RCS liquid (M3)", "observable_process", "candidate"),
    ("LVPZ", "Level Pressurizer (%)", "observable_process", "candidate"),
    ("VOID", "Void of RCS (%)", "safety_margin_or_consequence", "review"),
    ("WLR", "Flow RCS leak (t/hr)", "direct_accident_or_leak", "exclude"),
    ("WUP", "Flow Przr PORV and safeties (t/hr)", "protection_or_control_action", "exclude"),
    ("HUP", "Spec Enthalpy Przr discharge (kJ/kg)", "protection_or_control_action", "exclude"),
    ("HLW", "Spec Enthalpy RCS leak (kJ/kg)", "direct_accident_or_leak", "exclude"),
    ("WHPI", "Flow HPI (t/hr)", "protection_or_control_action", "exclude"),
    ("WECS", "Flow Total ECCS (t/hr)", "protection_or_control_action", "exclude"),
    ("QMWT", "PowerTotal megawatt thermal (MW)", "observable_process", "candidate"),
    ("LSGA", "Level SG A wide range (M)", "observable_process", "candidate"),
    ("LSGB", "Level SG B wide range (M)", "observable_process", "candidate"),
    ("QMGA", "Power SG A heat removal (MW)", "observable_process", "candidate"),
    ("QMGB", "Power SG B heat removal (MW)", "observable_process", "candidate"),
    ("NSGA", "Level SG A narrow range (%)", "observable_process", "candidate"),
    ("NSGB", "Level SG B narrow range (%)", "observable_process", "candidate"),
    ("TBLD", "Power Turbine load (%)", "protection_or_control_action", "exclude"),
    ("WTRA", "Flow SG A tube leak (t/hr)", "direct_accident_or_leak", "exclude"),
    ("WTRB", "Flow SG B tube leak (t/hr)", "direct_accident_or_leak", "exclude"),
    ("TSAT", "Temp Przr saturation (°C)", "observable_process", "candidate"),
    ("QRHR", "Power RHR removal rate (MW)", "protection_or_control_action", "exclude"),
    ("LVCR", "Level Core water (M)", "observable_process", "candidate"),
    ("SCMA", "Temp Loop A subcooling margin (°C)", "safety_margin_or_consequence", "review"),
    ("SCMB", "Temp Loop B subcooling margin (°C)", "safety_margin_or_consequence", "review"),
    ("FRCL", "Fraction Clad failure (%)", "safety_margin_or_consequence", "exclude"),
    ("PRB", "Press Reactor building (bar)", "safety_margin_or_consequence", "review"),
    ("PRBA", "Press Partial RB air (bar)", "safety_margin_or_consequence", "review"),
    ("TRB", "Temp Reactor building (°C)", "safety_margin_or_consequence", "review"),
    ("LWRB", "Level RB sump water (M)", "safety_margin_or_consequence", "review"),
    ("DNBR", "Ratio Departure from nuclear boiling", "safety_margin_or_consequence", "review"),
    ("QFCL", "Power Fan cooler heat removal (MW)", "protection_or_control_action", "exclude"),
    ("WBK", "Flow Total break entering RB (t/hr)", "direct_accident_or_leak", "exclude"),
    ("WSPY", "Flow Pressurizer spray (t/hr)", "protection_or_control_action", "exclude"),
    ("WCSP", "Flow Containment spray (t/hr)", "protection_or_control_action", "exclude"),
    ("HTR", "Power Pressurizer heater (KW)", "protection_or_control_action", "exclude"),
    ("MH2", "Mass H2 generated by Zr-H2O (kg)", "safety_margin_or_consequence", "exclude"),
    ("CNH2", "Concentration RB hydrogen (%)", "safety_margin_or_consequence", "exclude"),
    ("RHBR", "Reactivity Soluble boron (%dk/k)", "observable_process", "candidate"),
    ("RHMT", "Reactivity Mod temp (%dk/k)", "observable_process", "candidate"),
    ("RHFL", "Reactivity Fuel (Doppler) (%dk/k)", "observable_process", "candidate"),
    ("RHRD", "Reactivity Rod (%dk/k)", "protection_or_control_action", "exclude"),
    ("RH", "Reactivity Total (%dk/k)", "observable_process", "candidate"),
    ("PWNT", "Power Nuclear Flux (%)", "observable_process", "candidate"),
    ("PWR", "Power Core thermal (%)", "observable_process", "candidate"),
    ("TFSB", "Temp Submerged fuel avg (°C)", "observable_process", "candidate"),
    ("TFPK", "Temp Peak fuel (°C)", "observable_process", "candidate"),
    ("TF", "Temp Average fuel (°C)", "observable_process", "candidate"),
    ("TPCT", "Temp Peak clad (°C)", "safety_margin_or_consequence", "review"),
    ("WCFT", "Flow Accumulator (t/hr)", "protection_or_control_action", "exclude"),
    ("WLPI", "Flow LPSI (RHR) (t/hr)", "protection_or_control_action", "exclude"),
    ("WCHG", "Flow Charging (t/hr)", "protection_or_control_action", "exclude"),
    ("RM1", "Rad Monitor RB air (CPM)", "radiological_or_dose", "exclude"),
    ("RM2", "Rad Monitor Steam Line (CPM)", "radiological_or_dose", "exclude"),
    ("RM3", "Rad Monitor Condenser Off-gas (CPM)", "radiological_or_dose", "exclude"),
    ("RM4", "Rad Monitor Aux Building Air (CPM)", "radiological_or_dose", "exclude"),
    ("RC87", "Activity RC Coolant (CPM)", "radiological_or_dose", "exclude"),
    ("RC131", "Concentration RC I-131 Eq (GBq/cc)", "radiological_or_dose", "exclude"),
    ("STRB", "Rad Rel Rate RB (GBq/s)", "radiological_or_dose", "exclude"),
    ("STSG", "Rad Rel Rate SG Valves (GBq/s)", "radiological_or_dose", "exclude"),
    ("STTB", "Rad Rel Rate Cdsr Off-gas (GBq/s)", "radiological_or_dose", "exclude"),
    ("RBLK", "Mass Total Leakage out of RB (kg)", "radiological_or_dose", "exclude"),
    ("SGLK", "Mass Total Leakage out of SGs (kg)", "radiological_or_dose", "exclude"),
    ("DTHY", "Dose Rate EAB Thyroid (mSv/hr)", "radiological_or_dose", "exclude"),
    ("DWB", "Dose Rate EAB Whole Body (mSv/hr)", "radiological_or_dose", "exclude"),
    ("WRLA", "Flow SG A MSV/ADV (t/h)", "protection_or_control_action", "exclude"),
    ("WRLB", "Flow SG B MSV/ADV (t/h)", "protection_or_control_action", "exclude"),
    ("WLD", "Flow Letdown (t/hr)", "protection_or_control_action", "exclude"),
    ("MBK", "Integrated Break Flow (kg)", "direct_accident_or_leak", "exclude"),
    ("EBK", "Integrated Break Energy (MJ)", "direct_accident_or_leak", "exclude"),
    ("TKLV", "Volume RWST Water (M3)", "protection_or_control_action", "exclude"),
    ("FRZR", "Fraction Zr Oxidation (%)", "safety_margin_or_consequence", "exclude"),
    ("TDBR", "Temp of Debris in Cavity (°C)", "safety_margin_or_consequence", "exclude"),
    ("MDBR", "Mass of Corium in DW (Kg)", "safety_margin_or_consequence", "exclude"),
    ("MCRT", "Mass of molten concrete (Kg)", "safety_margin_or_consequence", "exclude"),
    ("MGAS", "Mass of CCI gases (Kg)", "safety_margin_or_consequence", "exclude"),
    ("TCRT", "Temp of Molten Concrete (°C)", "safety_margin_or_consequence", "exclude"),
    ("TSLP", "Temp of Debris in Lower Plenum (°C)", "safety_margin_or_consequence", "exclude"),
    ("PPM", "Concentration RCS Boron (ppm)", "observable_process", "candidate"),
    ("RRCA", "Ratio Loop A Flow", "observable_process", "candidate"),
    ("RRCB", "Ratio Loop B Flow", "observable_process", "candidate"),
    ("RRCO", "Ratio Core Flow", "observable_process", "candidate"),
    ("WFLB", "Flow FW Line Break (kg/s)", "direct_accident_or_leak", "exclude"),
]


def feature_group_table(columns: list[str] | None = None) -> pd.DataFrame:
    """Return the README-backed role table for the 97 operation columns."""
    table = pd.DataFrame(
        _DEFINITIONS,
        columns=["feature", "readme_definition", "group", "ml_policy"],
    )
    table["basis"] = table["group"].map(FEATURE_GROUP_BASIS)
    table["source"] = "NPPAD README.md, Table 2 (local copy)"
    if columns is not None:
        missing = sorted(set(columns) - set(table["feature"]))
        if missing:
            raise ValueError(f"Columns missing from local feature dictionary: {missing}")
        table = table.set_index("feature").loc[columns].reset_index()
    return table


_EVENT_PATTERNS = {
    "injection_s": r"Malfunction #\s*1 Fraction",
    "hpi_pump_1_s": r"HPI Pump #1 Position Change: 100%",
    "hpi_pump_2_s": r"HPI Pump #2 Position Change: 100%",
    "hpsi_start_s": r"HPSI start",
    "scram_s": r"Reactor Scram",
    "fw_isolation_s": r"FW isolation",
    "ctmt_spray_s": r"Ctmt Spray Starts",
    "accumulator_1_s": r"Accumulator Valve #1 Position Change: 100%",
    "sump_recirc_s": r"Sump Recirc",
    "core_uncovered_s": r"Core Uncovered",
}
_EVENT_LINE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+sec,\s*(.*)$")


def parse_transient_report(report_path: str | Path) -> dict[str, float | None]:
    """Extract the first timestamp for each named event in one report."""
    path = Path(report_path)
    found = {name: None for name in _EVENT_PATTERNS}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _EVENT_LINE.match(line)
            if not match:
                continue
            timestamp = float(match.group(1))
            description = match.group(2)
            for name, pattern in _EVENT_PATTERNS.items():
                if found[name] is None and re.search(pattern, description):
                    found[name] = timestamp
    protection_times = [
        found["hpi_pump_1_s"],
        found["scram_s"],
    ]
    protection_times = [value for value in protection_times if value is not None]
    found["first_protection_s"] = min(protection_times) if protection_times else None
    return found


def build_loca_event_table(
    loca_dir: str | Path = DEFAULT_LOCA_DIR,
    transient_dir: str | Path = DEFAULT_TRANSIENT_DIR,
) -> pd.DataFrame:
    """Build one reusable event row per LOCA trajectory."""
    rows = []
    for path in loca_paths(loca_dir):
        number = int(path.stem)
        report = Path(transient_dir) / f"{number}Transient Report.txt"
        row = {
            "sample_id": f"LOCA_{number:03d}",
            "severity": number,
            "source_file": path.name,
            "report_file": report.name,
            "report_exists": report.exists(),
        }
        row.update(parse_transient_report(report))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("severity").reset_index(drop=True)


def label_time_windows(
    times: pd.Series | np.ndarray,
    injection_s: float,
    first_protection_s: float,
) -> pd.Series:
    """Label time points without treating pre-injection samples as LOCA positives."""
    values = pd.to_numeric(pd.Series(times), errors="coerce")
    labels = np.full(len(values), "unknown", dtype=object)
    labels[values.le(injection_s).to_numpy()] = "normal_or_pre_fault"
    labels[
        values.gt(injection_s).to_numpy()
        & values.lt(first_protection_s).to_numpy()
    ] = "early_loca"
    labels[values.ge(first_protection_s).to_numpy()] = "post_protection_loca"
    return pd.Series(labels, index=getattr(times, "index", None), name="window")


def interleaved_group_split(groups: list[str] | pd.Series) -> pd.DataFrame:
    """Create a deterministic 80/10/10 group-only split template."""
    unique_groups = sorted(
        {str(group) for group in groups},
        key=lambda value: (
            int(match.group(1)) if (match := re.search(r"(\d+)$", value)) else 10**9,
            value,
        ),
    )
    rows = []
    for rank, group in enumerate(unique_groups):
        phase = rank % 10
        split = "test" if phase == 0 else "validation" if phase == 1 else "train"
        rows.append({"group_id": group, "split": split})
    return pd.DataFrame(rows)


def candidate_pre_protection_features() -> list[str]:
    """Return the strict process-only candidate set for the first baseline."""
    return [
        feature
        for feature, _, _, policy in _DEFINITIONS
        if policy == "candidate"
    ]


def excluded_pre_protection_features() -> list[str]:
    """Return all columns not admitted to the strict process-only baseline."""
    return [
        feature
        for feature, _, _, policy in _DEFINITIONS
        if policy != "candidate"
    ]


if __name__ == "__main__":
    table = feature_group_table()
    assert len(table) == 97
    assert table["feature"].is_unique
    assert set(table["group"]) <= set(FEATURE_GROUP_BASIS)
    print(table.groupby(["group", "ml_policy"]).size().to_string())
