"""Generic Transient Report event taxonomy for milestone 13.

This module is additive: the milestone-09 parser and its historical CSV stay
unchanged.  The taxonomy deliberately excludes malfunction/fraction lines from
protection matches because those lines describe accident injection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


EVENT_TAXONOMY = (
    {
        "category": "reactor_scram",
        "protection_relevance": "automatic_reactor_protection",
        "regex": r"\bReactor\s+Scram\b",
        "description": "Reactor Scram",
    },
    {
        "category": "safety_relief_valve_open",
        "protection_relevance": "automatic_pressure_protection",
        "regex": r"\bSafety\s+Relief\s+Valve\b.*\bPosition\s+Change:\s*100\s*%",
        "description": "Safety Relief Valve opening",
    },
    {
        "category": "hpi_pump_start",
        "protection_relevance": "automatic_safety_injection",
        "regex": r"\bHPI\s+Pump\b.*\bPosition\s+Change:\s*100\s*%",
        "description": "HPI pump start",
    },
    {
        "category": "hpsi_start",
        "protection_relevance": "automatic_safety_injection",
        "regex": r"\bHPSI\b.*\bstart\b",
        "description": "HPSI start",
    },
    {
        "category": "safety_injection",
        "protection_relevance": "automatic_safety_injection",
        "regex": r"\bSafety\s+Injection\b",
        "description": "Safety Injection",
    },
    {
        "category": "spray_start",
        "protection_relevance": "automatic_pressure_or_containment_protection",
        "regex": r"\b(?:Containment|Pressurizer)?\s*Spray\b.*(?:\bstart\b|\bPosition\s+Change:\s*100\s*%)",
        "description": "Spray start",
    },
    {
        "category": "feedwater_isolation",
        "protection_relevance": "automatic_isolation",
        "regex": r"\bFeed\s*water\b.*(?:\bisolat\w*\b|\bPosition\s+Change:\s*0\s*%)",
        "description": "Feedwater isolation",
    },
    {
        "category": "steam_isolation",
        "protection_relevance": "automatic_isolation",
        "regex": r"\b(?:Main\s+)?Steam(?:\s+Line)?\b.*(?:\bisolat\w*\b|\bPosition\s+Change:\s*0\s*%)",
        "description": "Steam isolation",
    },
    {
        "category": "turbine_trip",
        "protection_relevance": "automatic_turbine_protection",
        "regex": r"\bTurbine\s+Trip\b|\bTurbine\b.*\bPosition\s+Change:\s*0\s*%",
        "description": "Turbine trip",
    },
    {
        "category": "pump_trip",
        "protection_relevance": "automatic_component_protection",
        "regex": r"\b(?:HPI|LPSI|RHR|Charging|Pump)\b.*(?:\btrip\b|\bPosition\s+Change:\s*0\s*%)",
        "description": "Pump trip",
    },
)

_EVENT_LINE = re.compile(r"^\s*(?P<time>\d+(?:\.\d+)?)\s+sec,\s*(?P<description>.+?)\s*$")
_INJECTION = re.compile(r"\bMalfunction\s*#\s*\d+\s+Fraction\b", flags=re.IGNORECASE)


def _empty_result(path: Path) -> dict[str, Any]:
    return {
        "report_exists": path.exists(),
        "report_line_count": 0,
        "injection_s": None,
        "injection_event": None,
        "injection_parse_ok": False,
        "first_protection_s": None,
        "first_protection_event": None,
        "first_protection_category": None,
        "protection_match_count": 0,
        "protection_categories": "",
        "protection_match_examples": "",
        "event_parse_ok": False,
        "parser_version": "13_generic_taxonomy_v1",
    }


def parse_transient_report(report_path: str | Path) -> dict[str, Any]:
    """Parse injection separately and return the earliest non-injection safety event."""
    path = Path(report_path)
    result = _empty_result(path)
    if not path.exists():
        return result

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    result["report_line_count"] = len(lines)
    protection_matches: list[dict[str, Any]] = []
    category_seen: list[str] = []
    examples: list[str] = []
    for line in lines:
        match = _EVENT_LINE.match(line)
        if not match:
            continue
        timestamp = float(match.group("time"))
        description = match.group("description").strip()
        if _INJECTION.search(description):
            if result["injection_s"] is None:
                result["injection_s"] = timestamp
                result["injection_event"] = description
                result["injection_parse_ok"] = True
            continue
        for taxonomy in EVENT_TAXONOMY:
            if re.search(taxonomy["regex"], description, flags=re.IGNORECASE):
                category = taxonomy["category"]
                category_seen.append(category)
                protection_matches.append(
                    {
                        "time_s": timestamp,
                        "description": description,
                        "category": category,
                    }
                )
                if len(examples) < 8:
                    examples.append(f"{timestamp:g} sec, {description}")

    if protection_matches:
        first = min(protection_matches, key=lambda item: (item["time_s"], category_seen.index(item["category"])))
        result["first_protection_s"] = first["time_s"]
        result["first_protection_event"] = first["description"]
        result["first_protection_category"] = first["category"]
        result["event_parse_ok"] = True
    result["protection_match_count"] = len(protection_matches)
    result["protection_categories"] = ";".join(dict.fromkeys(category_seen))
    result["protection_match_examples"] = " || ".join(examples)
    return result


def scan_event_inventory(
    project_root: str | Path,
    inventory_path: str | Path,
    classes: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse selected trajectory reports without modifying historical event tables."""
    project_root = Path(project_root)
    inventory = pd.read_csv(inventory_path)
    selected = inventory.loc[inventory["accident_class"].isin(classes)].copy()
    report_root = project_root / "data" / "NuclearPowerPlantAccidentData" / "NPPAD"
    rows: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {item["category"]: 0 for item in EVENT_TAXONOMY}
    category_examples: dict[str, list[str]] = {item["category"]: [] for item in EVENT_TAXONOMY}
    for record in selected.sort_values("sample_id").itertuples(index=False):
        report_path = report_root / record.accident_class / f"{Path(record.operation_csv).stem}Transient Report.txt"
        parsed = parse_transient_report(report_path)
        for category in str(parsed["protection_categories"]).split(";"):
            if not category:
                continue
            category_counts[category] += 1
            if len(category_examples[category]) < 3:
                category_examples[category].append(
                    f"{record.sample_id}: {parsed['protection_match_examples']}"
                )
        rows.append(
            {
                "sample_id": record.sample_id,
                "accident_class": record.accident_class,
                "numeric_case_id": int(record.numeric_case_id),
                "operation_csv": record.operation_csv,
                "transient_report": report_path.relative_to(project_root).as_posix(),
                **parsed,
            }
        )
    frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    summary = {
        "parser_version": "13_generic_taxonomy_v1",
        "taxonomy": [dict(item) for item in EVENT_TAXONOMY],
        "trajectory_total": int(len(frame)),
        "injection_parsed": int(frame["injection_parse_ok"].sum()),
        "protection_parsed": int(frame["event_parse_ok"].sum()),
        "class_summary": {
            accident_class: {
                "trajectory_count": int((frame["accident_class"] == accident_class).sum()),
                "protection_parsed": int(
                    ((frame["accident_class"] == accident_class) & frame["event_parse_ok"]).sum()
                ),
                "first_protection_categories": sorted(
                    frame.loc[
                        (frame["accident_class"] == accident_class) & frame["event_parse_ok"],
                        "first_protection_category",
                    ].dropna().unique().tolist()
                ),
            }
            for accident_class in classes
        },
        "category_report_coverage": category_counts,
        "category_examples": category_examples,
        "limitations": [
            "Malfunction/Fraction lines are treated as accident injection and never as protection.",
            "Unknown reports remain unknown; no protection time is imputed.",
            "Safety Relief Valve opening is retained as an automatic pressure-protection event when explicitly logged.",
        ],
    }
    return frame, summary


def write_event_outputs(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    Path(json_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
