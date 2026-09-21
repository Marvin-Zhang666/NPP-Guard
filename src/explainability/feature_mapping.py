"""Small, schema-checked mappings used by the explanation reports."""

from __future__ import annotations

from collections.abc import Iterable

from src.features import feature_group_table
from src.multi_accident_features import strict_process_features


SUMMARY_STATISTICS = ("last", "delta_t0", "mean", "std", "slope")
FORBIDDEN_GROUPS = {
    "direct_accident_or_leak",
    "protection_or_control_action",
    "radiological_or_dose",
    "safety_margin_or_consequence",
}


def split_feature(feature: str) -> tuple[str, str]:
    base, statistic = feature.split("__", 1)
    return base, statistic


def variable_for_feature(feature: str) -> str:
    return split_feature(feature)[0]


def assert_allowed_features(features: Iterable[str], *, severity: bool = False) -> None:
    features = list(features)
    strict = set(strict_process_features())
    table = feature_group_table()
    groups = dict(zip(table["feature"], table["group"]))
    for feature in features:
        base, statistic = split_feature(feature)
        if base not in strict or statistic not in SUMMARY_STATISTICS:
            raise AssertionError(f"Explanation feature is outside the frozen schema: {feature}")
        if groups[base] in FORBIDDEN_GROUPS:
            raise AssertionError(f"Explanation uses a forbidden variable group: {base}")
        if severity and base in {"LVPZ", "P", "TSAT", "VOL"}:
            raise AssertionError(f"Severity explanation uses excluded base variable: {base}")


def aggregate_variables(rows: list[dict], value_key: str = "contribution") -> list[dict]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        variable = str(row["variable"])
        item = grouped.setdefault(variable, {"variable": variable, "contribution": 0.0, "importance": 0.0})
        value = float(row.get(value_key, 0.0))
        item["contribution"] += value
        item["importance"] += abs(value)
    return sorted(grouped.values(), key=lambda item: (-item["importance"], item["variable"]))
