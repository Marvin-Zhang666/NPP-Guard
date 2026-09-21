"""Small stdlib smoke tests for dashboard rendering data."""

import unittest

import pandas as pd

from dashboard.view_model import inspect_input, prepare_view_model


FEATURES = ["P", "WSTA"]


def response(status: str, family: str | None = "feedwater_line_break", tier: str = "Tier A") -> dict:
    return {
        "status": status,
        "accident_family": family if status != "unknown" else None,
        "family_top1": family,
        "family_confidence": 0.9 if status != "invalid_input" else None,
        "conformal_set": [family] if family and status != "unknown" else [],
        "ood_warning": {"flag": status == "requires_review"},
        "distance_score": 1.0,
        "assessment": {"family_capability": {"tier": tier}},
        "severity_estimate": 12.0 if status == "accepted" and family == "primary_coolant_boundary_break" else None,
        "severity_status": "available_exploratory" if status == "accepted" and family == "primary_coolant_boundary_break" else "not_run_due_to_family_gating",
        "model_version": "test",
    }


class DashboardViewModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({"TIME": [0, 10, 120], "P": [1, 2, 3], "WSTA": [4, 5, 6]})

    def test_four_statuses_prepare_without_browser(self) -> None:
        for status in ("accepted", "requires_review", "unknown", "invalid_input"):
            prepared = prepare_view_model(self.frame, response(status), None, {"valid": status != "invalid_input"})
            self.assertEqual(prepared["status"], status)
            self.assertIn("trends", prepared)

    def test_invalid_quality_detects_short_window(self) -> None:
        quality = inspect_input(self.frame.iloc[:2], FEATURES)
        self.assertFalse(quality["valid"])
        self.assertIn("window_shorter_than_120_s", quality["errors"])

    def test_loca_severity_requires_accepted_tier_a(self) -> None:
        accepted = prepare_view_model(self.frame, response("accepted", "primary_coolant_boundary_break"), None, {"valid": True})
        review = prepare_view_model(self.frame, response("requires_review", "primary_coolant_boundary_break"), None, {"valid": True})
        self.assertTrue(accepted["loca_assessment"]["run"])
        self.assertFalse(review["loca_assessment"]["run"])


if __name__ == "__main__":
    unittest.main()
