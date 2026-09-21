"""Minimal example for the fixed-artifact NPP-Guard v1 API."""

from __future__ import annotations

import json
from pathlib import Path

from src.inference import diagnose_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData" / "Operation_csv_data" / "LOCA" / "1.csv"


if __name__ == "__main__":
    print(json.dumps(diagnose_csv(INPUT), ensure_ascii=False, indent=2))
