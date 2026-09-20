"""Train and save the validated full-power early classifier prototype."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from validate_gated_fault_classifier import (
    CLASSES,
    CUTOFF_S,
    CONFIDENCE_THRESHOLD,
    fit_model,
)
from validate_loca_slbic import EARLY_FEATURES, case_vector
from validate_npp_guard import build_scale


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData" / "Operation_csv_data"
MODEL_ROOT = PROJECT_ROOT / "models"


def main() -> None:
    reference = pd.read_csv(CSV_ROOT / "Normal" / "1.csv")
    scale = build_scale(reference, EARLY_FEATURES)
    rows = []
    for accident in CLASSES:
        for case in range(1, 101):
            path = CSV_ROOT / accident / f"{case}.csv"
            if path.exists():
                rows.append({
                    "label": accident,
                    "vector": case_vector(
                        reference, scale, accident, case, CUTOFF_S
                    ),
                })

    values = np.vstack([row["vector"] for row in rows])
    fill = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, fill)
    labels = pd.Series([row["label"] for row in rows])
    model = fit_model(values, labels)

    artifact = {
        "model": model,
        "feature_names": [
            f"{feature}_{stat}"
            for feature in EARLY_FEATURES
            for stat in ["median", "mean", "max", "last"]
        ],
        "fill_values": fill,
        "reference_scale": scale,
        "cutoff_s": CUTOFF_S,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classes": CLASSES,
        "scope": "full_power_only",
        "unknown_policy": "max_probability_below_threshold",
        "variable_power_policy": "do_not_use_without_condition_validation",
        "training_cases_per_class": 100,
    }
    MODEL_ROOT.mkdir(exist_ok=True)
    output = MODEL_ROOT / "npp_guard_full_power_early.joblib"
    joblib.dump(artifact, output)
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"model", "fill_values", "reference_scale"}
    }
    (MODEL_ROOT / "npp_guard_full_power_early.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved: {output}")
    print(f"Saved: {MODEL_ROOT / 'npp_guard_full_power_early.json'}")


if __name__ == "__main__":
    main()
