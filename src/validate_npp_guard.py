"""Reproducible short-term validation for the NPP-Guard prototype."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyodbc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuclearPowerPlantAccidentData"
CSV_ROOT = DATA_ROOT / "Operation_csv_data"
VARIABLE_ROOT = DATA_ROOT / "Variable_Power_Data"
RESULT_ROOT = PROJECT_ROOT / "results"

RULE_FEATURES = ["WECS", "WSTB", "SCMA", "TRB", "PRB", "LWRB", "PRBA"]
CLASSIFIER_FEATURES = [
    "P", "TSAT", "HUP", "LVCR", "WECS", "SCMA", "SCMB",
    "RRCB", "WRCB", "RRCO", "WRCA", "RRCA", "VOL", "WSTB",
    "TRB", "PRB",
]
THRESHOLD = 5.0
MIN_CONSECUTIVE = 3
EARLY_LIMIT = 2000.0


def read_mdb(scenario: str) -> pd.DataFrame:
    path = VARIABLE_ROOT / scenario / "case1.mdb"
    connection = pyodbc.connect(
        f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={path}"
    )
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM PlotData")
        columns = [column[0] for column in cursor.description]
        frame = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
    finally:
        connection.close()
    return frame.sort_values("TIME").drop_duplicates("TIME").reset_index(drop=True)


def build_scale(reference: pd.DataFrame, features: list[str]) -> pd.Series:
    baseline = reference.loc[reference["TIME"] <= 100, features]
    scale = np.maximum(baseline.std(), 0.01 * baseline.abs().median())
    scale[scale < 1e-6] = np.nan
    return scale


def score_against_reference(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    scale: pd.Series,
    limit: float | None = EARLY_LIMIT,
) -> pd.DataFrame:
    reference = reference.sort_values("TIME")
    test = test.sort_values("TIME")
    reference_time = reference["TIME"].to_numpy(dtype=float)
    test_time = test["TIME"].to_numpy(dtype=float)
    valid = (test_time >= reference_time.min()) & (
        test_time <= reference_time.max()
    )
    if limit is not None:
        valid &= test_time <= limit
    time = test_time[valid]
    scores = pd.DataFrame(index=time)
    for feature in features:
        reference_values = np.interp(
            time,
            reference_time,
            reference[feature].to_numpy(dtype=float),
        )
        test_values = test.loc[valid, feature].to_numpy(dtype=float)
        scores[feature] = (
            np.abs(test_values - reference_values) / float(scale[feature])
        )
    return scores


def first_persistent_alarm(score: pd.Series) -> float | None:
    score = score.dropna().sort_index()
    hit = score > THRESHOLD
    run_length = hit.groupby((~hit).cumsum()).cumcount() + 1
    alarm = hit & (run_length >= MIN_CONSECUTIVE)
    return float(alarm.index[alarm][0]) if alarm.any() else None


def composite(scores: pd.DataFrame, features: list[str]) -> pd.Series:
    return scores[features].median(axis=1)


def evaluate_pair(
    baseline_name: str,
    test_name: str,
    injection_time: float | None,
    reference: pd.DataFrame,
    scale: pd.Series,
) -> dict:
    baseline = read_mdb(baseline_name)
    test = read_mdb(test_name)
    scores = score_against_reference(baseline, test, RULE_FEATURES, scale, None)
    alarm_score = composite(scores, RULE_FEATURES)
    first = first_persistent_alarm(alarm_score)
    result = {
        "baseline": baseline_name,
        "test": test_name,
        "first_alarm_s": first,
        "peak_score": float(alarm_score.max()),
    }
    if injection_time is not None:
        pre = alarm_score.index < injection_time
        result.update(
            injection_s=injection_time,
            detection_delay_s=(
                first - injection_time if first is not None else None
            ),
            pre_injection_alarm=bool((alarm_score.loc[pre] > THRESHOLD).any()),
            max_pre_injection=float(alarm_score.loc[pre].max()),
        )
    return result


def scan_full_power(reference: pd.DataFrame, scale: pd.Series) -> list[dict]:
    rows = []
    for accident in ["LOCA", "LOCAC", "FLB", "SLBIC", "RW", "SGATR", "SGBTR"]:
        for path in sorted(
            (CSV_ROOT / accident).glob("*.csv"),
            key=lambda item: int(item.stem),
        ):
            test = pd.read_csv(path)
            scores = score_against_reference(
                reference, test, RULE_FEATURES, scale
            )
            alarm_score = composite(scores, RULE_FEATURES)
            first = first_persistent_alarm(alarm_score)
            rows.append(
                {
                    "accident": accident,
                    "case": int(path.stem),
                    "alarm": first is not None,
                    "first_alarm_s": first,
                    "peak_score": float(alarm_score.max()),
                }
            )
    return rows


def case_vector(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    scale: pd.Series,
) -> np.ndarray:
    scores = score_against_reference(
        reference, test, CLASSIFIER_FEATURES, scale
    )
    time = scores.index.to_numpy(dtype=float)
    values = []
    for feature in CLASSIFIER_FEATURES:
        series = scores[feature].to_numpy(dtype=float)
        early = series[time <= 300]
        values.extend(
            [
                np.nanmedian(series),
                np.nanmean(series),
                np.nanmax(series),
                np.nanmedian(early) if len(early) else np.nan,
            ]
        )
    return np.asarray(values, dtype=float)


def centroid_classifier(reference: pd.DataFrame, scale: pd.Series) -> dict:
    positive = {"LOCA", "LOCAC"}
    negative = {"FLB", "SLBIC", "RW", "SGATR", "SGBTR"}
    train = []
    test = []
    for accident in sorted(positive | negative):
        for path in sorted(
            (CSV_ROOT / accident).glob("*.csv"),
            key=lambda item: int(item.stem),
        ):
            case = int(path.stem)
            row = {
                "accident": accident,
                "label": "LOCA_family" if accident in positive else "Other",
                "case": case,
                "vector": case_vector(reference, pd.read_csv(path), scale),
            }
            (train if case <= 50 else test).append(row)

    train_values = np.vstack([row["vector"] for row in train])
    test_values = np.vstack([row["vector"] for row in test])
    fill = np.nanmedian(train_values, axis=0)
    train_values = np.where(np.isfinite(train_values), train_values, fill)
    test_values = np.where(np.isfinite(test_values), test_values, fill)
    center = train_values.mean(axis=0)
    spread = train_values.std(axis=0)
    spread[spread < 1e-9] = 1
    train_values = (train_values - center) / spread
    test_values = (test_values - center) / spread
    labels = ["LOCA_family", "Other"]
    centroids = {
        label: train_values[
            [row["label"] == label for row in train]
        ].mean(axis=0)
        for label in labels
    }

    result = []
    for values, row in zip(test_values, test):
        distances = {
            label: float(np.mean((values - centroids[label]) ** 2))
            for label in labels
        }
        predicted = min(distances, key=distances.get)
        result.append(
            {
                "accident": row["accident"],
                "case": row["case"],
                "actual": row["label"],
                "predicted": predicted,
                "correct": predicted == row["label"],
            }
        )
    return {"rows": result, "accuracy": float(np.mean([r["correct"] for r in result]))}


def main() -> None:
    reference = pd.read_csv(CSV_ROOT / "Normal" / "1.csv")
    scale = build_scale(reference, RULE_FEATURES)
    classifier_scale = build_scale(reference, CLASSIFIER_FEATURES)
    variable_power = [
        evaluate_pair(
            "NORM_80_to_100", "LOCA_80_to_100", 2298.5, reference, scale
        ),
        evaluate_pair(
            "NORM_100_to_80", "LOCA_100_to_80", 915.0, reference, scale
        ),
    ]
    full_power = scan_full_power(reference, scale)
    classifier = centroid_classifier(reference, classifier_scale)
    report = {
        "rule_features": RULE_FEATURES,
        "threshold": THRESHOLD,
        "min_consecutive": MIN_CONSECUTIVE,
        "variable_power_loca": variable_power,
        "full_power_cases": full_power,
        "centroid_classifier": classifier,
    }
    print("Variable-power LOCA:")
    print(pd.DataFrame(variable_power).to_string(index=False))
    print("\nFull-power rule summary:")
    full_frame = pd.DataFrame(full_power)
    print(
        full_frame.groupby("accident").agg(
            cases=("case", "count"),
            alarms=("alarm", "sum"),
            alarm_rate=("alarm", "mean"),
            median_first_alarm=("first_alarm_s", "median"),
        ).to_string()
    )
    print(f"\nCentroid classifier accuracy: {classifier['accuracy']:.4f}")
    payload = json.dumps(report, indent=2)
    output = RESULT_ROOT / "short_term_validation.json"
    try:
        RESULT_ROOT.mkdir(exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"\nSaved: {output}")
    except PermissionError:
        print(f"\nCould not write result file: {output}")


if __name__ == "__main__":
    main()
