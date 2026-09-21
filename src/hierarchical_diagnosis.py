"""Official-name-based family mapping and hierarchical accident diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

try:
    from .multi_accident_features import strict_process_features
    from .multi_accident_models import model_suite
    from .ood_validation import CLASS_LABELS
    from .severity_invariant import MODEL_NAMES, WINDOWS_S, feature_columns
except ImportError:
    from multi_accident_features import strict_process_features
    from multi_accident_models import model_suite
    from ood_validation import CLASS_LABELS
    from severity_invariant import MODEL_NAMES, WINDOWS_S, feature_columns


SOURCE = "data/NuclearPowerPlantAccidentData/README.md Table 1 (local copy)"

OFFICIAL_ACCIDENTS = {
    "ATWS": ("Anticipated Transient Without Scram", "reactivity_and_control", "Name explicitly identifies a reactivity/control transient.", "high"),
    "FLB": ("Feedwater Line Break", "feedwater_line_break", "Official name identifies a feedwater-line boundary break.", "high"),
    "LACP": ("Loss of AC Power", "electrical_power_loss", "Official name identifies loss of alternating-current power.", "high"),
    "LLB": ("Letdown Line Break in auxiliary buildings", "auxiliary_letdown_break", "Official name identifies an auxiliary-building letdown-line break.", "high"),
    "LOCA": ("Loss of Coolant Accident (Hot Leg)", "primary_coolant_boundary_break", "Hot-leg LOCA and cold-leg LOCA share the primary-coolant boundary-break mechanism.", "high"),
    "LOCAC": ("Loss of Coolant Accident (Cold Leg)", "primary_coolant_boundary_break", "Hot-leg LOCA and cold-leg LOCA share the primary-coolant boundary-break mechanism.", "high"),
    "LOF": ("Loss of Flow (Locked Rotor)", "reactor_coolant_flow_loss", "Official name identifies a locked-rotor loss-of-flow event.", "high"),
    "LR": ("Load Rejection", "turbine_and_load_transient", "Load rejection and turbine trip are grouped by the official turbine/load transient descriptions.", "medium"),
    "MD": ("Moderator Dilution", "reactivity_and_control", "Official name identifies a moderator/reactivity event; grouped with other reactivity/control transients.", "medium"),
    "RI": ("Rod Insertion", "reactivity_and_control", "Rod insertion and rod withdrawal are grouped by the official control-rod descriptions.", "high"),
    "RW": ("Rod Withdrawal", "reactivity_and_control", "Rod insertion and rod withdrawal are grouped by the official control-rod descriptions.", "high"),
    "SGBTR": ("Steam Generator B Tube Rupture", "steam_generator_tube_rupture", "Steam-generator A/B tube ruptures share the official tube-rupture mechanism.", "high"),
    "SGATR": ("Steam Generator A Tube Rupture", "steam_generator_tube_rupture", "Steam-generator A/B tube ruptures share the official tube-rupture mechanism.", "high"),
    "SLBIC": ("Steam Line Break Inside Containment", "steam_line_break", "Inside/outside-containment steam-line breaks share the official steam-line-break mechanism.", "high"),
    "SLBOC": ("Steam Line Break Outside Containment", "steam_line_break", "Inside/outside-containment steam-line breaks share the official steam-line-break mechanism.", "high"),
    "SP": ("Spark Presence for Hydrogen Burn", "hydrogen_combustion", "Official name identifies a hydrogen-burn spark event.", "high"),
    "TT": ("Turbine Trip", "turbine_and_load_transient", "Load rejection and turbine trip are grouped by the official turbine/load transient descriptions.", "medium"),
}


def family_mapping(classes: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    names = list(classes) if classes is not None else sorted(OFFICIAL_ACCIDENTS)
    rows = []
    for accident_class in names:
        if accident_class not in OFFICIAL_ACCIDENTS:
            rows.append(
                {
                    "class": accident_class,
                    "official_description": "",
                    "family": "unknown_or_other",
                    "source": SOURCE,
                    "justification": "No local official definition was found in the Table 1 mapping used by this milestone.",
                    "confidence": "low",
                    "notes": "Excluded from family-level claims unless support is available.",
                }
            )
            continue
        description, family, justification, confidence = OFFICIAL_ACCIDENTS[accident_class]
        rows.append(
            {
                "class": accident_class,
                "official_description": description,
                "family": family,
                "source": SOURCE,
                "justification": justification,
                "confidence": confidence,
                "notes": "Family is a research taxonomy derived from the official description; it is not an NPPAD-native label.",
            }
        )
    return pd.DataFrame(rows)


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    supports = pd.Series(y_true).value_counts().reindex(labels, fill_value=0).to_numpy(dtype=int)
    observed = [label for label, support in zip(labels, supports) if support > 0]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_observed": float(f1_score(y_true, y_pred, labels=observed, average="macro", zero_division=0)) if observed else np.nan,
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=observed, average="macro", zero_division=0)) if observed else np.nan,
        "n_classes_eval": int(len(observed)),
        "minimum_support": int(supports[supports > 0].min()) if np.any(supports > 0) else 0,
    }


def _support_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    supports = pd.Series(y_true).value_counts().reindex(labels, fill_value=0)
    return [
        {**base, "label": label, "recall": float(recall) if supports[label] else np.nan, "support": int(supports[label])}
        for label, recall in zip(labels, recalls)
    ]


def _confusion_rows(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for true_label in labels:
        for predicted_label in labels:
            rows.append(
                {
                    **base,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(((y_true == true_label) & (y_pred == predicted_label)).sum()),
                }
            )
    return rows


def evaluate_hierarchy(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    fine_predictions: pd.DataFrame,
    mapping: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compare fine, family, and two-stage predictions on the same trajectory splits."""
    class_to_family = dict(zip(mapping["class"], mapping["family"]))
    family_labels = sorted({class_to_family[label] for label in CLASS_LABELS})
    strict = strict_process_features()
    columns = feature_columns(strict)
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for (split_type, split_id), split in assignments.groupby(["split_type", "split_id"], sort=False):
        active = split.loc[split["partition"].isin(["train", "validation", "test"])]
        split_frame = dataset.merge(active[["sample_id", "partition"]], on="sample_id", how="inner", validate="many_to_one")
        split_frame["family"] = split_frame["accident_class"].map(class_to_family)
        seed_values = split["seed"].dropna()
        seed = int(seed_values.iloc[0]) if not seed_values.empty else 20260921
        for window_s in WINDOWS_S:
            window = split_frame.loc[split_frame["window_s"].eq(window_s)].copy()
            train = window.loc[window["partition"].eq("train")]
            validation = window.loc[window["partition"].eq("validation")]
            test = window.loc[window["partition"].eq("test")]
            x_train = train[columns]
            x_validation = validation[columns]
            x_test = test[columns]
            for family in family_labels:
                family_train = train.loc[train["family"].eq(family)]
                family_test = test.loc[test["family"].eq(family)]
                support_rows.append(
                    {
                        "split_type": split_type,
                        "split_id": split_id,
                        "window_s": int(window_s),
                        "unit": "family",
                        "label": family,
                        "train_support": int(len(family_train)),
                        "validation_support": int((validation["family"] == family).sum()),
                        "test_support": int(len(family_test)),
                        "support_warning": bool(len(family_train) < 20 or len(family_test) < 5),
                    }
                )
            family_models: dict[str, tuple[object | None, str, str]] = {}
            family_validation: list[tuple[str, str, float, float, float]] = []
            for model_name in MODEL_NAMES:
                model = model_suite(seed)[model_name]
                model.fit(x_train, train["family"])
                validation_pred = model.predict(x_validation)
                test_pred = model.predict(x_test)
                validation_values = _metric_values(validation["family"].to_numpy(), validation_pred, family_labels)
                family_validation.append((model_name, "family_model", validation_values["macro_f1"], validation_values["balanced_accuracy"], validation_values["accuracy"]))
                family_models[model_name] = (model, "fitted", "")
            selected_family = sorted(family_validation, key=lambda row: (-row[2], -row[3], -row[4], row[0]))[0][0]
            selected_family_model = family_models[selected_family][0]
            family_test_pred = selected_family_model.predict(x_test)
            family_values = _metric_values(test["family"].to_numpy(), family_test_pred, family_labels)
            base = {
                "split_type": split_type,
                "split_id": split_id,
                "window_s": int(window_s),
                "evaluation_scope": "family_model",
                "model": selected_family,
                "selected_for_test": True,
                **family_values,
            }
            metric_rows.append(base)
            per_class_rows.extend(_support_rows(test["family"].to_numpy(), family_test_pred, family_labels, {**base, "level": "family"}))
            confusion_rows.extend(_confusion_rows(test["family"].to_numpy(), family_test_pred, family_labels, {**base, "level": "family"}))

            fine = fine_predictions.loc[
                fine_predictions["split_type"].eq(split_type)
                & fine_predictions["split_id"].eq(split_id)
                & fine_predictions["window_s"].eq(window_s)
                & fine_predictions["input_group"].eq("A_absolute_38")
            ]
            if len(fine) != len(test):
                raise AssertionError(f"Fine prediction/test mismatch for {split_type}/{split_id}/{window_s}s")
            fine = fine.set_index("sample_id").loc[test["sample_id"]].reset_index()
            fine_true_class = fine["true_class"].to_numpy()
            fine_pred_class = fine["predicted_class"].to_numpy()
            fine_values = _metric_values(fine_true_class, fine_pred_class, list(CLASS_LABELS))
            fine_base = {
                "split_type": split_type,
                "split_id": split_id,
                "window_s": int(window_s),
                "evaluation_scope": "fine_grained_12_class",
                "model": str(fine["model"].iloc[0]),
                "selected_for_test": True,
                **fine_values,
            }
            metric_rows.append(fine_base)
            per_class_rows.extend(_support_rows(fine_true_class, fine_pred_class, list(CLASS_LABELS), {**fine_base, "level": "subtype"}))
            confusion_rows.extend(_confusion_rows(fine_true_class, fine_pred_class, list(CLASS_LABELS), {**fine_base, "level": "subtype"}))
            fine_true_family = fine["true_class"].map(class_to_family).to_numpy()
            fine_pred_family = fine["predicted_class"].map(class_to_family).to_numpy()
            mapped_values = _metric_values(fine_true_family, fine_pred_family, family_labels)
            mapped_base = {
                "split_type": split_type,
                "split_id": split_id,
                "window_s": int(window_s),
                "evaluation_scope": "fine_predictions_mapped_to_family",
                "model": str(fine["model"].iloc[0]),
                "selected_for_test": True,
                **mapped_values,
            }
            metric_rows.append(mapped_base)
            per_class_rows.extend(_support_rows(fine_true_family, fine_pred_family, family_labels, {**mapped_base, "level": "family"}))
            confusion_rows.extend(_confusion_rows(fine_true_family, fine_pred_family, family_labels, {**mapped_base, "level": "family"}))

            stage_models: dict[str, tuple[object | None, str]] = {}
            eligible_families: list[str] = []
            for family in family_labels:
                family_train = train.loc[train["family"].eq(family)]
                subtype_counts = family_train["accident_class"].value_counts()
                if subtype_counts.empty:
                    stage_models[family] = (None, "")
                    continue
                if len(subtype_counts) <= 1:
                    stage_models[family] = (None, str(subtype_counts.index[0]))
                    continue
                if len(family_train) < 20 or int(subtype_counts.min()) < 5:
                    majority = str(subtype_counts.sort_values(ascending=False).index[0])
                    stage_models[family] = (None, majority)
                    continue
                eligible_families.append(family)
                family_validation_rows = validation.loc[validation["family"].eq(family)]
                candidates: list[tuple[str, float, float, float, object]] = []
                for model_name in MODEL_NAMES:
                    subtype_model = model_suite(seed)[model_name]
                    subtype_model.fit(family_train[columns], family_train["accident_class"])
                    validation_pred = subtype_model.predict(family_validation_rows[columns])
                    subtype_labels = sorted(subtype_counts.index.tolist())
                    values = _metric_values(family_validation_rows["accident_class"].to_numpy(), validation_pred, subtype_labels)
                    candidates.append((model_name, values["macro_f1"], values["balanced_accuracy"], values["accuracy"], subtype_model))
                best = sorted(candidates, key=lambda row: (-row[1], -row[2], -row[3], row[0]))[0]
                stage_models[family] = (best[4], best[0])
            pipeline_pred = []
            for row in test.itertuples(index=False):
                predicted_family = str(family_test_pred[len(pipeline_pred)])
                stage_model, fallback = stage_models[predicted_family]
                if stage_model is None:
                    prediction = fallback
                else:
                    single = pd.DataFrame([row._asdict()])[columns]
                    prediction = str(stage_model.predict(single)[0])
                pipeline_pred.append(prediction)
            pipeline_pred_array = np.asarray(pipeline_pred)
            pipeline_values = _metric_values(test["accident_class"].to_numpy(), pipeline_pred_array, list(CLASS_LABELS))
            pipeline_base = {
                "split_type": split_type,
                "split_id": split_id,
                "window_s": int(window_s),
                "evaluation_scope": "two_stage_end_to_end",
                "model": "family_then_subtype",
                "selected_for_test": True,
                "eligible_families": ";".join(eligible_families),
                **pipeline_values,
            }
            metric_rows.append(pipeline_base)
            per_class_rows.extend(_support_rows(test["accident_class"].to_numpy(), pipeline_pred_array, list(CLASS_LABELS), {**pipeline_base, "level": "subtype"}))
            confusion_rows.extend(_confusion_rows(test["accident_class"].to_numpy(), pipeline_pred_array, list(CLASS_LABELS), {**pipeline_base, "level": "subtype"}))

    return {
        "metrics": pd.DataFrame(metric_rows),
        "per_class": pd.DataFrame(per_class_rows),
        "confusion": pd.DataFrame(confusion_rows),
        "support": pd.DataFrame(support_rows),
    }


def write_outputs(result: dict[str, pd.DataFrame], result_root: str | Path) -> None:
    root = Path(result_root)
    result["metrics"].to_csv(root / "14_hierarchical_metrics.csv", index=False)
    result["per_class"].to_csv(root / "14_hierarchical_per_class.csv", index=False)
    result["confusion"].to_csv(root / "14_hierarchical_confusion.csv", index=False)
    result["support"].to_csv(root / "14_family_support_audit.csv", index=False)
