"""SUSC-12B/12C observed-event calibration validation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402

DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12b_event_feature_contrast_dataset_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_12b_event_feature_contrast_manifest_v1.json"
FEATURE_CONTRAST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_feature_contrast.csv"
SCORE_CONTRAST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_score_contrast.csv"
REGION_CONTRAST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_region_contrast.csv"
VALIDITY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_feature_validity_matrix.csv"
CONTRAST_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_contrast_summary.json"
RECS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_proxy_calibration_recommendations.csv"
CHANGELOG = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_score_v6_to_v7_change_log.csv"
CAL_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_proxy_calibration_summary.json"
CAL_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_proxy_calibration_report.md"
NOT_READY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_score_v7_not_ready_report.md"
INTEGRATED = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_12C_observed_event_contrast_and_calibration_report.md"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"
SCORE_V7_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v7_candidate_manifest_v1.json"
SCORE_V7_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_score_v7_candidate_summary.csv"

MANDATORY = (
    "O SUSC-12B/12C compara métricas de suscetibilidade com evidências observacionais "
    "de alagamento/inundação para calibração review-only do proxy. A análise não cria "
    "ground truth, não treina modelo supervisionado e não transforma ausência de registro "
    "em negativo verdadeiro."
)

REQUIRED = [
    DATASET,
    MANIFEST,
    FEATURE_CONTRAST,
    SCORE_CONTRAST,
    REGION_CONTRAST,
    VALIDITY,
    CONTRAST_SUMMARY,
    RECS,
    CHANGELOG,
    CAL_SUMMARY,
    CAL_REPORT,
    INTEGRATED,
]


def _header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return set(csv.DictReader(f).fieldnames or [])


def _is_false(v: str | None) -> bool:
    return str(v or "").strip().lower() == "false"


def _is_true(v: str | None) -> bool:
    return str(v or "").strip().lower() == "true"


def _governance(rows: list[dict], label: str) -> list[str]:
    errors = []
    for i, row in enumerate(rows, start=2):
        if "can_be_ground_truth" in row and not _is_false(row.get("can_be_ground_truth")):
            errors.append(f"{label}:{i}: can_be_ground_truth != false")
        if "can_be_used_as_ground_truth" in row and not _is_false(row.get("can_be_used_as_ground_truth")):
            errors.append(f"{label}:{i}: can_be_used_as_ground_truth != false")
        if "allowed_for_training" in row and not _is_false(row.get("allowed_for_training")):
            errors.append(f"{label}:{i}: allowed_for_training != false")
        if "review_only" in row and not _is_true(row.get("review_only")):
            errors.append(f"{label}:{i}: review_only != true")
    return errors


def main() -> int:
    print("=" * 60)
    print("SUSC-12B/12C Observed Event Calibration Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/9] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if not errors:
        print("  OK")
    else:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    print("[2/9] Dataset and contrast columns...")
    for col in ("contrast_group", "control_selection_reason", "can_be_ground_truth", "allowed_for_training", "review_only"):
        if col not in _header(DATASET):
            errors.append(f"dataset missing column: {col}")
    for col in ("mean_event", "mean_control", "direction_agreement", "confidence_flag"):
        if col not in _header(FEATURE_CONTRAST):
            errors.append(f"feature contrast missing column: {col}")
    if not [e for e in errors if "missing column" in e]:
        print("  OK")

    print("[3/9] Governance flags closed...")
    tables = {
        "dataset": read_csv(DATASET),
        "feature_contrast": read_csv(FEATURE_CONTRAST),
        "score_contrast": read_csv(SCORE_CONTRAST),
        "validity": read_csv(VALIDITY),
        "recommendations": read_csv(RECS),
        "changelog": read_csv(CHANGELOG),
    }
    for label, rows in tables.items():
        errors.extend(_governance(rows, label))
    for label, obj in (("manifest", read_json(MANIFEST)), ("contrast_summary", read_json(CONTRAST_SUMMARY)), ("cal_summary", read_json(CAL_SUMMARY))):
        if obj.get("can_be_ground_truth") is not False:
            errors.append(f"{label}: can_be_ground_truth != false")
        if obj.get("allowed_for_training") is not False:
            errors.append(f"{label}: allowed_for_training != false")
        if obj.get("review_only") is not True:
            errors.append(f"{label}: review_only != true")
    if not [e for e in errors if "ground_truth" in e or "allowed_for_training" in e or "review_only" in e]:
        print("  OK")

    print("[4/9] Controls are no_documented_event_control only...")
    for i, row in enumerate(tables["dataset"], start=2):
        if row["contrast_group"] == "no_documented_event_control":
            if "no_documented_event_control" not in row.get("control_selection_reason", ""):
                errors.append(f"dataset:{i}: control reason missing no_documented_event_control")
        forbidden = (row.get("contrast_group", "") + " " + row.get("control_selection_reason", "")).lower()
        if "true_negative" in forbidden or "negative_true" in forbidden:
            errors.append(f"dataset:{i}: control uses true-negative wording")
    if not [e for e in errors if "control" in e]:
        print("  OK")

    print("[5/9] Context evidence is not promoted to strong...")
    for i, row in enumerate(tables["dataset"], start=2):
        if "risk_area_context" in row.get("evidence_levels", "") and row["contrast_group"] != "weak_context_patch":
            errors.append(f"dataset:{i}: risk_area_context promoted outside weak_context_patch")
        if "administrative_record_only" in row.get("evidence_levels", "") and row["contrast_group"] != "weak_context_patch":
            errors.append(f"dataset:{i}: administrative_record_only promoted outside weak_context_patch")
    if not [e for e in errors if "promoted" in e]:
        print("  OK")

    print("[6/9] Score v7 readiness gate...")
    score_v7_exists = SCORE_V7.exists() or SCORE_V7_MANIFEST.exists() or SCORE_V7_SUMMARY.exists()
    cal_summary = read_json(CAL_SUMMARY)
    if score_v7_exists:
        if cal_summary.get("score_v7_created") is not True:
            errors.append("score v7 artifacts exist but summary score_v7_created is not true")
        if SCORE_V7.exists():
            errors.extend(_governance(read_csv(SCORE_V7), "score_v7"))
    else:
        if cal_summary.get("score_v7_created") is not False:
            errors.append("score_v7_created != false while no score v7 artifacts exist")
        if not NOT_READY.exists():
            errors.append("score v7 not ready report missing")
    if not [e for e in errors if "score v7" in e or "score_v7" in e]:
        print("  OK")

    print("[7/9] Mandatory integrated report statement...")
    if MANDATORY not in INTEGRATED.read_text(encoding="utf-8"):
        errors.append("integrated report missing mandatory statement")
    else:
        print("  OK")

    print("[8/9] No persisted model artifacts...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")

    print("[9/9] Recommendation vocabulary...")
    allowed = set(read_json(ROOT / "schemas" / "suscetibilidade" / "susc_12c_proxy_calibration_schema_v1.json")["allowed_recommendations"])
    bad = [r["recommendation"] for r in tables["recommendations"] if r["recommendation"] not in allowed]
    if bad:
        errors.append(f"invalid recommendations: {sorted(set(bad))}")
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-12B/12C validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
