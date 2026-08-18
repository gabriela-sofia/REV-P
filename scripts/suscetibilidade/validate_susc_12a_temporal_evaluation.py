"""SUSC-12A temporal evaluation validation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_12a_temporal_evaluation_schema_v1.json"
DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12a_pre_event_evaluation_dataset_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_12a_pre_event_evaluation_manifest_v1.json"
METRICS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_evaluation_metrics.csv"
EVENT_VS_CONTROL = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_event_vs_control.csv"
HIT_RATE = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_hit_rate_topk.csv"
REGION_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_region_summary.csv"
LEAKAGE_AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_leakage_audit.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_evaluation_summary.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_pre_event_evaluation_report.md"

MANDATORY = (
    "O SUSC-12A avalia se patches com evidência observada apresentavam maior suscetibilidade "
    "antes do evento, quando a temporalidade dos dados permite. A análise é review-only, não "
    "cria ground truth, não treina modelo supervisionado e não constitui predição operacional."
)

REQUIRED = [
    SCHEMA,
    DATASET,
    MANIFEST,
    METRICS,
    EVENT_VS_CONTROL,
    HIT_RATE,
    REGION_SUMMARY,
    LEAKAGE_AUDIT,
    SUMMARY,
    REPORT,
]

DATASET_COLUMNS = {
    "patch_id",
    "event_id",
    "region",
    "event_date",
    "feature_reference_date",
    "feature_temporal_status",
    "pre_event_valid",
    "temporal_leakage_risk",
    "event_patch_status",
    "control_patch_status",
    "control_group_id",
    "control_selection_reason",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
}


def _header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return set(csv.DictReader(f).fieldnames or [])


def _is_true(v: str | None) -> bool:
    return str(v or "").strip().lower() == "true"


def _is_false(v: str | None) -> bool:
    return str(v or "").strip().lower() == "false"


def _governance_errors(rows: list[dict], label: str) -> list[str]:
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
    print("SUSC-12A Temporal Evaluation Validation")
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

    print("[2/9] Dataset columns/schema...")
    missing_cols = DATASET_COLUMNS - _header(DATASET)
    if missing_cols:
        errors.append(f"dataset missing columns: {sorted(missing_cols)}")
    schema = read_json(SCHEMA)
    if schema.get("allowed_for_training") is not False or schema.get("review_only") is not True:
        errors.append("schema governance flags incorrect")
    if not [e for e in errors if "dataset missing" in e or "schema" in e]:
        print("  OK")

    print("[3/9] Governance flags closed...")
    tables = {
        "dataset": read_csv(DATASET),
        "metrics": read_csv(METRICS),
        "event_vs_control": read_csv(EVENT_VS_CONTROL),
        "hit_rate": read_csv(HIT_RATE),
        "region_summary": read_csv(REGION_SUMMARY),
        "leakage_audit": read_csv(LEAKAGE_AUDIT),
    }
    for label, rows in tables.items():
        errors.extend(_governance_errors(rows, label))
    manifest = read_json(MANIFEST)
    summary = read_json(SUMMARY)
    for label, obj in (("manifest", manifest), ("summary", summary)):
        if obj.get("can_be_ground_truth") is not False:
            errors.append(f"{label}: can_be_ground_truth != false")
        if obj.get("allowed_for_training") is not False:
            errors.append(f"{label}: allowed_for_training != false")
        if obj.get("review_only") is not True:
            errors.append(f"{label}: review_only != true")
    if not [e for e in errors if "ground_truth" in e or "allowed_for_training" in e or "review_only" in e]:
        print("  OK")

    print("[4/9] Post-event features are not treated as pre-event...")
    for i, row in enumerate(tables["dataset"], start=2):
        if row["feature_temporal_status"] == "post_event_risk" and row["pre_event_valid"] == "true":
            errors.append(f"dataset:{i}: post_event_risk with pre_event_valid=true")
        if row["temporal_leakage_risk"] == "high" and row["pre_event_valid"] == "true":
            errors.append(f"dataset:{i}: high leakage with pre_event_valid=true")
    if not [e for e in errors if "post_event_risk" in e or "high leakage" in e]:
        print("  OK")

    print("[5/9] Controls are not absence labels...")
    forbidden = ("negative" + "_true", "true" + "_negative", "negativo " + "verdadeiro", "negativos " + "verdadeiros")
    searchable_text = REPORT.read_text(encoding="utf-8").lower()
    searchable_text += "\n" + "\n".join(
        f"{r.get('control_patch_status','')} {r.get('control_selection_reason','')} {r.get('limitations','')}"
        for r in tables["dataset"]
    ).lower()
    for phrase in forbidden:
        if phrase in searchable_text:
            errors.append(f"forbidden negative-control phrase: {phrase}")
    for i, row in enumerate(tables["dataset"], start=2):
        if row["row_role"] == "control_patch" and row["control_patch_status"] != "no_documented_event_control":
            errors.append(f"dataset:{i}: control status is not no_documented_event_control")
    if not [e for e in errors if "negative" in e or "control status" in e or "negativo" in e]:
        print("  OK")

    print("[6/9] Report mandatory statement...")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    else:
        print("  OK")

    print("[7/9] Metrics/hit-rate outputs coherent...")
    metric_names = {r["metric"] for r in tables["metrics"]}
    for metric in (
        "event_patch_score_distribution_count",
        "control_patch_score_distribution_count",
        "event_enrichment_ratio",
        "low_sample_size",
    ):
        if metric not in metric_names:
            errors.append(f"metric missing: {metric}")
    if {r["top_k"] for r in tables["hit_rate"]} != {"top_10", "top_20", "top_30"}:
        errors.append("hit-rate top-k rows missing")
    if not [e for e in errors if "metric missing" in e or "hit-rate" in e]:
        print("  OK")

    print("[8/9] No persisted model artifacts...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")

    print("[9/9] Summary flags...")
    if summary.get("model_persisted") is not False:
        errors.append("summary model_persisted != false")
    if "leakage_risk_counts" not in summary or "temporal_validity_counts" not in summary:
        errors.append("summary missing temporal/leakage counts")
    if not [e for e in errors if "summary" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-12A validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
