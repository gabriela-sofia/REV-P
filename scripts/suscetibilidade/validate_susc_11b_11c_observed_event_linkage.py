"""SUSC-11B/11C observed-event linkage validation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402
import validate_susc_10a_score_v6_candidate as validate_10a  # noqa: E402

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_11b_observed_event_schema_v1.json"
REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_source_registry_v1.csv"
DOWNLOAD_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_download_manifest_v1.csv"
PARSE_AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11B_observed_event_parse_audit.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_11b_observed_event_catalog_v1.csv"
CATALOG_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_catalog_manifest_v1.json"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_11c_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11C_event_patch_linkage_summary.csv"
LIMITATIONS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11C_event_patch_linkage_limitations.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11B_11C_observed_event_linkage_report.md"

MANDATORY = (
    "O SUSC-11B/11C organiza ocorrências reais/documentais de alagamento/inundação e "
    "testa sua ligação espacial com patches. Esses vínculos são evidência observacional "
    "review-only e não constituem ground truth supervisionado, predição operacional ou "
    "confirmação automática de ocorrência por patch."
)

REQUIRED = [
    SCHEMA,
    REGISTRY,
    DOWNLOAD_MANIFEST,
    PARSE_AUDIT,
    CATALOG,
    CATALOG_MANIFEST,
    LINKAGE,
    SUMMARY,
    LIMITATIONS,
    REPORT,
]

LINKAGE_FIELDS = {
    "linkage_id",
    "event_id",
    "patch_id",
    "region",
    "event_type",
    "evidence_level",
    "spatial_relation",
    "temporal_relation",
    "distance_m",
    "patch_score_v6",
    "patch_score_class_v6",
    "source_confidence",
    "linkage_confidence",
    "can_be_used_as_observed_evidence",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
    "requires_manual_review",
    "interpretation",
    "limitations",
}

REQUIRED_SUMMARY_METRICS = {
    "total_events",
    "total_patches",
    "total_links",
    "links_by_region",
    "links_by_spatial_relation",
    "links_by_evidence_level",
    "strong_links",
    "moderate_links",
    "weak_links",
    "events_linked_to_any_patch",
    "patches_with_observed_event_evidence",
    "patches_with_only_contextual_evidence",
    "review_only_count",
    "training_allowed_count",
    "ground_truth_count",
}

STRONG_RELATIONS = {"exact_patch_overlap", "event_polygon_intersects_patch", "event_point_inside_patch"}
BLOCKED_STRONG_LEVELS = {
    "risk_area_context",
    "alert_only",
    "administrative_record_only",
    "documentary_context_only",
}


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return (csv.DictReader(f).fieldnames or [])


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def _is_false(value: str | None) -> bool:
    return str(value or "").strip().lower() == "false"


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


def _summary_values() -> dict[str, str]:
    return {r["metric"]: r["value"] for r in read_csv(SUMMARY)}


def main() -> int:
    print("=" * 60)
    print("SUSC-11B/11C Observed Event Linkage Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/10] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if not errors:
        print("  OK")

    if errors:
        print("\nFAILED: required artifacts missing")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    print("[2/10] Basic schemas/columns...")
    schema = read_json(SCHEMA)
    if schema.get("review_only") is not True or schema.get("allowed_for_training") is not False:
        errors.append("schema governance flags incorrect")
    missing_linkage_cols = LINKAGE_FIELDS - set(_read_header(LINKAGE))
    if missing_linkage_cols:
        errors.append(f"linkage missing columns: {sorted(missing_linkage_cols)}")
    summary_metrics = set(_summary_values())
    missing_metrics = REQUIRED_SUMMARY_METRICS - summary_metrics
    if missing_metrics:
        errors.append(f"summary missing metrics: {sorted(missing_metrics)}")
    if not [e for e in errors if "schema" in e or "missing columns" in e or "summary missing" in e]:
        print("  OK")

    print("[3/10] Governance flags remain closed...")
    tables = {
        "registry": read_csv(REGISTRY),
        "download_manifest": read_csv(DOWNLOAD_MANIFEST),
        "parse_audit": read_csv(PARSE_AUDIT),
        "catalog": read_csv(CATALOG),
        "linkage": read_csv(LINKAGE),
        "summary": read_csv(SUMMARY),
    }
    for label, rows in tables.items():
        errors.extend(_governance_errors(rows, label))
    if read_json(CATALOG_MANIFEST).get("allowed_for_training") is not False:
        errors.append("catalog manifest allowed_for_training != false")
    if read_json(LIMITATIONS).get("allowed_for_training") is not False:
        errors.append("limitations allowed_for_training != false")
    if not [e for e in errors if "ground_truth" in e or "allowed_for_training" in e or "review_only" in e]:
        print("  OK")

    print("[4/10] Strong links require explicit observed geometry...")
    catalog_by_id = {r["event_id"]: r for r in tables["catalog"]}
    for i, row in enumerate(tables["linkage"], start=2):
        level = row.get("evidence_level", "")
        relation = row.get("spatial_relation", "")
        confidence = row.get("linkage_confidence", "")
        event = catalog_by_id.get(row.get("event_id", ""), {})
        has_geom = bool(event.get("bbox") or (event.get("lat") and event.get("lon")) or event.get("geojson_ref"))
        if confidence == "strong":
            if relation not in STRONG_RELATIONS:
                errors.append(f"linkage:{i}: strong link has invalid relation {relation}")
            if not has_geom:
                errors.append(f"linkage:{i}: strong link without explicit geometry")
            if level in BLOCKED_STRONG_LEVELS:
                errors.append(f"linkage:{i}: blocked evidence level promoted to strong")
    if not [e for e in errors if "strong link" in e or "promoted to strong" in e]:
        print("  OK")

    print("[5/10] Observed-evidence rows require geometry and date...")
    for i, row in enumerate(tables["linkage"], start=2):
        if not _is_true(row.get("can_be_used_as_observed_evidence")):
            continue
        event = catalog_by_id.get(row["event_id"], {})
        has_geom = bool(event.get("bbox") or (event.get("lat") and event.get("lon")) or event.get("geojson_ref"))
        has_date = bool(event.get("event_date") or event.get("event_period_start"))
        if not (has_geom and has_date):
            errors.append(f"linkage:{i}: observed evidence without geometry+date")
        if row.get("patch_id") == "REGION_LEVEL_NO_PATCH_RESOLUTION":
            errors.append(f"linkage:{i}: observed evidence is region-only")
    if not [e for e in errors if "observed evidence" in e]:
        print("  OK")

    print("[6/10] Summary counts are coherent...")
    summary = _summary_values()
    if int(summary["total_events"]) != len(catalog_by_id):
        errors.append("summary total_events mismatch")
    if int(summary["total_links"]) != len(tables["linkage"]):
        errors.append("summary total_links mismatch")
    if int(summary["training_allowed_count"]) != 0:
        errors.append("summary training_allowed_count != 0")
    if int(summary["ground_truth_count"]) != 0:
        errors.append("summary ground_truth_count != 0")
    if not [e for e in errors if "summary" in e and "missing" not in e]:
        print("  OK")

    print("[7/10] Report mandatory statement...")
    report_text = REPORT.read_text(encoding="utf-8")
    if MANDATORY not in report_text:
        errors.append("report missing mandatory statement")
    else:
        print("  OK")

    print("[8/10] SUSC-03 and SUSC-10A still valid...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix SHA256 changed")
    if validate_10a.main() != 0:
        errors.append("SUSC-10A validator failed")
    if not [e for e in errors if "SUSC-03" in e or "SUSC-10A" in e]:
        print("  OK")

    print("[9/10] No persisted model artifacts...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")

    print("[10/10] Limitations document required blockers...")
    limits_text = json.dumps(read_json(LIMITATIONS), ensure_ascii=False).lower()
    for phrase in (
        "sem ground truth",
        "sem treino supervisionado",
        "sem predicao operacional",
        "fontes sem url direta",
        "fontes nao baixadas",
        "risco/alerta/contexto nao sao ocorrencia",
        "link regional nao confirma evento por patch",
    ):
        if phrase not in limits_text:
            errors.append(f"limitations missing phrase: {phrase}")
    if not [e for e in errors if "limitations missing" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-11B/11C validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
