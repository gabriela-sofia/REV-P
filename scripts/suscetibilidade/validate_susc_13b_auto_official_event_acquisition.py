"""SUSC-13B-AUTO automatic discovery/acquisition validation (offline-safe)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402

SCHEMAS = ROOT / "schemas" / "suscetibilidade"
DATASETS = ROOT / "datasets" / "suscetibilidade"
MANIFESTS = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

SCHEMA = SCHEMAS / "susc_13b_auto_discovery_schema_v1.json"
QUERY_MANIFEST = MANIFESTS / "susc_13b_auto_discovery_query_manifest_v1.csv"
DISCOVERED = MANIFESTS / "susc_13b_auto_discovered_sources_v1.csv"
DOWNLOAD_MANIFEST = MANIFESTS / "susc_13b_auto_download_manifest_v1.csv"
PARSED = DATASETS / "susc_13b_auto_observed_events_parsed_v1.csv"
CATALOG = DATASETS / "susc_13b_auto_consolidated_observed_event_catalog_v1.csv"
CATALOG_MANIFEST = MANIFESTS / "susc_13b_auto_consolidated_catalog_manifest_v1.json"
LINKAGE = DATASETS / "susc_13b_auto_event_patch_linkage_v1.csv"
DROP_README = DATASETS / "observed_event_sources_susc13b_auto" / "README.md"

DISCOVERY_REPORT = OUT / "SUSC_13B_auto_discovery_report.md"
DEBUG_LOG = OUT / "SUSC_13B_auto_discovery_debug_log.csv"
ACQ_REPORT = OUT / "SUSC_13B_auto_acquisition_report.csv"
PARSE_AUDIT = OUT / "SUSC_13B_auto_parse_audit.csv"
CONSOLIDATION_AUDIT = OUT / "SUSC_13B_auto_consolidation_audit.csv"
LINK_SUMMARY = OUT / "SUSC_13B_auto_event_patch_linkage_summary.csv"
LINK_LIMITS = OUT / "SUSC_13B_auto_event_patch_linkage_limitations.json"
LINK_GEOJSON = OUT / "SUSC_13B_auto_event_patch_linkage_geojson.geojson"
READINESS_CSV = OUT / "SUSC_13B_auto_observational_readiness.csv"
READINESS_MD = OUT / "SUSC_13B_auto_observational_readiness.md"
DIAG_CSV = OUT / "SUSC_13B_auto_event_score_diagnostics.csv"
DIAG_REGION = OUT / "SUSC_13B_auto_event_score_diagnostics_by_region.csv"
DIAG_SUMMARY = OUT / "SUSC_13B_auto_event_score_diagnostics_summary.json"
STRONG_REPORT = OUT / "SUSC_13B_AUTO_official_event_discovery_acquisition_report.md"

MANDATORY = (
    "O SUSC-13B-AUTO realiza descoberta e aquisição automática de fontes oficiais/rastreáveis "
    "para fortalecer a camada observacional de alagamento/inundação. Mesmo quando encontra "
    "eventos fortes, a etapa mantém todos os vínculos em modo review-only, não cria ground truth, "
    "não treina modelo supervisionado e não cria score v7 automaticamente."
)

REQUIRED = [
    SCHEMA, QUERY_MANIFEST, DISCOVERED, DOWNLOAD_MANIFEST, PARSED, CATALOG,
    CATALOG_MANIFEST, LINKAGE, DROP_README, DISCOVERY_REPORT, DEBUG_LOG,
    ACQ_REPORT, PARSE_AUDIT, CONSOLIDATION_AUDIT, LINK_SUMMARY, LINK_LIMITS,
    LINK_GEOJSON, READINESS_CSV, READINESS_MD, DIAG_CSV, DIAG_REGION,
    DIAG_SUMMARY, STRONG_REPORT,
]

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}
OBSERVED = STRONG | MODERATE
WEAK_NOT_OBSERVED = {"weak_risk_area_context", "weak_alert_context", "weak_administrative_context"}
RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf", ".grib"}
MAX_BYTES = 250 * 1024 * 1024


def _header(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return set(csv.DictReader(f).fieldnames or [])


def _is_false(v) -> bool:
    return str(v or "").strip().lower() == "false"


def _is_true(v) -> bool:
    return str(v or "").strip().lower() == "true"


def _has_geom(row: dict) -> bool:
    return bool(row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref"))


def _governance(rows: list[dict], label: str) -> list[str]:
    errs = []
    for i, row in enumerate(rows, start=2):
        if "can_be_ground_truth" in row and not _is_false(row.get("can_be_ground_truth")):
            errs.append(f"{label}:{i}: can_be_ground_truth != false")
        if "can_be_used_as_ground_truth" in row and not _is_false(row.get("can_be_used_as_ground_truth")):
            errs.append(f"{label}:{i}: can_be_used_as_ground_truth != false")
        if "allowed_for_training" in row and not _is_false(row.get("allowed_for_training")):
            errs.append(f"{label}:{i}: allowed_for_training != false")
        if "review_only" in row and not _is_true(row.get("review_only")):
            errs.append(f"{label}:{i}: review_only != true")
    return errs


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Official Event Acquisition Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/12] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("  OK")

    print("[2/12] Schema present with governance closed...")
    schema = read_json(SCHEMA)
    if schema.get("can_be_ground_truth") is not False:
        errors.append("schema: can_be_ground_truth != false")
    if schema.get("allowed_for_training") is not False:
        errors.append("schema: allowed_for_training != false")
    if schema.get("review_only") is not True:
        errors.append("schema: review_only != true")
    if not [e for e in errors if e.startswith("schema")]:
        print("  OK")

    print("[3/12] Parsed/linkage columns present...")
    required_fields = set(schema.get("required_fields", {}).keys())
    parsed_header = _header(PARSED)
    for col in required_fields:
        if col not in parsed_header:
            errors.append(f"parsed missing column: {col}")
    for col in ("linkage_id", "event_id", "patch_id", "spatial_relation",
                "can_be_used_for_observational_evaluation", "can_be_ground_truth",
                "allowed_for_training", "review_only"):
        if col not in _header(LINKAGE):
            errors.append(f"linkage missing column: {col}")
    if not [e for e in errors if "missing column" in e]:
        print("  OK")

    print("[4/12] Governance flags closed across tables...")
    tables = {
        "query_manifest": read_csv(QUERY_MANIFEST),
        "discovered": read_csv(DISCOVERED),
        "downloads": read_csv(DOWNLOAD_MANIFEST),
        "parsed": read_csv(PARSED),
        "catalog": read_csv(CATALOG),
        "consolidation_audit": read_csv(CONSOLIDATION_AUDIT),
        "linkage": read_csv(LINKAGE),
        "acq_report": read_csv(ACQ_REPORT),
        "parse_audit": read_csv(PARSE_AUDIT),
        "link_summary": read_csv(LINK_SUMMARY),
        "readiness": read_csv(READINESS_CSV),
        "diag": read_csv(DIAG_CSV),
    }
    for label, rows in tables.items():
        errors.extend(_governance(rows, label))
    for jpath in (LINK_LIMITS, CATALOG_MANIFEST, DIAG_SUMMARY):
        obj = read_json(jpath)
        if obj.get("can_be_ground_truth") is not False:
            errors.append(f"{jpath.name}: can_be_ground_truth != false")
        if obj.get("allowed_for_training") is not False:
            errors.append(f"{jpath.name}: allowed_for_training != false")
        if obj.get("review_only") is not True:
            errors.append(f"{jpath.name}: review_only != true")
    gj = read_json(LINK_GEOJSON)
    if gj.get("review_only") is not True or gj.get("can_be_ground_truth") is not False:
        errors.append("geojson: governance flags not closed")
    if not [e for e in errors if "_truth" in e or "training" in e or "review_only" in e or "governance" in e]:
        print("  OK")

    print("[5/12] Strong events require geometry and date...")
    for i, row in enumerate(tables["parsed"], start=2):
        if row.get("evidence_level") in STRONG:
            if not _has_geom(row):
                errors.append(f"parsed:{i}: strong event without geometry")
            if not (row.get("event_date") or row.get("event_period_start")):
                errors.append(f"parsed:{i}: strong event without date/period")
    for i, row in enumerate(tables["catalog"], start=2):
        if row.get("evidence_level") in STRONG:
            if not _has_geom(row) or not (row.get("event_date") or row.get("event_period_start")):
                errors.append(f"catalog:{i}: strong event without geometry/date")
    if not [e for e in errors if "strong event" in e]:
        print("  OK")

    print("[6/12] Risk/alert not promoted to observed/strong...")
    for i, row in enumerate(tables["parsed"] + tables["catalog"], start=2):
        if row.get("evidence_level") in WEAK_NOT_OBSERVED and row.get("is_observed_event") not in (None, "", "false"):
            errors.append(f"row {i}: weak context marked observed")
        text = " ".join([row.get("source_title", ""), row.get("classification_reason", ""),
                         row.get("evidence_level", ""), row.get("notes", "")]).lower()
        if ("risco" in text or "risk" in text or "alerta" in text or "alert" in text) and row.get("evidence_level") in STRONG:
            errors.append(f"row {i}: risk/alert source promoted to strong")
    if not [e for e in errors if "weak context" in e or "promoted" in e]:
        print("  OK")

    print("[7/12] Administrative without geometry not patch-level...")
    for i, row in enumerate(tables["linkage"], start=2):
        if row.get("evidence_level") in WEAK_NOT_OBSERVED and row.get("patch_id") not in ("", "REGION_LEVEL_NO_PATCH_RESOLUTION"):
            errors.append(f"linkage:{i}: weak/admin context resolved to a patch")
    if not [e for e in errors if "weak/admin" in e]:
        print("  OK")

    print("[8/12] Linkage evaluation only for observed levels with real relation...")
    for i, row in enumerate(tables["linkage"], start=2):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            if row.get("evidence_level") not in OBSERVED:
                errors.append(f"linkage:{i}: non-observed level allowed for evaluation")
            if row.get("spatial_relation") in {"same_region_period_context", "insufficient_for_patch_link"}:
                errors.append(f"linkage:{i}: regional/insufficient relation allowed for evaluation")
    if not [e for e in errors if "evaluation" in e]:
        print("  OK")

    print("[9/12] Reports include mandatory phrase + readiness/no-score-v7...")
    report_text = STRONG_REPORT.read_text(encoding="utf-8")
    if MANDATORY not in report_text:
        errors.append("strong report missing mandatory statement")
    readiness = {r["metric"]: r["value"] for r in tables["readiness"]}
    if "ready_for_score_v7" not in readiness:
        errors.append("readiness missing ready_for_score_v7 metric")
    if not [e for e in errors if "mandatory" in e or "ready_for_score_v7" in e]:
        print("  OK")

    print("[10/12] SUSC-03 matrix unchanged...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    else:
        print("  OK")

    print("[11/12] No persisted model / no score v7 created...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    if (DATASETS / "susc_score_v7_candidate_by_patch_v1.csv").exists():
        errors.append("score v7 artifact exists; SUSC-13B-AUTO must not create score v7")
    diag = read_json(DIAG_SUMMARY)
    if diag.get("score_v7_created") is not False:
        errors.append("diag summary: score_v7_created != false")
    if not [e for e in errors if "model artifact" in e or "score v7" in e]:
        print("  OK")

    print("[12/12] Controlled acquisition policy (no raster / >250MB / network offline OK)...")
    for i, row in enumerate(tables["downloads"], start=2):
        if row.get("download_status") == "downloaded":
            ext = (row.get("content_ext") or "").lower()
            if ext in RASTER_EXTS:
                errors.append(f"downloads:{i}: raster downloaded")
            try:
                if row.get("file_size_bytes") and int(row["file_size_bytes"]) > MAX_BYTES:
                    errors.append(f"downloads:{i}: file > 250MB")
            except ValueError:
                errors.append(f"downloads:{i}: invalid file_size_bytes")
    if not [e for e in errors if "raster" in e or "250MB" in e or "file_size" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-13B-AUTO validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
