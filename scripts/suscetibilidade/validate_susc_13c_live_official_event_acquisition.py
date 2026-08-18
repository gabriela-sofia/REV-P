"""SUSC-13C-LIVE official event acquisition validation."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402

DATASETS = ROOT / "datasets" / "suscetibilidade"
MANIFESTS = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

HEALTH_JSON = OUT / "SUSC_13C_live_network_healthcheck.json"
HEALTH_MD = OUT / "SUSC_13C_live_network_healthcheck.md"
LIVE_DISCOVERY_REPORT = OUT / "SUSC_13C_live_discovery_report.md"
LIVE_SOURCES = OUT / "SUSC_13C_live_discovered_sources.csv"
CKAN_CSV = OUT / "SUSC_13C_live_ckan_resources.csv"
ARCGIS_CSV = OUT / "SUSC_13C_live_arcgis_layers.csv"
WFS_CSV = OUT / "SUSC_13C_live_wfs_layers.csv"
HTML_CSV = OUT / "SUSC_13C_live_html_file_links.csv"
DOWNLOAD_MANIFEST = MANIFESTS / "susc_13c_live_download_manifest_v1.csv"
ACQ_REPORT = OUT / "SUSC_13C_live_acquisition_report.md"
PARSE_AUDIT = OUT / "SUSC_13C_live_parse_audit.csv"
PARSED = DATASETS / "susc_13c_live_observed_events_parsed_v1.csv"
CATALOG = DATASETS / "susc_13c_consolidated_observed_event_catalog_v1.csv"
CATALOG_MANIFEST = MANIFESTS / "susc_13c_consolidated_catalog_manifest_v1.json"
LINKAGE = DATASETS / "susc_13c_event_patch_linkage_v1.csv"
LINK_SUMMARY = OUT / "SUSC_13C_event_patch_linkage_summary.csv"
LINK_LIMITS = OUT / "SUSC_13C_event_patch_linkage_limitations.json"
LINK_GEOJSON = OUT / "SUSC_13C_event_patch_linkage_geojson.geojson"
DIAG_CSV = OUT / "SUSC_13C_event_score_diagnostics.csv"
DIAG_REGION = OUT / "SUSC_13C_event_score_diagnostics_by_region.csv"
DIAG_SUMMARY = OUT / "SUSC_13C_event_score_diagnostics_summary.json"
READINESS_CSV = OUT / "SUSC_13C_observational_readiness.csv"
READINESS_MD = OUT / "SUSC_13C_observational_readiness.md"
FINAL_REPORT = OUT / "SUSC_13C_LIVE_official_event_acquisition_report.md"
BLOCKED_REPORT = OUT / "SUSC_13C_LIVE_BLOCKED_NETWORK_REPORT.md"

MANDATORY = (
    "O SUSC-13C-LIVE executa aquisição online real de fontes oficiais/rastreáveis para tentar "
    "materializar eventos observados de alagamento/inundação com data e geometria. Mesmo quando "
    "eventos fortes ou moderados são encontrados, todos os vínculos permanecem review-only, sem "
    "ground truth, sem treino supervisionado, sem score v7 automático e sem uso operacional preditivo."
)

REQUIRED = [
    HEALTH_JSON, HEALTH_MD, LIVE_DISCOVERY_REPORT, LIVE_SOURCES, CKAN_CSV, ARCGIS_CSV,
    WFS_CSV, HTML_CSV, DOWNLOAD_MANIFEST, ACQ_REPORT, PARSE_AUDIT, PARSED, CATALOG,
    CATALOG_MANIFEST, LINKAGE, LINK_SUMMARY, LINK_LIMITS, LINK_GEOJSON, DIAG_CSV,
    DIAG_REGION, DIAG_SUMMARY, READINESS_CSV, READINESS_MD, FINAL_REPORT,
]

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}
OBSERVED = STRONG | MODERATE
WEAK_NOT_OBSERVED = {"weak_risk_area_context", "weak_alert_context", "weak_administrative_context"}
RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf", ".grib"}
MAX_BYTES = 250 * 1024 * 1024


def _is_false(v):
    return str(v or "").strip().lower() == "false"


def _is_true(v):
    return str(v or "").strip().lower() == "true"


def _has_geom(row):
    return bool(row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref"))


def _governance(rows, label):
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
    print("SUSC-13C-LIVE Official Event Acquisition Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/13] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("  OK")

    print("[2/13] Healthcheck recorded real attempts...")
    health = read_json(HEALTH_JSON)
    if "results" not in health or not health["results"]:
        errors.append("healthcheck has no results")
    for r in health.get("results", []):
        if not (r.get("status_code") != "" or r.get("error")):
            errors.append("healthcheck row without status_code or error")
            break
    network_ok = bool(health.get("network_ok"))
    if not network_ok and not BLOCKED_REPORT.exists():
        errors.append("network not ok but blocked report missing")
    if not [e for e in errors if "healthcheck" in e or "blocked report" in e]:
        print("  OK")

    print("[3/13] Live probe CSVs have real HTTP status fields...")
    ckan = read_csv(CKAN_CSV)
    for r in ckan[:1]:
        if "status_code" not in r:
            errors.append("ckan resources missing status_code")
    dl = read_csv(DOWNLOAD_MANIFEST)
    for r in dl[:1]:
        for col in ("http_status", "content_type", "download_status", "sha256"):
            if col not in r:
                errors.append(f"download manifest missing {col}")
    if not [e for e in errors if "status_code" in e or "download manifest" in e]:
        print("  OK")

    print("[4/13] Governance flags closed across tables...")
    tables = {
        "live_sources": read_csv(LIVE_SOURCES), "ckan": ckan,
        "arcgis": read_csv(ARCGIS_CSV), "wfs": read_csv(WFS_CSV), "html": read_csv(HTML_CSV),
        "downloads": dl, "parsed": read_csv(PARSED), "catalog": read_csv(CATALOG),
        "linkage": read_csv(LINKAGE), "link_summary": read_csv(LINK_SUMMARY),
        "readiness": read_csv(READINESS_CSV), "diag": read_csv(DIAG_CSV),
        "parse_audit": read_csv(PARSE_AUDIT),
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

    print("[5/13] Strong events require geometry and date...")
    for i, row in enumerate(tables["parsed"] + tables["catalog"], start=2):
        if row.get("evidence_level") in STRONG:
            if not _has_geom(row) or not (row.get("event_date") or row.get("event_period_start")):
                errors.append(f"row {i}: strong event without geometry/date")
    if not [e for e in errors if "strong event" in e]:
        print("  OK")

    print("[6/13] Risk/alert not promoted to strong; weak not observed...")
    for i, row in enumerate(tables["parsed"] + tables["catalog"], start=2):
        if row.get("evidence_level") in WEAK_NOT_OBSERVED and row.get("is_observed_event") not in (None, "", "false"):
            errors.append(f"row {i}: weak context marked observed")
        text = " ".join([row.get("source_title", ""), row.get("classification_reason", ""),
                         row.get("evidence_level", ""), row.get("notes", "")]).lower()
        if ("risco" in text or "risk" in text or "alerta" in text or "alert" in text) and row.get("evidence_level") in STRONG:
            errors.append(f"row {i}: risk/alert promoted to strong")
    if not [e for e in errors if "weak context" in e or "promoted" in e]:
        print("  OK")

    print("[7/13] Administrative/weak without geometry not patch-level...")
    for i, row in enumerate(tables["linkage"], start=2):
        if row.get("evidence_level") in WEAK_NOT_OBSERVED and row.get("patch_id") not in ("", "REGION_LEVEL_NO_PATCH_RESOLUTION"):
            errors.append(f"linkage:{i}: weak/admin resolved to a patch")
    if not [e for e in errors if "weak/admin" in e]:
        print("  OK")

    print("[8/13] Linkage evaluation only for observed levels with real relation...")
    for i, row in enumerate(tables["linkage"], start=2):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            if row.get("evidence_level") not in OBSERVED:
                errors.append(f"linkage:{i}: non-observed level allowed for evaluation")
            if row.get("spatial_relation") in {"same_region_period_context", "insufficient_for_patch_link"}:
                errors.append(f"linkage:{i}: regional/insufficient relation allowed for evaluation")
    if not [e for e in errors if "evaluation" in e]:
        print("  OK")

    print("[9/13] Final report has mandatory phrase + readiness present...")
    if MANDATORY not in FINAL_REPORT.read_text(encoding="utf-8"):
        errors.append("final report missing mandatory statement")
    readiness = {r["metric"]: r["value"] for r in tables["readiness"]}
    if "ready_for_score_v7" not in readiness:
        errors.append("readiness missing ready_for_score_v7")
    if not [e for e in errors if "mandatory" in e or "ready_for_score_v7" in e]:
        print("  OK")

    print("[10/13] SUSC-03 matrix unchanged...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    else:
        print("  OK")

    print("[11/13] No persisted model / no score v7 created...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    if (DATASETS / "susc_score_v7_candidate_by_patch_v1.csv").exists():
        errors.append("score v7 artifact exists")
    if read_json(DIAG_SUMMARY).get("score_v7_created") is not False:
        errors.append("diag summary score_v7_created != false")
    if not [e for e in errors if "model artifact" in e or "score v7" in e]:
        print("  OK")

    print("[12/13] Controlled acquisition (no raster / >250MB downloaded)...")
    for i, row in enumerate(dl, start=2):
        if row.get("download_status") in {"downloaded", "downloaded_by_probe"}:
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

    print("[13/13] Real online attempt registered (HTTP statuses present)...")
    statuses = {str(r.get("download_status", "")) for r in dl} | {str(r.get("status_code", "")) for r in ckan}
    statuses |= {str(r.get("query_status", "")) for r in tables["arcgis"]}
    statuses |= {str(r.get("getcapabilities_status", "")) for r in tables["wfs"]}
    # at minimum the healthcheck proves real attempts; probes must carry status tokens
    if not any(s for s in statuses if s):
        errors.append("no real online attempt status recorded in probes")
    if not [e for e in errors if "online attempt" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-13C-LIVE validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
