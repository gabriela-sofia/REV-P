"""SUSC-13A strong observed-event acquisition validation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_13a_strong_observed_event_schema_v1.json"
REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_source_registry_v1.csv"
DOWNLOADS = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_download_manifest_v1.csv"
DROP_README = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13a" / "README.md"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_observed_events_parsed_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_event_patch_linkage_v1.csv"
ACQ_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_acquisition_report.csv"
PARSE_AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_parse_audit.csv"
LINK_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_patch_linkage_summary.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_patch_linkage_limitations.json"
GAP_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_manual_acquisition_gap_report.md"
REQUEST_TABLE = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_manual_source_request_table.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_observed_event_acquisition_report.md"

MANDATORY = (
    "O SUSC-13A busca evidências observacionais mais fortes de alagamento/inundação "
    "para reduzir a fragilidade detectada em SUSC-11/12. Mesmo eventos fortes permanecem "
    "review-only nesta etapa e não criam ground truth, treino supervisionado ou confirmação "
    "operacional automática por patch."
)

REQUIRED = [
    SCHEMA,
    REGISTRY,
    DOWNLOADS,
    DROP_README,
    PARSED,
    LINKAGE,
    ACQ_REPORT,
    PARSE_AUDIT,
    LINK_SUMMARY,
    LIMITS,
    GAP_REPORT,
    REQUEST_TABLE,
    REPORT,
]

STRONG_LEVELS = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
OBSERVED_LEVELS = STRONG_LEVELS | {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}
WEAK_NOT_OBSERVED = {"weak_risk_area_context", "weak_alert_context", "weak_administrative_context"}


def _header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return set(csv.DictReader(f).fieldnames or [])


def _is_false(v) -> bool:
    return str(v or "").strip().lower() == "false"


def _is_true(v) -> bool:
    return str(v or "").strip().lower() == "true"


def _has_geom(row: dict) -> bool:
    return bool(row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref"))


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
    print("SUSC-13A Strong Observed Event Acquisition Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/10] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if not errors:
        print("  OK")
    else:
        for err in errors:
            print(f"  ERROR: {err}")
        return 1

    print("[2/10] Schema and columns...")
    schema = read_json(SCHEMA)
    required_fields = set(schema.get("required_fields", {}).keys())
    parsed_header = _header(PARSED)
    linkage_header = _header(LINKAGE)
    for col in required_fields:
        if col not in parsed_header:
            errors.append(f"parsed missing column: {col}")
    for col in (
        "linkage_id",
        "event_id",
        "patch_id",
        "spatial_relation",
        "can_be_used_for_observational_evaluation",
        "can_be_ground_truth",
        "allowed_for_training",
        "review_only",
    ):
        if col not in linkage_header:
            errors.append(f"linkage missing column: {col}")
    if not [e for e in errors if "missing column" in e]:
        print("  OK")

    print("[3/10] Governance flags closed...")
    tables = {
        "registry": read_csv(REGISTRY),
        "downloads": read_csv(DOWNLOADS),
        "parsed": read_csv(PARSED),
        "linkage": read_csv(LINKAGE),
        "acq_report": read_csv(ACQ_REPORT),
        "parse_audit": read_csv(PARSE_AUDIT),
        "link_summary": read_csv(LINK_SUMMARY),
        "request_table": read_csv(REQUEST_TABLE),
    }
    for label, rows in tables.items():
        errors.extend(_governance(rows, label))
    limits = read_json(LIMITS)
    if limits.get("can_be_ground_truth") is not False:
        errors.append("limits: can_be_ground_truth != false")
    if limits.get("allowed_for_training") is not False:
        errors.append("limits: allowed_for_training != false")
    if limits.get("review_only") is not True:
        errors.append("limits: review_only != true")
    if not [e for e in errors if "ground_truth" in e or "allowed_for_training" in e or "review_only" in e]:
        print("  OK")

    print("[4/10] Strong events require geometry and date...")
    for i, row in enumerate(tables["parsed"], start=2):
        if row["evidence_level"] in STRONG_LEVELS:
            if not _has_geom(row):
                errors.append(f"parsed:{i}: strong event without geometry")
            if not (row.get("event_date") or row.get("event_period_start")):
                errors.append(f"parsed:{i}: strong event without date/period")
    if not [e for e in errors if "strong event" in e]:
        print("  OK")

    print("[5/10] Risk/alert/admin not promoted to observed event...")
    for i, row in enumerate(tables["parsed"], start=2):
        if row["evidence_level"] in WEAK_NOT_OBSERVED and row.get("is_observed_event") != "false":
            errors.append(f"parsed:{i}: weak context marked observed")
        text = " ".join([row.get("source_name", ""), row.get("notes", ""), row.get("evidence_level", "")]).lower()
        if ("risco" in text or "risk" in text) and row["evidence_level"] in STRONG_LEVELS:
            errors.append(f"parsed:{i}: risk source promoted to strong")
        if ("alerta" in text or "alert" in text) and row["evidence_level"] in STRONG_LEVELS:
            errors.append(f"parsed:{i}: alert source promoted to strong")
    if not [e for e in errors if "promoted" in e or "weak context" in e]:
        print("  OK")

    print("[6/10] Linkage evaluation only strong/moderate geometry...")
    for i, row in enumerate(tables["linkage"], start=2):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            if row.get("evidence_level") not in OBSERVED_LEVELS:
                errors.append(f"linkage:{i}: non-observed level allowed for evaluation")
            if row.get("spatial_relation") in {"same_region_period_context", "insufficient_for_patch_link"}:
                errors.append(f"linkage:{i}: regional/insufficient relation allowed for evaluation")
    if not [e for e in errors if "evaluation" in e]:
        print("  OK")

    print("[7/10] Reports include mandatory phrase and manual gaps...")
    report_text = REPORT.read_text(encoding="utf-8")
    gap_text = GAP_REPORT.read_text(encoding="utf-8")
    if MANDATORY not in report_text:
        errors.append("final report missing mandatory statement")
    for word in ("Recife", "Petropolis", "Curitiba"):
        if word not in gap_text:
            errors.append(f"gap report missing region: {word}")
    if not [e for e in errors if "report" in e or "mandatory" in e]:
        print("  OK")

    print("[8/10] SUSC-03 matrix unchanged...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    else:
        print("  OK")

    print("[9/10] No persisted model artifacts / no score v7 creation...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    score_v7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"
    if score_v7.exists():
        errors.append("score v7 artifact exists; SUSC-13A must not create score v7")
    if not [e for e in errors if "model artifact" in e or "score v7" in e]:
        print("  OK")

    print("[10/10] Controlled acquisition policy...")
    for i, row in enumerate(tables["downloads"], start=2):
        if row.get("download_status") == "downloaded":
            ext = row.get("content_ext", "").lower()
            if ext in {".tif", ".tiff", ".jp2", ".img", ".nc", ".hdf", ".grib"}:
                errors.append(f"downloads:{i}: raster downloaded")
            try:
                if row.get("file_size_bytes") and int(row["file_size_bytes"]) > 100 * 1024 * 1024:
                    errors.append(f"downloads:{i}: file > 100MB")
            except ValueError:
                errors.append(f"downloads:{i}: invalid file_size_bytes")
    if not [e for e in errors if "download" in e or "raster" in e or "100MB" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"  ERROR: {err}")
        return 1
    print("PASSED: All SUSC-13A validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
