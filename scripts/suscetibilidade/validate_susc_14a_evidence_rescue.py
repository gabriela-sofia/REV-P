"""SUSC-14A evidence rescue validation (review-only, fail-closed)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402

SCHEMAS = ROOT / "schemas" / "suscetibilidade"
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DATASETS = ROOT / "datasets" / "suscetibilidade"
MANIFESTS = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

MANDATORY = (
    "O SUSC-14A tenta resgatar evidencia observacional a partir de registros oficiais sem coordenada "
    "explicita, usando apenas referencias espaciais oficiais/rastreaveis e footprints publicos quando "
    "disponiveis. Eventos geocodificados por referencia oficial permanecem review-only, nao criam "
    "ground truth, nao liberam treino supervisionado e nao autorizam score v7 automatico."
)

REQUIRED = [
    SCHEMAS / "susc_14a_evidence_rescue_schema_v1.json",
    SCRIPTS / "discover_susc_14a_official_geocoding_references.py",
    SCRIPTS / "acquire_susc_14a_official_geocoding_references.py",
    SCRIPTS / "parse_susc_14a_official_geocoding_references.py",
    SCRIPTS / "build_susc_14a_occurrence_address_normalizer.py",
    SCRIPTS / "geocode_susc_14a_official_occurrences.py",
    SCRIPTS / "discover_susc_14a_flood_footprint_sources.py",
    SCRIPTS / "acquire_susc_14a_flood_footprints.py",
    SCRIPTS / "build_susc_14a_rescued_observed_event_catalog.py",
    SCRIPTS / "build_susc_14a_event_patch_linkage.py",
    SCRIPTS / "run_susc_14a_observational_diagnostics.py",
    MANIFESTS / "susc_14a_official_geocoding_reference_registry_v1.csv",
    MANIFESTS / "susc_14a_official_reference_download_manifest_v1.csv",
    MANIFESTS / "susc_14a_flood_footprint_manifest_v1.csv",
    MANIFESTS / "susc_14a_rescued_observed_event_catalog_manifest_v1.json",
    DATASETS / "official_geocoding_references_susc14a",
    DATASETS / "flood_footprints_susc14a",
    DATASETS / "susc_14a_official_geocoding_reference_features_v1.csv",
    DATASETS / "susc_14a_geocoded_official_occurrences_v1.csv",
    DATASETS / "susc_14a_rescued_observed_event_catalog_v1.csv",
    DATASETS / "susc_14a_event_patch_linkage_v1.csv",
    OUT / "SUSC_14A_official_reference_discovery_report.md",
    OUT / "SUSC_14A_official_reference_acquisition_report.md",
    OUT / "SUSC_14A_official_reference_parse_audit.csv",
    OUT / "SUSC_14A_official_occurrence_geocoding_audit.csv",
    OUT / "SUSC_14A_flood_footprint_discovery_report.md",
    OUT / "SUSC_14A_rescued_event_consolidation_audit.csv",
    OUT / "SUSC_14A_event_patch_linkage_summary.csv",
    OUT / "SUSC_14A_event_patch_linkage_limitations.json",
    OUT / "SUSC_14A_event_patch_linkage_geojson.geojson",
    OUT / "SUSC_14A_observational_readiness.csv",
    OUT / "SUSC_14A_observational_readiness.md",
    OUT / "SUSC_14A_score_v6_event_diagnostics.csv",
    OUT / "SUSC_14A_score_v6_event_diagnostics_by_region.csv",
    OUT / "SUSC_14A_score_v6_event_diagnostics_summary.json",
    OUT / "SUSC_14A_evidence_rescue_report.md",
]

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {
    "moderate_official_occurrence_point",
    "moderate_official_occurrence_geocoded",
    "moderate_official_street_segment_match",
    "moderate_official_flood_bbox",
}
OBSERVED = STRONG | MODERATE
WEAK_CONTEXT = {"weak_official_neighborhood_context", "weak_risk_area_context", "weak_alert_context", "weak_administrative_context"}
NON_PATCH_RELATIONS = {"same_neighborhood_context", "same_region_period_context", "insufficient_for_patch_link"}


def _is_true(value):
    return str(value or "").strip().lower() == "true"


def _is_false(value):
    return str(value or "").strip().lower() == "false"


def _has_geom(row):
    return bool(row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref"))


def _has_date(row):
    return bool(row.get("event_date") or row.get("event_period_start"))


def _governance(rows, label):
    errors = []
    for idx, row in enumerate(rows, start=2):
        if "allowed_for_training" in row and not _is_false(row.get("allowed_for_training")):
            errors.append(f"{label}:{idx}: allowed_for_training != false")
        if "can_be_ground_truth" in row and not _is_false(row.get("can_be_ground_truth")):
            errors.append(f"{label}:{idx}: can_be_ground_truth != false")
        if "can_be_used_as_ground_truth" in row and not _is_false(row.get("can_be_used_as_ground_truth")):
            errors.append(f"{label}:{idx}: can_be_used_as_ground_truth != false")
        if "review_only" in row and not _is_true(row.get("review_only")):
            errors.append(f"{label}:{idx}: review_only != true")
    return errors


def _json_governance(path):
    obj = read_json(path)
    errors = []
    if obj.get("allowed_for_training") is not False:
        errors.append(f"{path.name}: allowed_for_training != false")
    if obj.get("can_be_ground_truth") is not False:
        errors.append(f"{path.name}: can_be_ground_truth != false")
    if "can_be_used_as_ground_truth" in obj and obj.get("can_be_used_as_ground_truth") is not False:
        errors.append(f"{path.name}: can_be_used_as_ground_truth != false")
    if obj.get("review_only") is not True:
        errors.append(f"{path.name}: review_only != true")
    return errors


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Evidence Rescue Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/16] Required artifacts exist and are non-empty...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"empty: {path.relative_to(ROOT)}")
    if not errors:
        print("  OK")

    refs = read_csv(MANIFESTS / "susc_14a_official_geocoding_reference_registry_v1.csv")
    downloads = read_csv(MANIFESTS / "susc_14a_official_reference_download_manifest_v1.csv")
    features = read_csv(DATASETS / "susc_14a_official_geocoding_reference_features_v1.csv")
    geocoded = read_csv(DATASETS / "susc_14a_geocoded_official_occurrences_v1.csv")
    catalog = read_csv(DATASETS / "susc_14a_rescued_observed_event_catalog_v1.csv")
    linkage = read_csv(DATASETS / "susc_14a_event_patch_linkage_v1.csv")
    link_summary = read_csv(OUT / "SUSC_14A_event_patch_linkage_summary.csv")
    readiness = read_csv(OUT / "SUSC_14A_observational_readiness.csv")
    diag = read_csv(OUT / "SUSC_14A_score_v6_event_diagnostics.csv")
    diag_region = read_csv(OUT / "SUSC_14A_score_v6_event_diagnostics_by_region.csv")
    parse_audit = read_csv(OUT / "SUSC_14A_official_reference_parse_audit.csv")
    geocode_audit = read_csv(OUT / "SUSC_14A_official_occurrence_geocoding_audit.csv")
    consolidation_audit = read_csv(OUT / "SUSC_14A_rescued_event_consolidation_audit.csv")

    print("[2/16] Governance flags closed across CSV artifacts...")
    for label, rows in {
        "refs": refs,
        "downloads": downloads,
        "features": features,
        "geocoded": geocoded,
        "catalog": catalog,
        "linkage": linkage,
        "link_summary": link_summary,
        "readiness": readiness,
        "diag": diag,
        "diag_region": diag_region,
        "parse_audit": parse_audit,
        "geocode_audit": geocode_audit,
        "consolidation_audit": consolidation_audit,
    }.items():
        errors.extend(_governance(rows, label))
    if not [e for e in errors if "!= false" in e or "!= true" in e]:
        print("  OK")

    print("[3/16] Governance flags closed across JSON/GeoJSON artifacts...")
    before_json = len(errors)
    for path in (
        MANIFESTS / "susc_14a_rescued_observed_event_catalog_manifest_v1.json",
        OUT / "SUSC_14A_event_patch_linkage_limitations.json",
        OUT / "SUSC_14A_event_patch_linkage_geojson.geojson",
        OUT / "SUSC_14A_score_v6_event_diagnostics_summary.json",
    ):
        errors.extend(_json_governance(path))
    if len(errors) == before_json:
        print("  OK")

    print("[4/16] Strong events require date/period and explicit geometry...")
    for idx, row in enumerate(catalog, start=2):
        if row.get("evidence_level") in STRONG and (not _has_geom(row) or not _has_date(row)):
            errors.append(f"catalog:{idx}: strong event without geometry/date")
    if not [e for e in errors if "strong event" in e]:
        print("  OK")

    print("[5/16] Street segment match is never strong...")
    for idx, row in enumerate(geocoded + catalog + linkage, start=2):
        text = " ".join(str(row.get(col, "")) for col in ("evidence_level", "georeferencing_status", "match_method", "spatial_relation"))
        if "street_segment" in text and row.get("evidence_level") in STRONG:
            errors.append(f"row:{idx}: street segment promoted to strong")
        if row.get("spatial_relation") == "moderate_street_segment_intersects_patch" and row.get("linkage_confidence") == "strong":
            errors.append(f"linkage:{idx}: street segment linkage marked strong")
    if not [e for e in errors if "street segment" in e]:
        print("  OK")

    print("[6/16] Neighborhood/municipality/risk/alert do not become patch-level...")
    for idx, row in enumerate(linkage, start=2):
        if row.get("evidence_level") in WEAK_CONTEXT or row.get("spatial_relation") in NON_PATCH_RELATIONS:
            if row.get("patch_id") not in {"", "REGION_LEVEL_NO_PATCH_RESOLUTION"}:
                errors.append(f"linkage:{idx}: weak/context relation resolved to patch")
            if row.get("can_be_used_for_observational_evaluation") == "true":
                errors.append(f"linkage:{idx}: weak/context relation allowed for evaluation")
    if not [e for e in errors if "weak/context" in e]:
        print("  OK")

    print("[7/16] Non-official geocoding cannot become moderate/strong...")
    for idx, row in enumerate(geocoded, start=2):
        level = row.get("evidence_level")
        if level in OBSERVED and not row.get("georeferencing_status", "").startswith("official_"):
            errors.append(f"geocoded:{idx}: observed level without official georeferencing")
    if not [e for e in errors if "without official georeferencing" in e]:
        print("  OK")

    print("[8/16] Observational evaluation only for strong/moderate official geometry...")
    for idx, row in enumerate(linkage, start=2):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            if row.get("evidence_level") not in OBSERVED:
                errors.append(f"linkage:{idx}: non-observed level allowed for evaluation")
            if row.get("spatial_relation") in NON_PATCH_RELATIONS:
                errors.append(f"linkage:{idx}: context relation allowed for evaluation")
            if not row.get("patch_id") or row.get("patch_id") == "REGION_LEVEL_NO_PATCH_RESOLUTION":
                errors.append(f"linkage:{idx}: evaluation without patch id")
    if not [e for e in errors if "evaluation" in e]:
        print("  OK")

    print("[9/16] Readiness stays false when no strong/moderate links exist...")
    readiness_map = {r["metric"]: r["value"] for r in readiness}
    strong_mod = sum(1 for r in linkage if r.get("linkage_confidence") in {"strong", "moderate"})
    if strong_mod == 0:
        for metric in (
            "ready_for_12a_temporal",
            "ready_for_12b_feature_contrast",
            "ready_for_12c_proxy_calibration",
            "ready_for_score_v7",
        ):
            if readiness_map.get(metric) != "false":
                errors.append(f"readiness:{metric} must be false when no strong/moderate links")
        if readiness_map.get("score_v7_status") != "blocked":
            errors.append("readiness: score_v7_status must be blocked")
        if readiness_map.get("observational_evaluation_blocked") != "true":
            errors.append("readiness: observational_evaluation_blocked must be true")
    if not [e for e in errors if "readiness" in e]:
        print("  OK")

    print("[10/16] Diagnostic summary records explicit block and no score v7...")
    diag_summary = read_json(OUT / "SUSC_14A_score_v6_event_diagnostics_summary.json")
    if diag_summary.get("score_v7_created") is not False:
        errors.append("diagnostic summary score_v7_created != false")
    if strong_mod == 0 and diag_summary.get("reason") != "no_strong_or_moderate_patch_links":
        errors.append("diagnostic summary missing no_strong_or_moderate_patch_links reason")
    if not [e for e in errors if "diagnostic summary" in e]:
        print("  OK")

    print("[11/16] Final report contains mandatory statement and blocker sections...")
    report = (OUT / "SUSC_14A_evidence_rescue_report.md").read_text(encoding="utf-8")
    for text in (MANDATORY, "## Achado negativo principal", "## Consequencia para score v7"):
        if text not in report:
            errors.append(f"final report missing: {text[:60]}")
    if not [e for e in errors if "final report" in e]:
        print("  OK")

    print("[12/16] SUSC-03 matrix intact and score v6 present...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    if not (DATASETS / "susc_score_v6_candidate_by_patch_v1.csv").exists():
        errors.append("score v6 candidate missing")
    if not [e for e in errors if "SUSC-03" in e or "score v6" in e]:
        print("  OK")

    print("[13/16] No score v7 artifact exists...")
    score_v7 = DATASETS / "susc_score_v7_candidate_by_patch_v1.csv"
    if score_v7.exists():
        errors.append(f"score v7 artifact exists: {score_v7.relative_to(ROOT)}")
    else:
        print("  OK")

    print("[14/16] No persisted model artifacts were created...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")

    print("[15/16] Schema and primary tables have expected columns...")
    required_cols = {
        "geocoded": {"evidence_level", "georeferencing_status", "allowed_for_training", "can_be_ground_truth", "review_only"},
        "catalog": {"event_id", "evidence_level", "georeferencing_status", "allowed_for_training", "can_be_used_as_ground_truth", "review_only"},
        "linkage": {
            "event_id",
            "patch_id",
            "evidence_level",
            "spatial_relation",
            "can_be_used_for_observational_evaluation",
            "can_be_used_as_ground_truth",
            "allowed_for_training",
            "review_only",
        },
    }
    for label, rows in {"geocoded": geocoded, "catalog": catalog, "linkage": linkage}.items():
        cols = set(rows[0]) if rows else set()
        missing = required_cols[label] - cols
        if missing:
            errors.append(f"{label}: missing columns {sorted(missing)}")
    if not [e for e in errors if "missing columns" in e]:
        print("  OK")

    print("[16/16] Public outputs remain review-only and fail-closed...")
    if any(_is_true(r.get("allowed_for_training")) or _is_true(r.get("can_be_ground_truth")) for r in catalog + linkage + geocoded):
        errors.append("public evidence table contains training or ground truth true")
    if not [e for e in errors if "public evidence" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("PASSED: All SUSC-14A validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
