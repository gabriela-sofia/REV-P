"""SUSC-15A precision coordinate rescue validator."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_15a_precision_observed_event_schema_v1.json"
REPORT = OUT / "SUSC_15A_PRECISION_observed_event_coordinate_rescue_report.md"
MANDATORY = (
    "O SUSC-15A-PRECISION tenta obter evidencia espacial precisa para eventos observados usando apenas geometrias "
    "diretas, referencias oficiais de enderecamento/intersecao/linear referencing e footprints rastreaveis. A etapa "
    "nao usa bairro como coordenada, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico."
)
REQUIRED = [
    SCHEMA,
    OUT / "SUSC_15A_official_occurrence_address_content_audit.csv",
    OUT / "SUSC_15A_official_occurrence_address_content_report.md",
    MAN / "susc_15a_precision_reference_source_registry_v1.csv",
    MAN / "susc_15a_precision_reference_download_manifest_v1.csv",
    DAT / "precision_references_susc15a",
    DAT / "susc_15a_precision_reference_features_v1.csv",
    DAT / "susc_15a_precision_geocoded_occurrences_v1.csv",
    DAT / "susc_15a_event_footprint_candidates_v1.csv",
    DAT / "susc_15a_calibration_eligible_event_catalog_v1.csv",
    MAN / "susc_15a_calibration_eligible_event_catalog_manifest_v1.json",
    DAT / "susc_15a_precision_event_patch_linkage_v1.csv",
    OUT / "SUSC_15A_precision_event_patch_linkage_summary.csv",
    OUT / "SUSC_15A_precision_event_patch_linkage_limitations.json",
    OUT / "SUSC_15A_precision_event_patch_linkage_geojson.geojson",
    OUT / "SUSC_15A_precision_score_event_diagnostics.csv",
    OUT / "SUSC_15A_precision_score_event_diagnostics_by_region.csv",
    OUT / "SUSC_15A_precision_score_event_diagnostics_summary.json",
    OUT / "SUSC_15A_precision_observational_readiness.csv",
    OUT / "SUSC_15A_precision_observational_readiness.md",
    REPORT,
]


def _true(v) -> bool:
    return str(v or "").strip().lower() == "true"


def _false(v) -> bool:
    return str(v or "").strip().lower() == "false"


def _governance(rows, label):
    errors = []
    for idx, row in enumerate(rows, start=2):
        if "allowed_for_training" in row and not _false(row.get("allowed_for_training")):
            errors.append(f"{label}:{idx}: allowed_for_training != false")
        if "can_be_ground_truth" in row and not _false(row.get("can_be_ground_truth")):
            errors.append(f"{label}:{idx}: can_be_ground_truth != false")
        if "can_be_used_as_ground_truth" in row and not _false(row.get("can_be_used_as_ground_truth")):
            errors.append(f"{label}:{idx}: can_be_used_as_ground_truth != false")
        if "review_only" in row and not _true(row.get("review_only")):
            errors.append(f"{label}:{idx}: review_only != true")
    return errors


def _has_geom(row: dict) -> bool:
    return bool((row.get("lat") and row.get("lon")) or row.get("bbox") or row.get("wkt") or row.get("geojson_ref"))


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Precision Coordinate Rescue Validation")
    print("=" * 60)
    errors: list[str] = []
    print("\n[1/12] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
        elif path.is_file() and path.stat().st_size == 0 and path.suffix != ".csv":
            errors.append(f"empty: {path.relative_to(ROOT)}")
    if not errors:
        print("  OK")
    geocoded = read_csv(DAT / "susc_15a_precision_geocoded_occurrences_v1.csv")
    catalog = read_csv(DAT / "susc_15a_calibration_eligible_event_catalog_v1.csv")
    linkage = read_csv(DAT / "susc_15a_precision_event_patch_linkage_v1.csv")
    tables = {
        "geocoded": geocoded,
        "catalog": catalog,
        "linkage": linkage,
        "downloads": read_csv(MAN / "susc_15a_precision_reference_download_manifest_v1.csv"),
        "footprints": read_csv(DAT / "susc_15a_event_footprint_candidates_v1.csv"),
        "features": read_csv(DAT / "susc_15a_precision_reference_features_v1.csv"),
    }
    print("[2/12] Governance flags are closed...")
    for label, rows in tables.items():
        errors.extend(_governance(rows, label))
    for path in (MAN / "susc_15a_calibration_eligible_event_catalog_manifest_v1.json",
                 OUT / "SUSC_15A_precision_event_patch_linkage_limitations.json",
                 OUT / "SUSC_15A_precision_event_patch_linkage_geojson.geojson",
                 OUT / "SUSC_15A_precision_score_event_diagnostics_summary.json"):
        obj = read_json(path)
        if obj.get("allowed_for_training") is not False:
            errors.append(f"{path.name}: allowed_for_training != false")
        if obj.get("can_be_ground_truth") is not False:
            errors.append(f"{path.name}: can_be_ground_truth != false")
        if obj.get("review_only") is not True:
            errors.append(f"{path.name}: review_only != true")
    if not [e for e in errors if "!=" in e]:
        print("  OK")
    print("[3/12] Bairro-only and weak street candidates excluded...")
    for idx, row in enumerate(geocoded, start=2):
        if row.get("precision_tier") in {"T5_official_street_segment_candidate", "T6_neighborhood_context_only", "T7_rejected_or_non_flood"}:
            if row.get("calibration_eligible") == "true":
                errors.append(f"geocoded:{idx}: weak/context tier marked calibration eligible")
    if not [e for e in errors if "weak/context" in e]:
        print("  OK")
    print("[4/12] T0-T3 eligible rows have geometry...")
    for idx, row in enumerate(geocoded, start=2):
        if row.get("precision_tier") in {"T0_official_event_polygon", "T1_official_event_point", "T2_official_address_point_or_parcel", "T3_official_intersection_point"}:
            if row.get("calibration_eligible") == "true" and not _has_geom(row):
                errors.append(f"geocoded:{idx}: eligible T0-T3 missing geometry")
    if not [e for e in errors if "T0-T3" in e]:
        print("  OK")
    print("[5/12] T4 uncertainty <= 100 when eligible...")
    for idx, row in enumerate(geocoded, start=2):
        if row.get("precision_tier") == "T4_official_house_number_linear_reference" and row.get("calibration_eligible") == "true":
            try:
                if float(row.get("uncertainty_m", "9999")) > 100:
                    errors.append(f"geocoded:{idx}: eligible T4 uncertainty > 100")
            except ValueError:
                errors.append(f"geocoded:{idx}: eligible T4 uncertainty invalid")
    if not [e for e in errors if "T4" in e]:
        print("  OK")
    print("[6/12] Calibration catalog contains only eligible tiers...")
    allowed = {"T0_official_event_polygon", "T1_official_event_point", "T2_official_address_point_or_parcel", "T3_official_intersection_point", "T4_official_house_number_linear_reference"}
    for idx, row in enumerate(catalog, start=2):
        if row.get("precision_tier") not in allowed or row.get("calibration_eligible") != "true":
            errors.append(f"catalog:{idx}: non-eligible row included")
    if not [e for e in errors if "catalog" in e]:
        print("  OK")
    print("[7/12] Observational evaluation requires calibration eligibility...")
    for idx, row in enumerate(linkage, start=2):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            if row.get("spatial_relation") in {"insufficient_for_patch_link", "ambiguous_multiple_patches"}:
                errors.append(f"linkage:{idx}: invalid relation allowed")
    if not [e for e in errors if "linkage" in e]:
        print("  OK")
    print("[8/12] Score v7 absent...")
    if (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists():
        errors.append("score v7 artifact exists")
    if read_json(OUT / "SUSC_15A_precision_score_event_diagnostics_summary.json").get("score_v7_created") is not False:
        errors.append("diag summary score_v7_created != false")
    if not [e for e in errors if "score v7" in e]:
        print("  OK")
    print("[9/12] SUSC-03 matrix intact...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    else:
        print("  OK")
    print("[10/12] No persisted model artifacts...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")
    print("[11/12] Report contains mandatory statement...")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    else:
        print("  OK")
    print("[12/12] Raw directory is gitignored...")
    if not (DAT / "precision_references_susc15a" / ".gitignore").exists():
        errors.append("raw precision directory missing .gitignore")
    else:
        print("  OK")
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("PASSED: All SUSC-15A validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
