"""SUSC-14C official reference expansion validator."""

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

MANDATORY = (
    "O SUSC-14C expande a aquisicao de referencias oficiais e reprocessa ocorrencias de cheia para aumentar "
    "vinculos observacionais evento-patch. Todos os vinculos permanecem review-only, sem ground truth, sem "
    "treino supervisionado e sem score v7 automatico."
)

REQUIRED = [
    OUT / "SUSC_14C_14B_matching_failure_audit.csv",
    OUT / "SUSC_14C_14B_matching_failure_report.md",
    MAN / "susc_14c_additional_reference_download_manifest_v1.csv",
    DAT / "official_spatial_references_susc14c",
    DAT / "susc_14c_additional_official_reference_features_v1.csv",
    OUT / "SUSC_14C_additional_reference_parse_audit.csv",
    DAT / "susc_14c_official_spatial_reference_features_union_v1.csv",
    MAN / "susc_14c_reference_feature_union_manifest_v1.json",
    DAT / "susc_14c_matched_official_occurrences_v1.csv",
    OUT / "SUSC_14C_occurrence_reference_match_audit.csv",
    DAT / "susc_14c_resolved_ambiguous_occurrences_v1.csv",
    OUT / "SUSC_14C_ambiguous_resolution_audit.csv",
    DAT / "susc_14c_event_patch_linkage_v1.csv",
    OUT / "SUSC_14C_event_patch_linkage_summary.csv",
    OUT / "SUSC_14C_event_patch_linkage_limitations.json",
    OUT / "SUSC_14C_event_patch_linkage_geojson.geojson",
    OUT / "SUSC_14C_observational_readiness.csv",
    OUT / "SUSC_14C_observational_readiness.md",
    OUT / "SUSC_14C_score_v6_event_diagnostics.csv",
    OUT / "SUSC_14C_score_v6_event_diagnostics_summary.json",
    OUT / "SUSC_14C_official_reference_expansion_report.md",
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


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Official Reference Expansion Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/12] Required artifacts exist and are non-empty...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"empty: {path.relative_to(ROOT)}")
    if not errors:
        print("  OK")

    tables = {
        "downloads": read_csv(MAN / "susc_14c_additional_reference_download_manifest_v1.csv"),
        "features_add": read_csv(DAT / "susc_14c_additional_official_reference_features_v1.csv"),
        "features_union": read_csv(DAT / "susc_14c_official_spatial_reference_features_union_v1.csv"),
        "matched": read_csv(DAT / "susc_14c_matched_official_occurrences_v1.csv"),
        "resolved": read_csv(DAT / "susc_14c_resolved_ambiguous_occurrences_v1.csv"),
        "linkage": read_csv(DAT / "susc_14c_event_patch_linkage_v1.csv"),
        "readiness": read_csv(OUT / "SUSC_14C_observational_readiness.csv"),
        "diag": read_csv(OUT / "SUSC_14C_score_v6_event_diagnostics.csv"),
    }

    print("[2/12] Governance flags are closed...")
    for label, rows in tables.items():
        errors.extend(_governance(rows, label))
    for path in (MAN / "susc_14c_reference_feature_union_manifest_v1.json",
                 OUT / "SUSC_14C_event_patch_linkage_limitations.json",
                 OUT / "SUSC_14C_event_patch_linkage_geojson.geojson",
                 OUT / "SUSC_14C_score_v6_event_diagnostics_summary.json"):
        obj = read_json(path)
        if obj.get("allowed_for_training") is not False:
            errors.append(f"{path.name}: allowed_for_training != false")
        if obj.get("can_be_ground_truth") is not False:
            errors.append(f"{path.name}: can_be_ground_truth != false")
        if obj.get("review_only") is not True:
            errors.append(f"{path.name}: review_only != true")
    if not [e for e in errors if "!=" in e]:
        print("  OK")

    print("[3/12] Neighborhood-only never becomes patch-level...")
    for idx, row in enumerate(tables["linkage"], start=2):
        if row.get("spatial_relation") == "weak_neighborhood_context":
            if row.get("patch_id") != "REGION_LEVEL_NO_PATCH_RESOLUTION":
                errors.append(f"linkage:{idx}: neighborhood-only resolved to patch")
            if row.get("can_be_used_for_observational_evaluation") == "true":
                errors.append(f"linkage:{idx}: neighborhood-only allowed for evaluation")
    if not [e for e in errors if "neighborhood-only" in e]:
        print("  OK")

    print("[4/12] Unresolved ambiguous rows stay rejected...")
    for idx, row in enumerate(tables["resolved"], start=2):
        if row.get("georeferencing_status") == "blocked_ambiguous_address":
            if row.get("evidence_level") != "rejected_not_observed_event":
                errors.append(f"resolved:{idx}: unresolved ambiguous not rejected")
    for idx, row in enumerate(tables["linkage"], start=2):
        if row.get("georeferencing_status") == "blocked_ambiguous_address":
            if row.get("linkage_confidence") == "moderate":
                errors.append(f"linkage:{idx}: unresolved ambiguous moderate")
            if row.get("can_be_used_for_observational_evaluation") == "true":
                errors.append(f"linkage:{idx}: unresolved ambiguous allowed evaluation")
    if not [e for e in errors if "ambiguous" in e]:
        print("  OK")

    print("[5/12] Street segment never becomes strong...")
    for idx, row in enumerate(tables["linkage"], start=2):
        if row.get("matched_feature_type") == "street_segment" and row.get("linkage_confidence") == "strong":
            errors.append(f"linkage:{idx}: street segment marked strong")
    if not [e for e in errors if "street segment" in e]:
        print("  OK")

    print("[6/12] Observational evaluation requires moderate official match...")
    for idx, row in enumerate(tables["linkage"], start=2):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            if row.get("linkage_confidence") != "moderate":
                errors.append(f"linkage:{idx}: evaluation without moderate confidence")
            if row.get("georeferencing_status") not in {"official_address_point_match", "official_street_segment_match"}:
                errors.append(f"linkage:{idx}: evaluation without official georeferencing")
    if not [e for e in errors if "evaluation" in e]:
        print("  OK")

    print("[7/12] Score v7 absent...")
    if (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists():
        errors.append("score v7 artifact exists")
    if read_json(OUT / "SUSC_14C_score_v6_event_diagnostics_summary.json").get("score_v7_created") is not False:
        errors.append("diag summary score_v7_created != false")
    if not [e for e in errors if "score v7" in e]:
        print("  OK")

    print("[8/12] SUSC-03 matrix intact...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    else:
        print("  OK")

    print("[9/12] No persisted model artifacts...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")

    print("[10/12] Report contains mandatory statement...")
    report = (OUT / "SUSC_14C_official_reference_expansion_report.md").read_text(encoding="utf-8")
    if MANDATORY not in report:
        errors.append("report missing mandatory statement")
    else:
        print("  OK")

    print("[11/12] Raw directory is gitignored...")
    if not (DAT / "official_spatial_references_susc14c" / ".gitignore").exists():
        errors.append("raw directory missing .gitignore")
    else:
        print("  OK")

    print("[12/12] Readiness keys present...")
    readiness = {r["metric"]: r["value"] for r in tables["readiness"]}
    for key in ("ready_for_12a", "ready_for_12b", "ready_for_12c", "ready_for_score_v7"):
        if key not in readiness:
            errors.append(f"readiness missing {key}")
    if readiness.get("ready_for_score_v7") != "false":
        errors.append("ready_for_score_v7 must remain false")
    if not [e for e in errors if "readiness" in e or "ready_for_score_v7" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("PASSED: All SUSC-14C validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
