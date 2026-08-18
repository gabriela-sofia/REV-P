"""SUSC-16A observed footprint rescue validator."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_susc_16a_local_project_event_geometry_sources as s16a  # noqa: E402
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402

REQUIRED = [
    s16a.LOCAL_AUDIT,
    s16a.LOCAL_REPORT,
    s16a.LOCAL_CANDIDATES,
    s16a.LOCAL_PARSE_AUDIT,
    s16a.EXTERNAL_DISCOVERY,
    s16a.EXTERNAL_MANIFEST,
    s16a.EXTERNAL_DIR / ".gitignore",
    s16a.EXTERNAL_DIR / "README.md",
    s16a.WINDOW_PLAN,
    s16a.WINDOW_REPORT,
    s16a.S1_TARGETS,
    s16a.S1_TARGETS_REPORT,
    s16a.GEE_STUB,
    s16a.STAC_STUB,
    s16a.SAR_CANDIDATES,
    s16a.SAR_REPORT,
    s16a.FOOTPRINT_CATALOG,
    s16a.FOOTPRINT_MANIFEST,
    s16a.FOOTPRINT_AUDIT,
    s16a.LINKAGE,
    s16a.LINKAGE_SUMMARY,
    s16a.LINKAGE_LIMITS,
    s16a.LINKAGE_GEOJSON,
    s16a.SCORE_DIAG,
    s16a.SCORE_DIAG_REGION,
    s16a.SCORE_SUMMARY,
    s16a.READINESS_MD,
    s16a.FINAL_REPORT,
]


def _true(value: str) -> bool:
    return str(value).lower() == "true"


def main() -> int:
    print("=" * 60)
    print("SUSC-16A Observed Footprint Rescue Validation")
    print("=" * 60)
    errors: list[str] = []
    print("[1/12] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(s16a.ROOT)}")
        elif path.is_file() and path.stat().st_size == 0 and path.suffix != ".csv":
            errors.append(f"empty: {path.relative_to(s16a.ROOT)}")
    if not errors:
        print("  OK")
    catalog = s16a.read_csv(s16a.FOOTPRINT_CATALOG)
    links = s16a.read_csv(s16a.LINKAGE)
    print("[2/12] Governance flags are closed...")
    for label, rows in {"catalog": catalog, "linkage": links}.items():
        for idx, row in enumerate(rows, start=2):
            for key in ("can_be_ground_truth", "allowed_for_training"):
                if row.get(key) != "false":
                    errors.append(f"{label}:{idx}: {key} != false")
            if "can_be_used_as_ground_truth" in row and row.get("can_be_used_as_ground_truth") != "false":
                errors.append(f"{label}:{idx}: can_be_used_as_ground_truth != false")
            if row.get("review_only") != "true":
                errors.append(f"{label}:{idx}: review_only != true")
    if not [err for err in errors if "!=" in err]:
        print("  OK")
    print("[3/12] Bairro-only is not eligible footprint...")
    for idx, row in enumerate(catalog, start=2):
        text = " ".join(str(row.get(k, "")) for k in ("footprint_type", "limitations", "method")).lower()
        if "bairro" in text and _true(row.get("eligible_for_patch_observational_evaluation", "")):
            errors.append(f"catalog:{idx}: bairro-only eligible")
    if not [err for err in errors if "bairro" in err]:
        print("  OK")
    print("[4/12] Risk and susceptibility context are not observed events...")
    for idx, row in enumerate(catalog, start=2):
        if row.get("footprint_type") in {"official_risk_context", "official_susceptibility_reference"}:
            if _true(row.get("eligible_for_patch_observational_evaluation", "")):
                errors.append(f"catalog:{idx}: context marked eligible")
    if not [err for err in errors if "context marked eligible" in err]:
        print("  OK")
    print("[5/12] Observational links require eligible footprints...")
    eligible_ids = {row.get("footprint_id") for row in catalog if _true(row.get("eligible_for_patch_observational_evaluation", ""))}
    for idx, row in enumerate(links, start=2):
        if _true(row.get("eligible_for_observational_evaluation", "")) and row.get("footprint_id") not in eligible_ids:
            errors.append(f"linkage:{idx}: eligible link without eligible footprint")
    if not [err for err in errors if "eligible link" in err]:
        print("  OK")
    print("[6/12] Score v7 absent...")
    summary = s16a.read_csv(s16a.SCORE_DIAG)
    if (s16a.DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists():
        errors.append("score v7 artifact exists")
    if any(row.get("metric") == "score_v7_created" and row.get("value") not in {"False", "false"} for row in summary):
        errors.append("score_v7_created not false")
    if not [err for err in errors if "score v7" in err or "score_v7" in err]:
        print("  OK")
    print("[7/12] SUSC-03 matrix intact...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    else:
        print("  OK")
    print("[8/12] No persisted model artifacts...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")
    print("[9/12] Mandatory final report statement present...")
    text = s16a.FINAL_REPORT.read_text(encoding="utf-8") if s16a.FINAL_REPORT.exists() else ""
    if s16a.MANDATORY not in text:
        errors.append("final report missing mandatory statement")
    else:
        print("  OK")
    print("[10/12] Sentinel targets stay external execution...")
    for idx, row in enumerate(s16a.read_csv(s16a.S1_TARGETS), start=2):
        if row.get("requires_external_execution") != "true" or row.get("review_only") != "true":
            errors.append(f"sentinel target:{idx}: execution/review flags invalid")
    if not [err for err in errors if "sentinel target" in err]:
        print("  OK")
    print("[11/12] External raw directory is gitignore-protected...")
    if not (s16a.EXTERNAL_DIR / ".gitignore").exists():
        errors.append("external raw dir missing .gitignore")
    else:
        print("  OK")
    print("[12/12] Linkage limitations exist...")
    limits = s16a.LINKAGE_LIMITS.read_text(encoding="utf-8") if s16a.LINKAGE_LIMITS.exists() else ""
    if "Bairro-only" not in limits:
        errors.append("linkage limitations missing bairro-only statement")
    else:
        print("  OK")
    print("=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("PASSED: All SUSC-16A validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
