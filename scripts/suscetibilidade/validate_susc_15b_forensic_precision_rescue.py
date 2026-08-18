"""SUSC-15B forensic precision rescue validator."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402
from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402

REQUIRED = [
    c.HIDDEN_AUDIT, c.DISCOVERY_REPORT, c.DOWNLOAD_MANIFEST, c.PARSE_AUDIT,
    c.JOIN_AUDIT, c.CATALOG, c.LINKAGE, c.READINESS, c.REPORT,
]


def main() -> int:
    print("=" * 60)
    print("SUSC-15B Forensic Precision Rescue Validation")
    print("=" * 60)
    errors: list[str] = []
    print("[1/10] Required artifacts exist...")
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(c.ROOT)}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"empty: {path.relative_to(c.ROOT)}")
    if not errors:
        print("  OK")
    print("[2/10] Governance flags are closed...")
    for label, rows in {"catalog": c.read_csv(c.CATALOG), "linkage": c.read_csv(c.LINKAGE)}.items():
        for idx, row in enumerate(rows, start=2):
            for key in ("can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training"):
                if row.get(key) != "false":
                    errors.append(f"{label}:{idx}: {key} != false")
            if row.get("review_only") != "true":
                errors.append(f"{label}:{idx}: review_only != true")
    if not [err for err in errors if "!=" in err]:
        print("  OK")
    print("[3/10] Bairro-only and street segment candidates excluded...")
    for idx, row in enumerate(c.read_csv(c.CATALOG), start=2):
        if row.get("precision_tier") in c.WEAK_TIERS and row.get("calibration_eligible") == "true":
            errors.append(f"catalog:{idx}: weak/context tier marked calibration eligible")
    if not [err for err in errors if "weak/context" in err]:
        print("  OK")
    print("[4/10] Eligible rows, if any, are T0-T4 with geometry...")
    for idx, row in enumerate(c.read_csv(c.CATALOG), start=2):
        if row.get("calibration_eligible") == "true":
            if row.get("precision_tier") not in c.PRECISE_TIERS or not c.has_geometry(row):
                errors.append(f"catalog:{idx}: eligible row is not precise T0-T4 with geometry")
    if not [err for err in errors if "eligible row" in err]:
        print("  OK")
    print("[5/10] Observational links require eligible catalog rows...")
    eligible_ids = {row.get("event_id") for row in c.read_csv(c.CATALOG) if row.get("calibration_eligible") == "true"}
    for idx, row in enumerate(c.read_csv(c.LINKAGE), start=2):
        if row.get("can_be_used_for_observational_evaluation") == "true" and row.get("event_id") not in eligible_ids:
            errors.append(f"linkage:{idx}: observational link without eligible event")
    if not [err for err in errors if "observational link" in err]:
        print("  OK")
    print("[6/10] Score v7 absent...")
    if (c.DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists():
        errors.append("score v7 artifact exists")
    else:
        print("  OK")
    print("[7/10] SUSC-03 matrix intact...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    else:
        print("  OK")
    print("[8/10] No persisted model artifacts...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact found: {models[:5]}")
    else:
        print("  OK")
    print("[9/10] Mandatory report statement present...")
    text = c.REPORT.read_text(encoding="utf-8") if c.REPORT.exists() else ""
    if c.MANDATORY not in text:
        errors.append("report missing mandatory statement")
    else:
        print("  OK")
    print("[10/10] Deep-source blockers documented...")
    if not c.read_csv(c.DOWNLOAD_MANIFEST):
        errors.append("download manifest has no blocker rows")
    else:
        print("  OK")
    print("=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("PASSED: All SUSC-15B validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
