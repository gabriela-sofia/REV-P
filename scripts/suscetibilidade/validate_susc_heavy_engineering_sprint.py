"""
SUSC Heavy Engineering Sprint Validation (master)

Confirms the sprint deliverables (hardening modules, SUSC-09A, 09B, 10A, master
report + manifest) exist, governance holds globally, the SUSC-03 matrix is
unchanged, and no model artifact was persisted.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_HEAVY_ENGINEERING_SPRINT_09_10_REPORT.md"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_heavy_engineering_sprint_manifest_v1.json"

HARDENING = ["susc_io.py", "susc_governance.py", "susc_aliases.py", "susc_geometry.py",
             "susc_downloads.py", "susc_common.py"]
KEY_ARTIFACTS = [
    ROOT / "schemas" / "suscetibilidade" / "susc_09a_human_review_workspace_schema_v1.json",
    ROOT / "schemas" / "suscetibilidade" / "susc_09b_external_vector_acquisition_schema_v1.json",
    ROOT / "schemas" / "suscetibilidade" / "susc_10a_score_v6_candidate_schema_v1.json",
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_human_review_workspace_report.md",
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_acquisition_report.md",
    ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv",
]
MANDATORY = "Esta sprint avançou a infraestrutura operacional do REV-P"


def main() -> int:
    print("=" * 60)
    print("SUSC Heavy Engineering Sprint Validation (master)")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/5] Hardening modules exist...")
    for m in HARDENING:
        if not (HERE / m).exists():
            errors.append(f"missing module: {m}")
    if not [e for e in errors if "module" in e]:
        print(f"  OK: {len(HARDENING)} modules")

    print("[2/5] Key sprint artifacts exist (09A/09B/10A)...")
    for p in KEY_ARTIFACTS:
        if not p.exists():
            errors.append(f"missing artifact: {p.relative_to(ROOT)}")
    if not [e for e in errors if "artifact" in e]:
        print(f"  OK: {len(KEY_ARTIFACTS)} artifacts")

    print("[3/5] Master report + manifest exist + mandatory statement...")
    if not REPORT.exists():
        errors.append("master report missing")
    elif MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("master report missing mandatory statement")
    if not MANIFEST.exists():
        errors.append("master manifest missing")
    if not [e for e in errors if "master" in e]:
        print("  OK")

    print("[4/5] SUSC-03 matrix unchanged...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix SHA256 changed")
    else:
        print("  OK")

    print("[5/5] No model artifact persisted...")
    models = find_model_artifacts()
    if models:
        errors.append(f"model artifact(s) found: {models}")
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All sprint master validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
