"""SUSC-10B->11A Integrated Analysis Validation."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_json  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10B_11A_INTEGRATED_ANALYSIS_REPORT.md"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_10b_11a_integrated_manifest_v1.json"
MANDATORY = "Ela não cria ground truth, não valida ocorrência real por patch, não treina modelo supervisionado"

# the phase artifacts that must all exist
PHASE_ARTIFACTS = [
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10B_score_comparison_report.md",
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10C_score_sensitivity_report.md",
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10D_dino_score_diagnostics_report.md",
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11A_public_visual_package" / "README.md",
]


def main() -> int:
    print("=" * 60)
    print("SUSC-10B->11A Integrated Analysis Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/4] Report + manifest exist + phase artifacts...")
    if not REPORT.exists():
        errors.append("integrated report missing")
    if not MANIFEST.exists():
        errors.append("integrated manifest missing")
    for p in PHASE_ARTIFACTS:
        if not p.exists():
            errors.append(f"missing phase artifact: {p.name}")
    if not errors:
        print("  OK")

    print("[2/4] Report mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[3/4] Manifest governance...")
    if MANIFEST.exists():
        m = read_json(MANIFEST)
        if m.get("can_be_used_as_ground_truth") is not False or m.get("allowed_for_training") is not False or m.get("review_only") is not True or m.get("trained_model") is not False:
            errors.append("manifest governance flags incorrect")
        else:
            print("  OK")

    print("[4/4] SUSC-03 matrix unchanged; no model...")
    if not matrix_sha256_ok():
        errors.append("matrix SHA256 changed")
    if find_model_artifacts():
        errors.append("model artifact found")
    if not [e for e in errors if "matrix" in e or "model" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-10B->11A integrated validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
