"""SUSC-10D DINO Score Diagnostics Validation."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_json  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
DISCOVERY = OUT / "SUSC_10D_dino_embedding_discovery.csv"
DISCOVERY_SUMMARY = OUT / "SUSC_10D_dino_embedding_discovery_summary.json"
ALIGN = OUT / "SUSC_10D_dino_score_alignment.csv"
REGION = OUT / "SUSC_10D_dino_region_diagnostics.csv"
OUTLIER = OUT / "SUSC_10D_dino_outlier_candidates.csv"
SUMMARY = OUT / "SUSC_10D_dino_score_diagnostics_summary.json"
REPORT = OUT / "SUSC_10D_dino_score_diagnostics_report.md"
MANDATORY = "Ele não treina classificador, não cria ground truth"


def main() -> int:
    print("=" * 60)
    print("SUSC-10D DINO Score Diagnostics Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/5] Discovery + diagnostics artifacts exist...")
    for p in (DISCOVERY, DISCOVERY_SUMMARY, ALIGN, REGION, OUTLIER, SUMMARY, REPORT):
        if not p.exists():
            errors.append(f"missing: {p.name}")
    if not errors:
        print("  OK: 7 artifacts (absence path still emits headed tables)")

    summ = read_json(SUMMARY) if SUMMARY.exists() else {}

    print("[2/5] DINO usage is diagnostic-only, not classifier...")
    if summ.get("dino_usage") != "representational_diagnostic_only_never_classifier":
        errors.append("dino_usage not diagnostic-only")
    else:
        print("  OK")

    print("[3/5] Governance: no GT / training / model...")
    if summ.get("can_be_used_as_ground_truth") is not False or summ.get("allowed_for_training") is not False or summ.get("trained_model") is not False or summ.get("model_persisted") is not False:
        errors.append("summary governance flags not False")
    if find_model_artifacts():
        errors.append("model artifact found")
    if not [e for e in errors if "governance" in e or "model" in e]:
        print("  OK")

    print("[4/5] Absence/presence handled honestly...")
    status = summ.get("dino_status", "")
    if not status:
        errors.append("dino_status missing")
    elif status.startswith("ABSENT") and summ.get("loadable_per_patch_vectors_available") is not False:
        errors.append("absence status inconsistent with availability flag")
    else:
        print(f"  OK: dino_status={status}")

    print("[5/5] Report mandatory statement; matrix unchanged...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    if not matrix_sha256_ok():
        errors.append("matrix SHA256 changed")
    if not [e for e in errors if "mandatory" in e or "matrix" in e]:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-10D validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
