"""SUSC-10C Score Sensitivity Validation."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
SCEN = OUT / "SUSC_10C_score_sensitivity_scenarios.csv"
PATCH = OUT / "SUSC_10C_score_sensitivity_by_patch.csv"
REGION = OUT / "SUSC_10C_score_sensitivity_by_region.csv"
UNSTABLE = OUT / "SUSC_10C_score_sensitivity_top_unstable_patches.csv"
SUMMARY = OUT / "SUSC_10C_score_sensitivity_summary.json"
REPORT = OUT / "SUSC_10C_score_sensitivity_report.md"

REQUIRED_SCENARIOS = {"baseline_v6", "no_rainfall_trigger", "no_urban_spectral",
                      "no_topography_hydrology", "no_evidence_support", "equal_group_weights",
                      "hydrology_heavy", "urban_heavy", "rainfall_heavy",
                      "strict_high_threshold", "loose_high_threshold"}
MANDATORY = "A análise de sensibilidade não constitui treinamento, validação supervisionada"


def main() -> int:
    print("=" * 60)
    print("SUSC-10C Score Sensitivity Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/6] Outputs exist...")
    for p in (SCEN, PATCH, REGION, UNSTABLE, SUMMARY, REPORT):
        if not p.exists():
            errors.append(f"missing: {p.name}")
    if not errors:
        print("  OK")

    scen = read_csv(SCEN) if SCEN.exists() else []
    patch = read_csv(PATCH) if PATCH.exists() else []
    summ = read_json(SUMMARY) if SUMMARY.exists() else {}

    print("[2/6] All required scenarios present...")
    have = {r["scenario"] for r in scen}
    missing = REQUIRED_SCENARIOS - have
    if missing:
        errors.append(f"missing scenarios: {missing}")
    else:
        print(f"  OK: {len(have)} scenarios")

    print("[3/6] baseline_v6 has spearman 1.0 vs itself...")
    base = next((r for r in scen if r["scenario"] == "baseline_v6"), None)
    if not base or abs(float(base["spearman_vs_baseline"]) - 1.0) > 1e-6:
        errors.append("baseline_v6 spearman != 1.0")
    else:
        print("  OK")

    print("[4/6] by_patch has 300 rows; stability flags valid...")
    if len(patch) != 300:
        errors.append(f"by_patch rows {len(patch)} != 300")
    bad = [r["patch_id"] for r in patch if r.get("stability_flag") not in {"robust", "moderate", "unstable"}]
    if bad:
        errors.append(f"invalid stability_flag: {bad[:3]}")
    if not [e for e in errors if "by_patch" in e or "stability_flag" in e]:
        print("  OK")

    print("[5/6] Governance: no GT / training / model...")
    if summ.get("can_be_used_as_ground_truth") is not False or summ.get("allowed_for_training") is not False or summ.get("trained_model") is not False:
        errors.append("summary governance flags not False")
    if find_model_artifacts():
        errors.append("model artifact found")
    for r in patch:
        if r.get("review_only") != "true":
            errors.append("patch rows not review_only")
            break
    if not [e for e in errors if "governance" in e or "model" in e or "review_only" in e]:
        print("  OK")

    print("[6/6] Report mandatory statement; matrix unchanged...")
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
    print("PASSED: All SUSC-10C validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
