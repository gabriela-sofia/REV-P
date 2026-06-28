"""SUSC-10B Score Comparison Validation."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
ARTIFACTS = [
    OUT / "SUSC_10B_score_comparison_metrics.csv",
    OUT / "SUSC_10B_score_rank_shift_by_patch.csv",
    OUT / "SUSC_10B_score_class_agreement.csv",
    OUT / "SUSC_10B_score_region_diagnostics.csv",
    OUT / "SUSC_10B_high_score_spatial_evidence_cases.csv",
    OUT / "SUSC_10B_score_disagreement_cases.csv",
    OUT / "SUSC_10B_score_comparison_summary.json",
    OUT / "SUSC_10B_score_comparison_report.md",
]
REPORT = OUT / "SUSC_10B_score_comparison_report.md"
SUMMARY = OUT / "SUSC_10B_score_comparison_summary.json"
MANDATORY = "A concordância entre score v6, proxy v5, baseline ou evidência documental não constitui"


def main() -> int:
    print("=" * 60)
    print("SUSC-10B Score Comparison Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/8] Outputs exist...")
    for p in ARTIFACTS:
        if not p.exists():
            errors.append(f"missing: {p.name}")
    if not errors:
        print("  OK: 8 outputs")

    rank = read_csv(OUT / "SUSC_10B_score_rank_shift_by_patch.csv") if ARTIFACTS[1].exists() else []
    summ = read_json(SUMMARY) if SUMMARY.exists() else {}

    print("[2/8] patch_id aligned (rank shift has 300 rows)...")
    if len(rank) != 300:
        errors.append(f"rank shift rows {len(rank)} != 300")
    else:
        print("  OK: 300 patches aligned")

    print("[3/8] score v6 in [0,1] in rank table...")
    bad = [r["patch_id"] for r in rank if not (0.0 <= float(r["score_v6"]) <= 1.0)]
    if bad:
        errors.append(f"score v6 out of [0,1]: {bad[:3]}")
    else:
        print("  OK")

    print("[4/8] No ground truth / no training in rows...")
    for r in rank:
        if r.get("review_only") != "true":
            errors.append("rank rows not review_only")
            break
    if summ.get("can_be_used_as_ground_truth") is not False or summ.get("allowed_for_training") is not False:
        errors.append("summary governance flags not False")
    if not [e for e in errors if "review_only" in e or "governance" in e]:
        print("  OK")

    print("[5/8] proxy v5 NOT used as score feature (summary flag)...")
    if summ.get("uses_proxy_v5_as_score_feature") is not False:
        errors.append("summary uses_proxy_v5_as_score_feature != False")
    else:
        print("  OK")

    print("[6/8] Correlation not described as real validation (guard)...")
    if summ.get("interpretation_guard") != "correlation_is_not_real_event_validation":
        errors.append("missing correlation guard")
    else:
        print("  OK")

    print("[7/8] Report has mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[8/8] SUSC-03 matrix unchanged; no model...")
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
    print("PASSED: All SUSC-10B validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
