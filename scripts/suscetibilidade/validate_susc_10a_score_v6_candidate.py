"""
SUSC-10A Score v6 Candidate Validation
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, sha256_file  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_methodology_report.md"
CARDS = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_limitations.json"

FLAGGED_OUT = {"curvature_laplacian_mean", "rain_3d_7d_ratio", "flow_acc_log_p75", "water_occurrence_patch"}
FORBIDDEN_GROUPS = {"score", "heuristic_label", "proxy_v5"}
MANDATORY = "Ele não é modelo supervisionado, não usa ground truth"


def main() -> int:
    print("=" * 60)
    print("SUSC-10A Score v6 Candidate Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/11] Score, manifest, report exist...")
    for p in (SCORE, MANIFEST, REPORT, LIMITS):
        if not p.exists():
            errors.append(f"missing: {p.relative_to(ROOT)}")
    if not errors:
        print("  OK")

    rows = read_csv(SCORE) if SCORE.exists() else []
    manifest = read_json(MANIFEST) if MANIFEST.exists() else {}
    cards = {c["feature_name"]: c for c in read_json(CARDS)["cards"]} if CARDS.exists() else {}

    print("[2/11] Score in [0,1]...")
    bad = []
    for r in rows:
        try:
            v = float(r["susc_score_v6_candidate"])
            if v < 0.0 or v > 1.0:
                bad.append(r["patch_id"])
        except (ValueError, KeyError):
            bad.append(r.get("patch_id", "?"))
    if bad:
        errors.append(f"scores out of [0,1]: {bad[:5]}")
    else:
        print(f"  OK: {len(rows)} scores in [0,1]")

    print("[3/11] No proxy target column used as feature...")
    used = manifest.get("features_used", [])
    if manifest.get("uses_proxy_target") is not False:
        errors.append("manifest uses_proxy_target != False")
    # ensure no score_* / label_* / proxy_v5_* in features used
    bad_feat = [f for f in used if f.startswith(("score_", "label_", "proxy_v5_"))]
    if bad_feat:
        errors.append(f"forbidden feature used: {bad_feat}")
    if not [e for e in errors if "proxy_target" in e or "forbidden feature" in e]:
        print("  OK")

    print("[4/11] No label/score-v5 as feature (group check)...")
    bad_group = [f for f in used if cards.get(f, {}).get("feature_group") in FORBIDDEN_GROUPS]
    if bad_group:
        errors.append(f"features from forbidden groups: {bad_group}")
    elif manifest.get("uses_label_as_feature") is not False or manifest.get("uses_score_v5_as_feature") is not False:
        errors.append("manifest label/score-v5 feature flags not False")
    else:
        print("  OK")

    print("[5/11] Flagged features excluded from main score...")
    leaked = [f for f in used if f in FLAGGED_OUT]
    if leaked:
        errors.append(f"flagged features in score: {leaked}")
    else:
        print(f"  OK: {sorted(FLAGGED_OUT)} excluded")

    print("[6/11] No training / no model...")
    if manifest.get("allowed_for_training") is not False or manifest.get("trained_model") is not False or manifest.get("model_persisted") is not False:
        errors.append("manifest training/model flags not False")
    if find_model_artifacts():
        errors.append("model artifact found")
    if not [e for e in errors if "training/model" in e or "model artifact" in e]:
        print("  OK")

    print("[7/11] No ground truth...")
    if manifest.get("can_be_used_as_ground_truth") is not False:
        errors.append("manifest GT flag not False")
    gt_rows = [r["patch_id"] for r in rows if str(r.get("can_be_used_as_ground_truth", "false")).lower() != "false"]
    if gt_rows:
        errors.append(f"score rows marked GT: {gt_rows[:5]}")
    if not [e for e in errors if "GT" in e]:
        print("  OK")

    print("[8/11] Governance columns on score rows...")
    bad_gov = [r["patch_id"] for r in rows if r.get("allowed_for_training") != "false" or r.get("review_only") != "true"]
    if bad_gov:
        errors.append(f"score rows governance violation: {bad_gov[:5]}")
    else:
        print("  OK")

    print("[9/11] Manifest SHA256 matches score CSV...")
    if SCORE.exists() and manifest.get("sha256") != sha256_file(SCORE):
        errors.append("manifest sha256 mismatch")
    else:
        print("  OK")

    print("[10/11] Report contains mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[11/11] SUSC-03 matrix unchanged...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix SHA256 changed")
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-10A validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
