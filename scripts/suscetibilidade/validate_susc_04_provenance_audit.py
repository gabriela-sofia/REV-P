"""
SUSC-04 Provenance Audit Validation

Validates the SUSC-04 deliverables (scientific direction vocabulary, provenance
audit CSV, score v6 decision matrix, public report) for structural integrity and
governance compliance.

Checks (fail-closed):
  - direction JSON exists and is valid;
  - audit CSV exists;
  - every critical feature appears in the audit CSV;
  - every audited feature has can_be_ground_truth=false;
  - every audited feature has allowed_for_training=false;
  - every audited feature has review_only=true;
  - no unresolved feature is marked safe_for_score_v6=true;
  - no feature with direction 'ambiguous'/'non_monotonic' has weight_class 'high';
  - decision matrix exists;
  - SUSC-04 report exists;
  - score v6 has NOT been computed (no score_v6 artifact present).

This script does NOT alter any file, does NOT unlock training, and does NOT
create ground truth.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

DIRECTION_PATH = (
    ROOT / "schemas" / "suscetibilidade" / "susc_feature_scientific_direction_v1.json"
)
AUDIT_CSV = ROOT / "manifests" / "suscetibilidade" / "susc_feature_provenance_audit_v1.csv"
DECISION_CSV = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_score_v6_feature_decision_matrix.csv"
)
REPORT_PATH = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_feature_provenance_audit_report.md"
)
SUSC_DATASETS = ROOT / "datasets" / "suscetibilidade"

# Critical features that must be present in the audit (real columns + requested
# names that must be explicitly accounted for, even if absent from SUSC-03).
CRITICAL_FEATURES = [
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "hand_mean", "twi_mean", "tpi_250m_mean", "curvature_laplacian_mean",
    "distance_to_water_mean", "water_occurrence_patch",
    "flow_acc_log_mean", "flow_acc_log_p75",
    "chirps_3d_mm", "chirps_7d_mm", "chirps_30d_mm",
    "rain_persistence_index",
    "s1_vv_mean_clean", "s1_vh_mean_clean", "s1_vv_minus_vh_mean_clean",
    "ndvi_mean", "mndwi_mean", "ndbi_mean",
    "urban_prop", "vegetation_prop",
    "urban_water_interaction", "urban_drainage_interaction",
    # requested names that must be explicitly resolved/marked:
    "chirps_3d_to_30d_ratio", "chirps_7d_to_30d_ratio", "runoff_score",
]

ALLOWED_DIRECTIONS = {
    "higher_increases", "lower_increases", "higher_decreases", "lower_decreases",
    "non_monotonic", "ambiguous", "not_scored",
}


def main() -> int:
    print("=" * 60)
    print("SUSC-04 Provenance Audit Validation")
    print("=" * 60)
    errors: list[str] = []

    # [1] direction JSON
    print("\n[1/10] Direction JSON exists and is valid...")
    if not DIRECTION_PATH.exists():
        errors.append(f"direction JSON missing: {DIRECTION_PATH}")
        direction = {"features": []}
    else:
        with DIRECTION_PATH.open("r", encoding="utf-8") as f:
            direction = json.load(f)
        for feat in direction.get("features", []):
            d = feat.get("expected_direction_for_flood_susceptibility")
            if d not in ALLOWED_DIRECTIONS:
                errors.append(f"feature '{feat.get('feature_name')}' invalid direction '{d}'")
        print(f"  OK: {len(direction.get('features', []))} features in direction vocabulary")

    # [2] audit CSV
    print("[2/10] Audit CSV exists...")
    if not AUDIT_CSV.exists():
        print(f"  STOP: audit CSV missing: {AUDIT_CSV}")
        return 1
    with AUDIT_CSV.open("r", encoding="utf-8", newline="") as f:
        audit = list(csv.DictReader(f))
    audit_by_name = {r["feature_name"]: r for r in audit}
    print(f"  OK: {len(audit)} audited features")

    # [3] critical features present
    print("[3/10] All critical features present in audit...")
    missing = [f for f in CRITICAL_FEATURES if f not in audit_by_name]
    if missing:
        errors.append(f"critical features absent from audit: {missing}")
    else:
        print(f"  OK: {len(CRITICAL_FEATURES)} critical features present")

    # [4-6] governance invariants
    print("[4/10] can_be_ground_truth=false for all...")
    gt = [r["feature_name"] for r in audit if r["can_be_ground_truth"] != "false"]
    if gt:
        errors.append(f"can_be_ground_truth != false: {gt}")
    else:
        print("  OK")
    print("[5/10] allowed_for_training=false for all...")
    tr = [r["feature_name"] for r in audit if r["allowed_for_training"] != "false"]
    if tr:
        errors.append(f"allowed_for_training != false: {tr}")
    else:
        print("  OK")
    print("[6/10] review_only=true for all...")
    ro = [r["feature_name"] for r in audit if r["review_only"] != "true"]
    if ro:
        errors.append(f"review_only != true: {ro}")
    else:
        print("  OK")

    # [7] unresolved not safe for v6
    print("[7/10] No unresolved feature is safe_for_score_v6=true...")
    bad = [
        r["feature_name"] for r in audit
        if r["provenance_status"] == "unresolved" and r["safe_for_score_v6"] == "true"
    ]
    if bad:
        errors.append(f"unresolved but safe_for_score_v6: {bad}")
    else:
        print("  OK")

    # [8] decision matrix + ambiguous-high invariant
    print("[8/10] Decision matrix exists; no ambiguous direction with weight high...")
    if not DECISION_CSV.exists():
        errors.append(f"decision matrix missing: {DECISION_CSV}")
    else:
        with DECISION_CSV.open("r", encoding="utf-8", newline="") as f:
            decision = list(csv.DictReader(f))
        amb_high = [
            r["feature_name"] for r in decision
            if r["direction"] in {"ambiguous", "non_monotonic"} and r["weight_class"] == "high"
        ]
        if amb_high:
            errors.append(f"ambiguous direction with weight high: {amb_high}")
        else:
            print(f"  OK: {len(decision)} decision rows, no ambiguous-high")

    # [9] report exists
    print("[9/10] SUSC-04 report exists...")
    if not REPORT_PATH.exists():
        errors.append(f"report missing: {REPORT_PATH}")
    else:
        print("  OK")

    # [10] no premature/final score v6 at the SUSC-04 stage.
    # The SUSC-10A *candidate* score (review-only, deterministic) is an allowed
    # downstream deliverable and is explicitly excluded from this guard.
    print("[10/10] no premature score v6 (candidate allowed)...")
    v6_artifacts = []
    if SUSC_DATASETS.exists():
        v6_artifacts = [p.name for p in SUSC_DATASETS.glob("*score*v6*")
                        if "candidate" not in p.name.lower()]
    if v6_artifacts:
        errors.append(f"unexpected non-candidate score v6 dataset artifact(s): {v6_artifacts}")
    else:
        print("  OK: no premature score v6 (SUSC-10A candidate excluded)")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-04 validations passed")
    print("\nSUSC-04 audits provenance only. No ground truth. No training unlocked.")
    print("Susceptibility matrix != confirmed flood occurrence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
