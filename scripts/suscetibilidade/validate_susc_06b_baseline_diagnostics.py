"""
SUSC-06B Baseline Diagnostics Validation

Validates the SUSC-06B diagnostic deliverables for structural integrity and
governance compliance.

Checks (fail-closed):
  - schema exists; all diagnostic CSV/JSON exist; report exists;
  - circularity risk is recorded;
  - high R2 is NOT described as predictive power;
  - all baseline divergences appear in the direction audit;
  - no divergent feature is auto-advanced to score v6 without review;
  - allowed_for_training=false; can_be_used_as_ground_truth=false; review_only=true;
  - no ground truth was created; no model artifact was persisted;
  - report contains the mandatory statement.

This script does NOT fit a model and does NOT create ground truth.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_06b_baseline_diagnostics_schema_v1.json"
OUT = ROOT / "outputs_public" / "suscetibilidade"
CIRC = OUT / "SUSC_06B_baseline_circularity_report.csv"
DIR_AUDIT = OUT / "SUSC_06B_partial_effect_direction_audit.csv"
REGION_DIAG = OUT / "SUSC_06B_region_stability_diagnostics.csv"
ACTION = OUT / "SUSC_06B_feature_action_matrix.csv"
OVERLAY = OUT / "SUSC_06B_event_overlay_readiness_matrix.csv"
SUMMARY = OUT / "SUSC_06B_diagnostic_summary.json"
REPORT = OUT / "SUSC_06B_proxy_baseline_diagnostics_report.md"
PARTIAL_06A = OUT / "SUSC_06_proxy_baseline_partial_effects.csv"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
MANDATORY = "não valida ocorrência real de enchente"


def load_csv(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    print("=" * 60)
    print("SUSC-06B Baseline Diagnostics Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/11] Schema + all diagnostic artifacts exist...")
    required = [SCHEMA, CIRC, DIR_AUDIT, REGION_DIAG, ACTION, OVERLAY, SUMMARY, REPORT]
    for p in required:
        if not p.exists():
            errors.append(f"missing artifact: {p.relative_to(ROOT)}")
    if not errors:
        print(f"  OK: {len(required)} artifacts present")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    circ = load_csv(CIRC) if CIRC.exists() else []
    dir_audit = load_csv(DIR_AUDIT) if DIR_AUDIT.exists() else []
    action = load_csv(ACTION) if ACTION.exists() else []

    print("[2/11] Circularity risk recorded...")
    risk = summary.get("circularity_risk") or (circ[0].get("circularity_risk") if circ else None)
    if not risk:
        errors.append("circularity_risk not recorded")
    else:
        print(f"  OK: circularity_risk={risk}")

    print("[3/11] R2 framed as recoverability, not predictive power...")
    if summary.get("r2_interpretation") != "heuristic_recoverability":
        errors.append("r2_interpretation is not heuristic_recoverability")
    if summary.get("predictive_power_claim") is not False:
        errors.append("predictive_power_claim is not False")
    if circ and circ[0].get("predictive_power_claim", "").lower() != "false":
        errors.append("circularity report does not deny predictive power")
    if not [e for e in errors if "predictive" in e or "r2_interpretation" in e]:
        print("  OK")

    print("[4/11] All baseline divergences appear in the direction audit...")
    # baseline divergences = partial effects that did not agree
    base_div = set()
    if PARTIAL_06A.exists():
        for r in load_csv(PARTIAL_06A):
            if r.get("agrees_with_expected") != "true":
                base_div.add(r["feature"])
    audited = {r["feature"] for r in dir_audit}
    missing = base_div - audited
    if missing:
        errors.append(f"baseline divergences absent from audit: {missing}")
    else:
        print(f"  OK: {len(base_div)} baseline divergence(s) all audited")

    print("[5/11] No divergent feature auto-advances to score v6 without review...")
    flagged = {
        r["feature"] for r in dir_audit
        if r.get("partial_effect_class") in {"direction_diverges", "near_zero_effect"}
    }
    bad = []
    for r in action:
        if r["feature"] in flagged and r.get("feature_action") == "advance_to_score_v6_candidate":
            bad.append(r["feature"])
    if bad:
        errors.append(f"divergent features auto-advanced to score v6: {bad}")
    else:
        print(f"  OK: {len(flagged)} flagged feature(s) held for manual_review_required")

    print("[6/11] Governance flags in summary/schema...")
    for obj, label in ((summary, "summary"), ):
        if obj.get("allowed_for_training") is not False:
            errors.append(f"{label} allowed_for_training != false")
        if obj.get("can_be_used_as_ground_truth") is not False:
            errors.append(f"{label} can_be_used_as_ground_truth != false")
        if obj.get("review_only") is not True:
            errors.append(f"{label} review_only != true")
    if SCHEMA.exists():
        sch = json.loads(SCHEMA.read_text(encoding="utf-8"))
        if sch.get("allowed_for_training") is not False or sch.get("can_be_used_as_ground_truth") is not False or sch.get("review_only") is not True:
            errors.append("schema governance flags incorrect")
    if not [e for e in errors if "allowed_for_training" in e or "ground_truth" in e or "review_only" in e or "governance" in e]:
        print("  OK")

    print("[7/11] No ground truth created...")
    # diagnostic must not assert any ground truth target / event validation
    if summary.get("scientific_status") != "review_only_proxy_diagnostic_not_event_validation":
        errors.append("scientific_status not review_only_proxy_diagnostic_not_event_validation")
    else:
        print("  OK")

    print("[8/11] No model artifact persisted...")
    models = []
    for base in (OUT, ROOT / "scripts" / "suscetibilidade", ROOT / "datasets" / "suscetibilidade"):
        if base.exists():
            models += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    if models:
        errors.append(f"model artifact(s) found: {models}")
    else:
        print("  OK")

    print("[9/11] Report contains mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[10/11] Overlay readiness gaps recorded for SUSC-07...")
    if OVERLAY.exists():
        ov = load_csv(OVERLAY)
        gaps = [r["required_field"] for r in ov if r.get("status") in {"missing", "weak"}]
        if not gaps:
            errors.append("no overlay gaps recorded (documentary evidence expected as gap)")
        else:
            print(f"  OK: gaps={gaps}")

    print("[11/11] Result tagged review_only/proxy_based/not_event_validation...")
    tags = set(summary.get("result_tags", []))
    if not {"review_only", "proxy_based", "not_event_validation"}.issubset(tags):
        errors.append(f"result_tags incomplete: {sorted(tags)}")
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-06B validations passed")
    print("\nReview-only proxy diagnostic. No real-event validation. No ground truth.")
    print("No training unlocked. No model persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
