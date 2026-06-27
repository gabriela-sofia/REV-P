"""
SUSC-06A Proxy Baseline Validation

Validates the SUSC-06A deliverables (schema, baseline dataset + manifest, target
policy, metrics/region/feature/partial outputs, report) for structural integrity
and governance compliance.

Checks (fail-closed):
  - baseline dataset exists; manifest exists; schema exists; target policy exists;
  - report exists; all output CSVs exist;
  - allowed_for_training=false; can_be_used_as_ground_truth=false; review_only=true;
  - NO persisted model artifact was created;
  - report contains the mandatory statement;
  - the chosen target is classified as a proxy (not ground truth);
  - no ground_truth column was used (ground_truth_targets empty);
  - features used have a card with an allowed status (ready_for_methodology /
    usable_with_caution).

This script does NOT fit a model, does NOT compute a score, and does NOT create
ground truth.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_06_proxy_baseline_schema_v1.json"
DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_v1.csv"
DS_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_manifest_v1.json"
)
OUT_DIR = ROOT / "outputs_public" / "suscetibilidade"
TARGET_POLICY = OUT_DIR / "SUSC_06_proxy_baseline_target_policy.json"
FEATURE_TABLE = OUT_DIR / "SUSC_06_proxy_baseline_feature_table.csv"
METRICS = OUT_DIR / "SUSC_06_proxy_baseline_metrics.csv"
REGION_METRICS = OUT_DIR / "SUSC_06_proxy_baseline_region_metrics.csv"
PARTIAL_EFFECTS = OUT_DIR / "SUSC_06_proxy_baseline_partial_effects.csv"
REPORT = OUT_DIR / "SUSC_06_proxy_gam_spgam_baseline_report.md"
CARDS_JSON = OUT_DIR / "feature_cards" / "susc_feature_cards_v1.json"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
ALLOWED_FEATURE_STATUS = {"ready_for_methodology", "usable_with_caution"}
PROXY_GROUPS = {"score", "heuristic_label", "proxy_v5"}
MANDATORY = "não valida ocorrência real de enchente"


def main() -> int:
    print("=" * 60)
    print("SUSC-06A Proxy Baseline Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/9] Required artifacts exist...")
    required = [SCHEMA, DATASET, DS_MANIFEST, TARGET_POLICY, REPORT,
                FEATURE_TABLE, METRICS, REGION_METRICS, PARTIAL_EFFECTS]
    for p in required:
        if not p.exists():
            errors.append(f"missing artifact: {p.relative_to(ROOT)}")
    if not errors:
        print(f"  OK: {len(required)} artifacts present")

    print("[2/9] Dataset manifest governance...")
    if DS_MANIFEST.exists():
        with DS_MANIFEST.open("r", encoding="utf-8") as f:
            man = json.load(f)
        if man.get("allowed_for_training") is not False:
            errors.append("manifest allowed_for_training != false")
        if man.get("can_be_used_as_ground_truth") is not False:
            errors.append("manifest can_be_used_as_ground_truth != false")
        if man.get("review_only") is not True:
            errors.append("manifest review_only != true")
        if man.get("ground_truth_targets"):
            errors.append(f"manifest ground_truth_targets not empty: {man['ground_truth_targets']}")
        else:
            print("  OK: training/GT off, ground_truth_targets empty")
    else:
        man = {}

    print("[3/9] Schema governance...")
    if SCHEMA.exists():
        with SCHEMA.open("r", encoding="utf-8") as f:
            sch = json.load(f)
        if sch.get("allowed_for_training") is not False or sch.get("can_be_used_as_ground_truth") is not False or sch.get("review_only") is not True:
            errors.append("schema governance flags incorrect")
        if sch.get("baseline_status") != "proxy_baseline_review_only":
            errors.append("schema baseline_status != proxy_baseline_review_only")
        else:
            print("  OK")

    print("[4/9] Target policy: proxy, not ground truth...")
    if TARGET_POLICY.exists():
        with TARGET_POLICY.open("r", encoding="utf-8") as f:
            pol = json.load(f)
        if pol.get("baseline_status") != "proxy_baseline_review_only":
            errors.append("target policy baseline_status incorrect")
        if pol.get("chosen_target") is not None:
            if pol.get("target_is_proxy") is not True:
                errors.append("chosen target not marked proxy")
            if pol.get("target_is_ground_truth") is not False:
                errors.append("chosen target marked ground truth")
            if pol.get("model_persisted") is not False:
                errors.append("target policy claims a persisted model")
            print(f"  OK: target '{pol.get('chosen_target')}' is proxy, not GT")
        else:
            print("  OK: no target (readiness-only), nothing fitted")
    else:
        pol = {}

    print("[5/9] No persisted model artifact...")
    models = []
    for base in (ROOT / "datasets" / "suscetibilidade",
                 ROOT / "scripts" / "suscetibilidade",
                 OUT_DIR):
        if base.exists():
            models += [str(p.relative_to(ROOT)) for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    if models:
        errors.append(f"persisted model artifact(s) found: {models}")
    else:
        print("  OK: no model file persisted")

    print("[6/9] Report contains mandatory statement...")
    if REPORT.exists():
        if MANDATORY not in REPORT.read_text(encoding="utf-8"):
            errors.append("report missing mandatory statement")
        else:
            print("  OK")

    print("[7/9] No ground_truth column used...")
    # dataset columns must not include any column flagged ground-truth-capable
    # (the SUSC chain marks everything can_be_used_as_ground_truth=false, so this
    # is a structural guard: ground_truth_targets must be empty and target is proxy)
    if man.get("ground_truth_targets"):
        errors.append("ground truth target used")
    else:
        print("  OK")

    print("[8/9] Features used have card with allowed status...")
    if CARDS_JSON.exists() and DS_MANIFEST.exists():
        with CARDS_JSON.open("r", encoding="utf-8") as f:
            cards = {c["feature_name"]: c for c in json.load(f)["cards"]}
        bad = []
        for feat in man.get("features_used", []):
            card = cards.get(feat)
            if not card or card["status"] not in ALLOWED_FEATURE_STATUS:
                bad.append(feat)
        if bad:
            errors.append(f"features without allowed-status card: {bad}")
        else:
            print(f"  OK: {len(man.get('features_used', []))} features carded with allowed status")

    print("[9/9] Metrics declared against proxy...")
    if METRICS.exists():
        with METRICS.open("r", encoding="utf-8", newline="") as f:
            mrows = list(csv.DictReader(f))
        if mrows and mrows[0].get("metric_basis") not in {"proxy_not_event"}:
            errors.append("metrics not declared against proxy")
        else:
            print("  OK: metric_basis=proxy_not_event")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-06A validations passed")
    print("\nProxy baseline review-only. No real-event validation. No ground truth.")
    print("No training unlocked. No model persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
