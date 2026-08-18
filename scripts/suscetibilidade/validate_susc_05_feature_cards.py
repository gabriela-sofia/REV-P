"""
SUSC-05 Feature Cards Validation

Validates the SUSC-05 deliverables (feature card schema, cards JSON, catalog MD,
summary CSV, methodology report, chain register/manifest) for structural integrity
and governance compliance.

Checks (fail-closed):
  - feature card schema exists;
  - cards JSON exists and is valid;
  - catalog MD exists; summary CSV exists; SUSC-05 report exists;
  - every obligatory feature has a card;
  - every card has all required fields;
  - can_be_ground_truth=false in all cards;
  - allowed_for_training=false in all cards;
  - review_only=true in all cards;
  - no 'unresolved' card is marked safe_for_score_v6=true;
  - no heuristic label/score/proxy card is ground truth;
  - score v6 has NOT been created (no score_v6 dataset);
  - no new model artifact was created.

This script does NOT compute a score, does NOT train a model, and does NOT
create ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CARD_SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_feature_card_schema_v1.json"
FC_DIR = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards"
CARDS_JSON = FC_DIR / "susc_feature_cards_v1.json"
CATALOG_MD = FC_DIR / "SUSC_05_feature_cards_catalog.md"
SUMMARY_CSV = FC_DIR / "SUSC_05_feature_cards_summary.csv"
REPORT_MD = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_05_feature_cards_methodology_report.md"
)
CHAIN_REGISTER = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_01_04_chain_closure_register.md"
)
CHAIN_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_chain_commits_manifest_v1.csv"
SUSC_DATASETS = ROOT / "datasets" / "suscetibilidade"
SUSC_SCRIPTS = ROOT / "scripts" / "suscetibilidade"

OBLIGATORY = [
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "hand_mean", "twi_mean", "tpi_250m_mean", "curvature_laplacian_mean",
    "distance_to_water_mean", "water_occurrence_patch",
    "flow_acc_log_mean", "flow_acc_log_p75",
    "chirps_3d_mm", "chirps_7d_mm", "chirps_30d_mm",
    "chirps_3d_to_30d_ratio", "chirps_7d_to_30d_ratio",
    "rain_persistence_index",
    "s1_vv_mean_clean", "s1_vh_mean_clean", "s1_vv_minus_vh_mean_clean",
    "ndvi_mean", "mndwi_mean", "ndbi_mean",
    "urban_prop", "vegetation_prop", "runoff_score",
]

COMMIT_HASHES = ["4b49eb5", "7986da4", "3c49608"]

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav"}


def main() -> int:
    print("=" * 60)
    print("SUSC-05 Feature Cards Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/13] Feature card schema exists...")
    if not CARD_SCHEMA.exists():
        errors.append(f"card schema missing: {CARD_SCHEMA}")
        required_fields = []
    else:
        with CARD_SCHEMA.open("r", encoding="utf-8") as f:
            cs = json.load(f)
        required_fields = cs["required_fields"]
        print(f"  OK: {len(required_fields)} required fields defined")

    print("[2/13] Cards JSON exists and valid...")
    if not CARDS_JSON.exists():
        print(f"  STOP: cards JSON missing: {CARDS_JSON}")
        return 1
    with CARDS_JSON.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    cards = doc["cards"]
    by_name = {c["feature_name"]: c for c in cards}
    print(f"  OK: {len(cards)} cards")

    print("[3/13] Catalog MD exists...")
    if not CATALOG_MD.exists():
        errors.append("catalog MD missing")
    else:
        print("  OK")
    print("[4/13] Summary CSV exists...")
    if not SUMMARY_CSV.exists():
        errors.append("summary CSV missing")
    else:
        print("  OK")
    print("[5/13] SUSC-05 report exists...")
    if not REPORT_MD.exists():
        errors.append("SUSC-05 report missing")
    else:
        print("  OK")

    print("[6/13] All obligatory features have a card...")
    miss = [f for f in OBLIGATORY if f not in by_name]
    if miss:
        errors.append(f"obligatory features without card: {miss}")
    else:
        print(f"  OK: {len(OBLIGATORY)} obligatory features carded")

    print("[7/13] All cards have required fields...")
    if required_fields:
        for c in cards:
            missing_f = [k for k in required_fields if k not in c]
            if missing_f:
                errors.append(f"card '{c['feature_name']}' missing fields: {missing_f}")
        if not any("missing fields" in e for e in errors):
            print("  OK")

    print("[8/13] can_be_ground_truth=false for all cards...")
    gt = [c["feature_name"] for c in cards if c["can_be_ground_truth"] is not False]
    if gt:
        errors.append(f"can_be_ground_truth != false: {gt}")
    else:
        print("  OK")

    print("[9/13] allowed_for_training=false for all cards...")
    tr = [c["feature_name"] for c in cards if c["allowed_for_training"] is not False]
    if tr:
        errors.append(f"allowed_for_training != false: {tr}")
    else:
        print("  OK")

    print("[10/13] review_only=true for all cards...")
    ro = [c["feature_name"] for c in cards if c["review_only"] is not True]
    if ro:
        errors.append(f"review_only != true: {ro}")
    else:
        print("  OK")

    print("[11/13] No unresolved card is safe_for_score_v6=true; heuristics not GT...")
    bad = [
        c["feature_name"] for c in cards
        if c["status"] == "unresolved" and c["safe_for_score_v6"] is True
    ]
    if bad:
        errors.append(f"unresolved but safe_for_score_v6: {bad}")
    heur_gt = [
        c["feature_name"] for c in cards
        if c["feature_group"] in {"score", "heuristic_label", "proxy_v5"}
        and (c["can_be_ground_truth"] is not False or c["status"] != "proxy_only")
    ]
    if heur_gt:
        errors.append(f"heuristic card not proxy_only / GT-false: {heur_gt}")
    if not bad and not heur_gt:
        print("  OK")

    print("[12/13] no premature score v6 (candidate allowed); no model artifact...")
    # SUSC-10A introduces a review-only *candidate* score v6 downstream; it is an
    # allowed deliverable and excluded from this stage guard.
    v6 = [p.name for p in SUSC_DATASETS.glob("*score*v6*")
          if "candidate" not in p.name.lower()] if SUSC_DATASETS.exists() else []
    if v6:
        errors.append(f"unexpected non-candidate score v6 dataset: {v6}")
    models = []
    for base in (SUSC_DATASETS, SUSC_SCRIPTS, FC_DIR):
        if base.exists():
            models += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    if models:
        errors.append(f"unexpected model artifact(s): {models}")
    if not v6 and not models:
        print("  OK: no score v6 dataset, no model artifact")

    print("[13/13] Chain register contains the three commits...")
    if not CHAIN_REGISTER.exists():
        errors.append("chain register missing")
    elif not CHAIN_MANIFEST.exists():
        errors.append("chain manifest missing")
    else:
        reg = CHAIN_REGISTER.read_text(encoding="utf-8")
        man = CHAIN_MANIFEST.read_text(encoding="utf-8")
        for h in COMMIT_HASHES:
            if h not in reg:
                errors.append(f"commit {h} absent from chain register")
            if h not in man:
                errors.append(f"commit {h} absent from chain manifest")
        if not any("commit" in e for e in errors):
            print("  OK: 3 commits registered")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-05 validations passed")
    print("\nSUSC-05 produces methodology cards only. No score. No model. No ground truth.")
    print("Susceptibility matrix != confirmed flood occurrence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
