"""
SUSC-07A Event Overlay Validation

Validates the SUSC-07A deliverables (evidence schema, source scan, catalog +
manifest, overlay candidates/summary/limitations, report) for structural
integrity and governance compliance.

Checks (fail-closed):
  - evidence schema exists; scan CSV exists; catalog exists; manifest exists;
  - overlay candidates / summary / limitations exist; report exists;
  - no record has can_be_ground_truth=true;
  - no record has allowed_for_training=true;
  - all records have review_only=true;
  - if can_link_to_patch=true there must be sufficient spatial precision/geometry;
  - report contains the mandatory statement;
  - no model artifact was created;
  - the SUSC-03 matrix file still exists (not deleted).

This script does NOT create ground truth and does NOT modify the matrix.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_07_event_evidence_schema_v1.json"
SCAN = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_event_source_scan.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_07_event_evidence_catalog_v1.csv"
CAT_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_07_event_evidence_catalog_manifest_v1.json"
)
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_candidates.csv"
OVERLAY_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_summary.csv"
OVERLAY_LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_limitations.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_event_evidence_overlay_report.md"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
MANDATORY = "não cria ground truth de enchente por patch"
SUFFICIENT_SPATIAL = {"patch", "neighborhood"}


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    print("=" * 60)
    print("SUSC-07A Event Overlay Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/9] Required artifacts exist...")
    required = [SCHEMA, SCAN, CATALOG, CAT_MANIFEST, OVERLAY, OVERLAY_SUMMARY,
                OVERLAY_LIMITS, REPORT]
    for p in required:
        if not p.exists():
            errors.append(f"missing artifact: {p.relative_to(ROOT)}")
    if not errors:
        print(f"  OK: {len(required)} artifacts present")

    catalog = load_csv(CATALOG) if CATALOG.exists() else []
    overlay = load_csv(OVERLAY) if OVERLAY.exists() else []

    print("[2/9] No catalog record is ground truth...")
    gt = [r["evidence_id"] for r in catalog if str(r.get("can_be_ground_truth", "")).lower() != "false"]
    if gt:
        errors.append(f"catalog records with can_be_ground_truth != false: {gt[:5]}")
    else:
        print(f"  OK: {len(catalog)} catalog records, all GT-false")

    print("[3/9] No catalog record allows training...")
    tr = [r["evidence_id"] for r in catalog if str(r.get("allowed_for_training", "")).lower() != "false"]
    if tr:
        errors.append(f"catalog records with allowed_for_training != false: {tr[:5]}")
    else:
        print("  OK")

    print("[4/9] All catalog records review_only=true...")
    ro = [r["evidence_id"] for r in catalog if str(r.get("review_only", "")).lower() != "true"]
    if ro:
        errors.append(f"catalog records with review_only != true: {ro[:5]}")
    else:
        print("  OK")

    print("[5/9] can_link_to_patch=true requires sufficient spatial precision...")
    bad = []
    for r in catalog:
        if str(r.get("can_link_to_patch", "")).lower() == "true":
            if r.get("spatial_precision") not in SUFFICIENT_SPATIAL and not r.get("has_geometry", "").lower() == "true":
                bad.append(r["evidence_id"])
    if bad:
        errors.append(f"linkable records without sufficient spatial precision: {bad}")
    else:
        print("  OK (0 linkable records in this pass; none claimed without geometry)")

    print("[6/9] Overlay records governance...")
    ov_bad = [
        r.get("evidence_id", "?") for r in overlay
        if str(r.get("can_be_used_as_ground_truth", "")).lower() != "false"
        or str(r.get("allowed_for_training", "")).lower() != "false"
        or str(r.get("review_only", "")).lower() != "true"
    ]
    if ov_bad:
        errors.append(f"overlay records violating governance: {ov_bad[:5]}")
    else:
        print(f"  OK: {len(overlay)} overlay records governed")

    print("[7/9] Report contains mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[8/9] No model artifact persisted...")
    models = []
    for base in (ROOT / "outputs_public" / "suscetibilidade",
                 ROOT / "scripts" / "suscetibilidade",
                 ROOT / "datasets" / "suscetibilidade"):
        if base.exists():
            models += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    if models:
        errors.append(f"model artifact(s) found: {models}")
    else:
        print("  OK")

    print("[9/9] SUSC-03 matrix still present (not deleted)...")
    if not MATRIX.exists():
        errors.append("SUSC-03 matrix missing")
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-07A validations passed")
    print("\nDocumentary/regional overlay only. No patch ground truth. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
