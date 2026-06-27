"""
SUSC-07B Real Coordinate Enrichment Validation

Validates the SUSC-07B deliverables for structural integrity and governance.

Checks (fail-closed):
  - schema, local scan, acquisition queue, acquisition manifest, coordinate
    extraction, enriched catalog, refined overlay, report all exist;
  - no record has can_be_ground_truth=true / allowed_for_training=true;
  - all records review_only=true;
  - no forbidden geometry source was used (centroid/geocoding/etc.);
  - can_link_to_patch=true requires explicit geometry (lat/lon, bbox, geojson, wkt);
  - report contains the mandatory statement;
  - the SUSC-03 matrix is unchanged (SHA256);
  - no model artifact was created.

This script does NOT create ground truth and does NOT modify the matrix.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_07b_event_geometry_acquisition_schema_v1.json"
SCAN = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_source_scan.csv"
QUEUE = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_external_coordinate_acquisition_queue.csv"
ACQ_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_07b_external_coordinate_acquisition_manifest_v1.csv"
COORDS = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"
ENRICHED = ROOT / "datasets" / "suscetibilidade" / "susc_07b_event_evidence_geometry_enriched_v1.csv"
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_enrichment_report.md"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ART_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_features_by_patch_v1_artifact_manifest.json"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
MANDATORY = "não cria ground truth de enchente por patch"
FORBIDDEN_TOKENS = ("centroid", "centroide", "geocod", "manual_guess", "visual_estimate",
                    "unreferenced")


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    print("=" * 60)
    print("SUSC-07B Real Coordinate Enrichment Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/10] Required artifacts exist...")
    required = [SCHEMA, SCAN, QUEUE, ACQ_MANIFEST, COORDS, ENRICHED, OVERLAY, REPORT]
    for p in required:
        if not p.exists():
            errors.append(f"missing artifact: {p.relative_to(ROOT)}")
    if not errors:
        print(f"  OK: {len(required)} artifacts present")

    coords = load_csv(COORDS) if COORDS.exists() else []
    enriched = load_csv(ENRICHED) if ENRICHED.exists() else []
    overlay = load_csv(OVERLAY) if OVERLAY.exists() else []

    def gov_ok(rows, gt_key, label):
        bad = []
        for r in rows:
            if str(r.get(gt_key, "false")).lower() == "true":
                bad.append(r)
            if str(r.get("allowed_for_training", "false")).lower() == "true":
                bad.append(r)
            if str(r.get("review_only", "true")).lower() != "true":
                bad.append(r)
        if bad:
            errors.append(f"{label}: {len(bad)} governance violation(s)")

    print("[2/10] Coordinates governance (GT/training/review_only)...")
    gov_ok(coords, "can_be_ground_truth", "coordinates")
    print("[3/10] Enriched catalog governance...")
    gov_ok(enriched, "can_be_ground_truth", "enriched")
    print("[4/10] Refined overlay governance...")
    gov_ok(overlay, "can_be_used_as_ground_truth", "overlay")
    if not [e for e in errors if "governance" in e]:
        print("  OK: coordinates/enriched/overlay all GT-false, training-false, review_only")

    print("[5/10] No forbidden geometry source used...")
    bad_src = []
    for r in coords:
        blob = (r.get("source_path_or_url", "") + r.get("extraction_method", "")
                + r.get("source_confidence", "")).lower()
        if any(t in blob for t in FORBIDDEN_TOKENS):
            bad_src.append(r.get("coordinate_id"))
    if bad_src:
        errors.append(f"forbidden geometry source used: {bad_src}")
    else:
        print("  OK")

    print("[6/10] can_link_to_patch=true requires explicit geometry...")
    bad_link = []
    for r in coords:
        if str(r.get("can_link_to_patch", "")).lower() == "true":
            has_geom = bool(r.get("bbox") or (r.get("lat") and r.get("lon"))
                            or r.get("wkt") or r.get("geojson_ref"))
            if not has_geom:
                bad_link.append(r.get("coordinate_id"))
    if bad_link:
        errors.append(f"linkable coords without explicit geometry: {bad_link}")
    else:
        print(f"  OK: {sum(1 for r in coords if r.get('can_link_to_patch')=='true')} linkable, all with geometry")

    print("[7/10] Report contains mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[8/10] SUSC-03 matrix unchanged (SHA256)...")
    if not MATRIX.exists():
        errors.append("SUSC-03 matrix missing")
    elif ART_MANIFEST.exists():
        expected = json.loads(ART_MANIFEST.read_text(encoding="utf-8")).get("sha256")
        h = hashlib.sha256()
        with MATRIX.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            errors.append("SUSC-03 matrix SHA256 changed")
        else:
            print("  OK: matrix SHA256 matches")

    print("[9/10] No model artifact persisted...")
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

    print("[10/10] Acquisition was offline-safe (no heavy/auto download)...")
    if ACQ_MANIFEST.exists():
        acq = load_csv(ACQ_MANIFEST)
        heavy = [r for r in acq if str(r.get("download_status", "")).startswith("downloaded")
                 and r.get("source_type") not in {"manual_placement"}]
        # downloads are allowed only as light manual placements; auto-heavy is a violation
        if heavy:
            errors.append(f"unexpected automatic downloads: {len(heavy)}")
        else:
            print("  OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-07B validations passed")
    print("\nReal traceable coordinates only. No invention. No patch ground truth. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
