"""
SUSC-08 Spatiotemporal Review Validation

Validates the SUSC-08 human-review package for structural integrity and governance.

Checks (fail-closed):
  - schema, review-cases CSV, casebook MD, summary CSV, checklist, form CSV,
    report all exist;
  - every case is can_be_ground_truth=false / allowed_for_training=false /
    review_only=true;
  - every real spatial case (bbox_overlap / near_patch_buffer_candidate) has
    requires_human_review=true;
  - no spatial case is marked as ground truth;
  - the review form initializes approved_for_ground_truth=false;
  - report contains the mandatory statement;
  - the SUSC-03 matrix is unchanged (SHA256);
  - no model artifact was created.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_08_spatiotemporal_review_schema_v1.json"
CASES = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"
CASEBOOK = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_review_casebook.md"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_review_casebook_summary.csv"
CHECKLIST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_manual_review_checklist.md"
FORM = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_manual_review_form.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_spatiotemporal_review_report.md"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ART_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_features_by_patch_v1_artifact_manifest.json"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
MANDATORY = "não cria ground truth de enchente por patch"
SPATIAL = {"bbox_overlap", "near_patch_buffer_candidate", "exact_patch_overlap"}


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    print("=" * 60)
    print("SUSC-08 Spatiotemporal Review Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/9] Required artifacts exist...")
    required = [SCHEMA, CASES, CASEBOOK, SUMMARY, CHECKLIST, FORM, REPORT]
    for p in required:
        if not p.exists():
            errors.append(f"missing artifact: {p.relative_to(ROOT)}")
    if not errors:
        print(f"  OK: {len(required)} artifacts present")

    cases = load_csv(CASES) if CASES.exists() else []
    form = load_csv(FORM) if FORM.exists() else []

    print("[2/9] Cases governance (GT/training/review_only)...")
    bad = [c["case_id"] for c in cases if c.get("can_be_ground_truth") != "false"
           or c.get("allowed_for_training") != "false" or c.get("review_only") != "true"]
    if bad:
        errors.append(f"cases violating governance: {bad}")
    else:
        print(f"  OK: {len(cases)} cases governed")

    print("[3/9] Every real spatial case requires_human_review=true...")
    bad_hr = [c["case_id"] for c in cases
              if c["spatial_relation"] in SPATIAL and c.get("requires_human_review") != "true"]
    if bad_hr:
        errors.append(f"spatial cases without requires_human_review: {bad_hr}")
    else:
        n = sum(1 for c in cases if c["spatial_relation"] in SPATIAL)
        print(f"  OK: {n} spatial cases all require human review")

    print("[4/9] No spatial case is ground truth...")
    gt = [c["case_id"] for c in cases if c["spatial_relation"] in SPATIAL
          and c.get("can_be_ground_truth") == "true"]
    if gt:
        errors.append(f"spatial cases marked ground truth: {gt}")
    else:
        print("  OK")

    print("[5/9] No case strength is strong without official footprint...")
    # honest guard: strong_candidate must not appear unless explicitly justified
    strong = [c["case_id"] for c in cases if c.get("case_strength_pre_review") == "strong_candidate"]
    if strong:
        errors.append(f"unexpected strong_candidate (no official validated footprint): {strong}")
    else:
        print("  OK: no strong_candidate asserted")

    print("[6/9] Review form initializes approved_for_ground_truth=false...")
    bad_form = [r.get("case_id") for r in form if str(r.get("approved_for_ground_truth", "")).lower() != "false"]
    if bad_form:
        errors.append(f"form rows with approved_for_ground_truth != false: {bad_form}")
    else:
        print(f"  OK: {len(form)} form rows, all GT-false")

    print("[7/9] Report contains mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[8/9] SUSC-03 matrix unchanged (SHA256)...")
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

    print("[9/9] No model artifact persisted...")
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

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All SUSC-08 validations passed")
    print("\nHuman review package. No patch ground truth. No training. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
