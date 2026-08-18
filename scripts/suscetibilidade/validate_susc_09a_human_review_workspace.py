"""
SUSC-09A Human Review Workspace Validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

WS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_human_review_workspace"
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_09a_human_review_workspace_schema_v1.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_human_review_workspace_report.md"
CASES = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"

FORBIDDEN_PRE_REVIEW = {"approved", "strong_candidate", "ground_truth", "training_ready"}
MANDATORY = "O SUSC-09A não executa nem substitui a revisão humana"


def main() -> int:
    print("=" * 60)
    print("SUSC-09A Human Review Workspace Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/8] Schema, workspace, report exist...")
    if not SCHEMA.exists():
        errors.append("schema missing")
    if not WS.exists():
        errors.append("workspace dir missing")
    if not REPORT.exists():
        errors.append("report missing")
    for rel_p in ("SUSC_09A_review_index.md", "SUSC_09A_review_index.csv",
                  "SUSC_09A_workspace_manifest.json",
                  "forms/SUSC_09A_human_review_form_prefill.csv",
                  "forms/SUSC_09A_human_review_instructions.md",
                  "geojson/SUSC_09A_review_cases.geojson",
                  "geojson/SUSC_09A_patch_bbox_layer.geojson",
                  "geojson/SUSC_09A_event_geometry_layer.geojson"):
        if not (WS / rel_p).exists():
            errors.append(f"missing workspace file: {rel_p}")
    if not errors:
        print("  OK")

    n_cases = len(read_csv(CASES)) if CASES.exists() else 0

    print("[2/8] One dossier and one SVG per case...")
    dossiers = list((WS / "case_dossiers").glob("*.md")) if (WS / "case_dossiers").exists() else []
    svgs = list((WS / "maps_svg").glob("*.svg")) if (WS / "maps_svg").exists() else []
    if len(dossiers) != n_cases:
        errors.append(f"dossiers {len(dossiers)} != cases {n_cases}")
    if len(svgs) != n_cases:
        errors.append(f"svg maps {len(svgs)} != cases {n_cases}")
    if not [e for e in errors if "dossier" in e or "svg" in e]:
        print(f"  OK: {len(dossiers)} dossiers, {len(svgs)} svg for {n_cases} cases")

    print("[3/8] Index governance + no forbidden machine_pre_review...")
    idx = read_csv(WS / "SUSC_09A_review_index.csv") if (WS / "SUSC_09A_review_index.csv").exists() else []
    for r in idx:
        if r.get("can_be_ground_truth") != "false" or r.get("allowed_for_training") != "false" or r.get("review_only") != "true":
            errors.append(f"index row {r.get('case_id')} governance violation")
        if r.get("machine_pre_review") in FORBIDDEN_PRE_REVIEW:
            errors.append(f"forbidden machine_pre_review: {r.get('case_id')}")
    if not [e for e in errors if "index row" in e or "forbidden" in e]:
        print(f"  OK: {len(idx)} index rows governed, no forbidden pre-review")

    print("[4/8] Every case requires human review...")
    if any(r.get("requires_human_review") != "true" for r in idx):
        errors.append("some index rows not requiring human review")
    else:
        print("  OK")

    print("[5/8] Form initializes approved_for_ground_truth=false...")
    form = read_csv(WS / "forms" / "SUSC_09A_human_review_form_prefill.csv") if (WS / "forms" / "SUSC_09A_human_review_form_prefill.csv").exists() else []
    bad = [r.get("case_id") for r in form if str(r.get("approved_for_ground_truth", "")).lower() != "false"]
    if bad:
        errors.append(f"form rows with GT != false: {bad}")
    else:
        print(f"  OK: {len(form)} form rows GT-false")

    print("[6/8] GeoJSON layers valid + not ground truth...")
    for layer in ("SUSC_09A_review_cases.geojson", "SUSC_09A_patch_bbox_layer.geojson",
                  "SUSC_09A_event_geometry_layer.geojson"):
        p = WS / "geojson" / layer
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if obj.get("type") != "FeatureCollection":
                    errors.append(f"{layer} not a FeatureCollection")
                if obj.get("properties", {}).get("can_be_ground_truth") is not False:
                    errors.append(f"{layer} not marked GT-false")
            except json.JSONDecodeError:
                errors.append(f"{layer} invalid JSON")
    if not [e for e in errors if ".geojson" in e]:
        print("  OK: 3 GeoJSON layers valid, GT-false")

    print("[7/8] Report contains mandatory statement...")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("report missing mandatory statement")
    elif REPORT.exists():
        print("  OK")

    print("[8/8] SUSC-03 matrix unchanged; no model...")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix SHA256 changed")
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
    print("PASSED: All SUSC-09A validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
