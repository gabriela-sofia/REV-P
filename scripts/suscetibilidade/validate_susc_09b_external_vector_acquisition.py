"""
SUSC-09B External Vector Acquisition Validation
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_09b_external_vector_acquisition_schema_v1.json"
REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_source_registry_v1.csv"
DL_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_download_manifest_v1.csv"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_09b_external_geometries_parsed_v1.csv"
CANDIDATES = ROOT / "datasets" / "suscetibilidade" / "susc_09b_event_geometry_candidates_v1.csv"
PARSE_AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_parse_audit.csv"
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_overlay_candidates.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_limitations.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_acquisition_report.md"
DL_DIR = ROOT / "datasets" / "suscetibilidade" / "external_event_geometry_sources_susc09b"

RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf", ".grib"}
MAX_BYTES = 100 * 1024 * 1024


def main() -> int:
    print("=" * 60)
    print("SUSC-09B External Vector Acquisition Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/8] Required artifacts exist...")
    for p in (SCHEMA, REGISTRY, DL_MANIFEST, PARSED, CANDIDATES, PARSE_AUDIT, OVERLAY, LIMITS, REPORT):
        if not p.exists():
            errors.append(f"missing: {p.relative_to(ROOT)}")
    if not errors:
        print("  OK: 9 artifacts present")

    dl = read_csv(DL_MANIFEST) if DL_MANIFEST.exists() else []
    parsed = read_csv(PARSED) if PARSED.exists() else []
    candidates = read_csv(CANDIDATES) if CANDIDATES.exists() else []
    overlay = read_csv(OVERLAY) if OVERLAY.exists() else []

    print("[2/8] No raster downloaded...")
    bad = [r for r in dl if any(str(r.get("content_ext", "")).lower() == e for e in RASTER_EXTS)
           and r.get("download_status") == "downloaded"]
    if bad:
        errors.append(f"raster downloaded: {len(bad)}")
    else:
        print("  OK")

    print("[3/8] No file > 100MB downloaded...")
    big = []
    for r in dl:
        sz = r.get("file_size_bytes", "")
        if sz and str(sz).isdigit() and int(sz) > MAX_BYTES:
            big.append(r.get("download_id"))
    if big:
        errors.append(f"files > 100MB: {big}")
    else:
        print("  OK")

    print("[4/8] No raster file landed in the download dir...")
    if DL_DIR.exists():
        raster_files = [p.name for p in DL_DIR.rglob("*") if p.suffix.lower() in RASTER_EXTS]
        if raster_files:
            errors.append(f"raster files present: {raster_files}")
        else:
            print("  OK")

    print("[5/8] Governance: no GT / no training / review_only...")
    for label, rows, gt in (("parsed", parsed, "can_be_ground_truth"),
                            ("candidates", candidates, "can_be_ground_truth"),
                            ("overlay", overlay, "can_be_used_as_ground_truth")):
        for r in rows:
            if str(r.get(gt, "false")).lower() == "true" or str(r.get("allowed_for_training", "false")).lower() == "true" or str(r.get("review_only", "true")).lower() != "true":
                errors.append(f"{label} governance violation")
                break
    if not [e for e in errors if "governance" in e]:
        print("  OK")

    print("[6/8] Downloads recorded with URL; sha256 present when downloaded...")
    for r in dl:
        if not r.get("url_or_local_path"):
            errors.append("download manifest row without url")
            break
        if r.get("download_status") == "downloaded" and not r.get("sha256"):
            errors.append("downloaded file without sha256")
            break
    if not [e for e in errors if "download manifest" in e or "sha256" in e]:
        print("  OK")

    print("[7/8] can_link_to_patch=true only with bbox geometry...")
    for r in candidates:
        if str(r.get("can_link_to_patch", "")).lower() == "true" and not r.get("bbox"):
            errors.append(f"candidate linkable without geometry: {r.get('candidate_id')}")
    if not [e for e in errors if "linkable" in e]:
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
    print("PASSED: All SUSC-09B validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
