"""SUSC-11A Public Visual Package Validation."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_json  # noqa: E402
from susc_common import matrix_sha256_ok, find_model_artifacts  # noqa: E402

PKG = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11A_public_visual_package"
MANIFEST = PKG / "SUSC_11A_visual_package_manifest.json"
README = PKG / "README.md"
MANDATORY = "As figuras não representam ground truth"
REQUIRED_TABLES = ["top_patches_for_tcc.csv", "score_v6_by_region.csv", "cases_for_tcc_with_caution.csv",
                   "limitations_table.csv", "methodology_artifacts_index.csv"]
REQUIRED_SLIDES = [f"slide_0{i}_" for i in range(1, 9)]


def main() -> int:
    print("=" * 60)
    print("SUSC-11A Public Visual Package Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/7] Package structure exists...")
    for sub in ("figures", "tables", "geojson", "slides_content"):
        if not (PKG / sub).exists():
            errors.append(f"missing dir: {sub}")
    for p in (README, MANIFEST):
        if not p.exists():
            errors.append(f"missing: {p.name}")
    if not errors:
        print("  OK")

    print("[2/7] At least 9 SVG figures...")
    figs = list((PKG / "figures").glob("*.svg")) if (PKG / "figures").exists() else []
    if len(figs) < 9:
        errors.append(f"only {len(figs)} figures (<9)")
    else:
        print(f"  OK: {len(figs)} figures")

    print("[3/7] All required tables present...")
    for t in REQUIRED_TABLES:
        if not (PKG / "tables" / t).exists():
            errors.append(f"missing table: {t}")
    if not [e for e in errors if "table" in e]:
        print("  OK: 5 tables")

    print("[4/7] All 8 slides present...")
    slides = [p.name for p in (PKG / "slides_content").glob("*.md")] if (PKG / "slides_content").exists() else []
    for pref in REQUIRED_SLIDES:
        if not any(s.startswith(pref) for s in slides):
            errors.append(f"missing slide: {pref}")
    if not [e for e in errors if "slide" in e]:
        print(f"  OK: {len(slides)} slides")

    print("[5/7] Figures are valid SVG with review-only footer...")
    bad = []
    for f in figs:
        txt = f.read_text(encoding="utf-8")
        if not txt.startswith("<svg") or "review-only" not in txt:
            bad.append(f.name)
    if bad:
        errors.append(f"figures missing svg/footer: {bad[:3]}")
    else:
        print("  OK")

    print("[6/7] README + manifest governance + mandatory statement...")
    if README.exists() and MANDATORY not in README.read_text(encoding="utf-8"):
        errors.append("README missing mandatory statement")
    if MANIFEST.exists():
        m = read_json(MANIFEST)
        if m.get("can_be_used_as_ground_truth") is not False or m.get("allowed_for_training") is not False or m.get("review_only") is not True:
            errors.append("manifest governance flags incorrect")
    if not [e for e in errors if "README" in e or "manifest" in e]:
        print("  OK")

    print("[7/7] SUSC-03 matrix unchanged; no model...")
    if not matrix_sha256_ok():
        errors.append("matrix SHA256 changed")
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
    print("PASSED: All SUSC-11A validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
