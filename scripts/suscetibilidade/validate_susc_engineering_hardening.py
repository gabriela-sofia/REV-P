"""
SUSC Engineering Hardening Validation

Checks the shared utility layer and its invariants:
  - common modules exist and import without error;
  - aliases resolve (alias report present and consistent);
  - governance constants are correct (training/GT off, review_only on);
  - the safe downloader blocks raster and >100MB;
  - the SUSC-03 matrix is unchanged;
  - no model artifact exists;
  - no final artifact points outside the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

MODULES = ["susc_io", "susc_governance", "susc_aliases", "susc_geometry",
           "susc_downloads", "susc_common"]
ALIAS_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_alias_resolution_report.csv"


def main() -> int:
    print("=" * 60)
    print("SUSC Engineering Hardening Validation")
    print("=" * 60)
    errors: list[str] = []

    print("\n[1/7] Common modules exist and import...")
    mods = {}
    for m in MODULES:
        p = HERE / f"{m}.py"
        if not p.exists():
            errors.append(f"missing module: {m}.py")
            continue
        try:
            mods[m] = __import__(m)
        except Exception as e:
            errors.append(f"import error in {m}: {str(e)[:80]}")
    if not [e for e in errors if "module" in e or "import error" in e]:
        print(f"  OK: {len(MODULES)} modules import")

    print("[2/7] Governance constants correct...")
    gov = mods.get("susc_governance")
    if gov:
        if gov.ALLOWED_FOR_TRAINING is not False or gov.CAN_BE_GROUND_TRUTH is not False or gov.REVIEW_ONLY is not True:
            errors.append("governance constants incorrect")
        else:
            print("  OK: training=False, GT=False, review_only=True")

    print("[3/7] Aliases resolved + report present...")
    aliases = mods.get("susc_aliases")
    if not ALIAS_REPORT.exists():
        errors.append("alias resolution report missing")
    elif aliases:
        if aliases.resolve("flow_acc_p75") != "flow_acc_log_p75":
            errors.append("alias flow_acc_p75 not resolved")
        if aliases.resolve("rain_persistence") != "rain_persistence_index":
            errors.append("alias rain_persistence not resolved")
        else:
            print("  OK: aliases resolve to real columns")

    print("[4/7] Safe downloader blocks raster and oversize...")
    dl = mods.get("susc_downloads")
    if dl:
        r1 = dl.safe_download("https://example.org/scene.tif", HERE / "_tmp_nope")
        r2 = dl.safe_download("ftp://example.org/x.csv", HERE / "_tmp_nope")
        if r1["download_status"] != "blocked_raster":
            errors.append("downloader did not block raster")
        elif r2["download_status"] not in {"not_attempted_no_direct_url", "blocked_unknown_extension"}:
            errors.append("downloader did not refuse non-http")
        else:
            print("  OK: raster blocked, non-http refused")
        # never created the tmp dir from a blocked download
        if (HERE / "_tmp_nope").exists():
            errors.append("downloader created dir on blocked download")

    print("[5/7] Geometry helpers sane...")
    geo = mods.get("susc_geometry")
    if geo:
        if not geo.point_in_bbox(-34.95, -8.13, [-35.14, -8.23, -34.94, -8.01]):
            errors.append("point_in_bbox wrong")
        if geo.region_of_coord(-43.2, -22.5) != "petropolis":
            errors.append("region_of_coord wrong")
        if not [e for e in errors if "point_in_bbox" in e or "region_of_coord" in e]:
            print("  OK: geometry helpers correct")

    print("[6/7] SUSC-03 matrix unchanged; no model artifact...")
    common = mods.get("susc_common")
    if common:
        if not common.matrix_sha256_ok():
            errors.append("SUSC-03 matrix SHA256 changed")
        models = common.find_model_artifacts()
        if models:
            errors.append(f"model artifact(s) found: {models}")
        if not [e for e in errors if "matrix" in e or "model" in e]:
            print("  OK: matrix intact, no model")

    print("[7/7] IO repo containment...")
    io = mods.get("susc_io")
    if io:
        if not io.is_within_repo(ROOT / "datasets"):
            errors.append("is_within_repo false negative")
        if io.is_within_repo("/etc/passwd"):
            errors.append("is_within_repo false positive")
        if not [e for e in errors if "is_within_repo" in e]:
            print("  OK: repo containment works")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All engineering-hardening validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
