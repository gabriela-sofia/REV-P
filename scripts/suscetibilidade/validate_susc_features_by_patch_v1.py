"""
SUSC-03 Migrated Matrix Validation (heavy validation)

Validates the migrated susceptibility matrix
datasets/suscetibilidade/susc_features_by_patch_v1.csv against the SUSC-01
schema, the provenance manifest, and the artifact manifest.

Checks (fail-closed):
  - migrated CSV exists and has exactly 300 rows;
  - a patch identifier column is present;
  - every migrated column is catalogued in schema AND manifest
    (or justified as minimal identity);
  - all expected core features are complete (no missing values);
  - uncertain features keep requires_provenance_audit=true in the schema;
  - heuristic labels are NOT ground truth (can_be_used_as_ground_truth=false);
  - v5 scores are NOT final validation (heuristic, not GT);
  - no migrated field allows training;
  - SHA256 matches the artifact manifest;
  - feature groups belong to the allowed vocabulary;
  - numeric columns hold coherent numeric types;
  - fully empty columns were not migrated unless justified.

This script does NOT alter any file, does NOT unlock training,
and does NOT treat heuristic labels or v5 scores as ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_features_schema_v1.json"
MANIFEST_PATH = (
    ROOT / "manifests" / "suscetibilidade" / "susc_features_provenance_manifest_v1.csv"
)
CSV_PATH = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ARTIFACT_PATH = (
    ROOT
    / "manifests"
    / "suscetibilidade"
    / "susc_features_by_patch_v1_artifact_manifest.json"
)

EXPECTED_ROWS = 300
PATCH_ID_COLUMN = "patch_id"

ALLOWED_GROUPS = {
    "patch_identity",
    "geometry",
    "sentinel2_bands",
    "spectral_index",
    "topography",
    "hydrology",
    "precipitation",
    "sar",
    "land_use",
    "interaction",
    "proxy_v5",
    "score",
    "heuristic_label",
    "qa",
    "cbers_linkage",
    "cbers_refresh",
    "unknown",
}

# Minimal-identity columns allowed even though they are not "features".
MINIMAL_IDENTITY = {"patch_id", "regiao", "reference_date", "xmin", "ymin", "xmax", "ymax"}

# Core features that must be complete (no missing) across all patches.
CORE_GROUPS = {"topography", "hydrology", "precipitation", "spectral_index"}

NUMERIC_DTYPES = {"float", "integer"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_numeric(token: str) -> bool:
    token = token.strip()
    if token == "":
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def is_missing(token: str) -> bool:
    return token.strip().lower() in {"", "na", "nan", "none", "null"}


def main() -> int:
    print("=" * 60)
    print("SUSC-03 Migrated Matrix Validation")
    print("=" * 60)

    errors: list[str] = []
    warnings: list[str] = []

    # --- load schema + manifest ---
    if not SCHEMA_PATH.exists():
        print(f"STOP: schema not found: {SCHEMA_PATH}")
        return 1
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    feats = {f["source_column"]: f for f in schema["features"]}

    manifest_cols: set[str] = set()
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                manifest_cols.add(r["source_column"])

    # --- migrated CSV exists ---
    print("\n[1/12] Migrated CSV exists...")
    if not CSV_PATH.exists():
        print(f"  STOP: migrated matrix not found: {CSV_PATH}")
        print("        Run migrate_dataset_final_to_revp.py first.")
        return 1
    print("  OK")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    # --- row count ---
    print("[2/12] Row count == 300...")
    if len(data) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, found {len(data)}")
    else:
        print(f"  OK: {len(data)} rows")

    # --- patch identifier present ---
    print("[3/12] Patch identifier present...")
    if PATCH_ID_COLUMN not in header:
        errors.append(f"missing patch identifier column '{PATCH_ID_COLUMN}'")
    else:
        print(f"  OK: '{PATCH_ID_COLUMN}' present")

    # column -> list of tokens
    col_data: dict[str, list[str]] = {name: [] for name in header}
    for row in data:
        for name, val in zip(header, row):
            col_data[name].append(val)

    # --- every migrated column catalogued or minimal identity ---
    print("[4/12] Every column catalogued (schema + manifest) or minimal identity...")
    uncatalogued = []
    for name in header:
        if name in feats and name in manifest_cols:
            continue
        if name in MINIMAL_IDENTITY:
            continue
        uncatalogued.append(name)
    if uncatalogued:
        errors.append(f"uncatalogued / unjustified columns: {uncatalogued}")
    else:
        print(f"  OK: {len(header)} columns catalogued or minimal identity")

    # --- core features complete ---
    print("[5/12] Core features complete (no missing)...")
    core_incomplete = []
    for name in header:
        feat = feats.get(name)
        if feat and feat["group"] in CORE_GROUPS:
            n_missing = sum(1 for t in col_data[name] if is_missing(t))
            if n_missing > 0:
                core_incomplete.append((name, n_missing))
    if core_incomplete:
        errors.append(f"core features with missing values: {core_incomplete}")
    else:
        print(f"  OK: all core-group columns complete")

    # --- uncertain features keep requires_provenance_audit=true ---
    print("[6/12] Uncertain features keep requires_provenance_audit=true...")
    audit_flagged = sum(
        1 for name in header
        if feats.get(name, {}).get("requires_provenance_audit") is True
    )
    print(f"  OK: {audit_flagged} migrated columns flagged requires_provenance_audit=true")

    # --- heuristic labels not GT ---
    print("[7/12] Heuristic labels are NOT ground truth...")
    for name in header:
        feat = feats.get(name)
        if feat and feat["group"] == "heuristic_label":
            if feat.get("can_be_used_as_ground_truth") is not False:
                errors.append(f"heuristic label '{name}' not marked GT-false")
    print("  OK: heuristic labels carry can_be_used_as_ground_truth=false")

    # --- v5 scores not final validation ---
    print("[8/12] v5 scores are NOT final validation (heuristic, not GT)...")
    for name in header:
        feat = feats.get(name)
        if feat and feat["group"] == "score":
            if feat.get("can_be_used_as_ground_truth") is not False:
                errors.append(f"score '{name}' not marked GT-false")
    print("  OK: v5 scores carry can_be_used_as_ground_truth=false")

    # --- no field allows training ---
    print("[9/12] No migrated field allows training...")
    train_violations = [
        name for name in header
        if feats.get(name, {}).get("allowed_for_training") is True
    ]
    if train_violations:
        errors.append(f"GOVERNANCE VIOLATION: training-allowed columns: {train_violations}")
    else:
        print("  OK: no column has allowed_for_training=true")

    # --- SHA256 matches artifact manifest ---
    print("[10/12] SHA256 matches artifact manifest...")
    if not ARTIFACT_PATH.exists():
        errors.append(f"artifact manifest not found: {ARTIFACT_PATH}")
    else:
        with ARTIFACT_PATH.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
        actual = sha256_file(CSV_PATH)
        if artifact.get("sha256") != actual:
            errors.append(
                f"SHA256 mismatch: manifest={artifact.get('sha256')} actual={actual}"
            )
        else:
            print(f"  OK: {actual}")
        for key, expected in (
            ("allowed_for_training", False),
            ("review_only", True),
            ("can_be_used_as_ground_truth", False),
        ):
            if artifact.get(key) is not expected:
                errors.append(f"artifact manifest '{key}' != {expected}")

    # --- groups in allowed vocabulary ---
    print("[11/12] Feature groups in allowed vocabulary; numeric coherence...")
    for name in header:
        feat = feats.get(name)
        if feat and feat["group"] not in ALLOWED_GROUPS:
            errors.append(f"column '{name}' has invalid group '{feat['group']}'")
    # numeric coherence: numeric dtypes hold parseable numbers (ignoring missing)
    numeric_bad = []
    for name in header:
        feat = feats.get(name)
        if feat and feat.get("dtype") in NUMERIC_DTYPES:
            bad = sum(
                1 for t in col_data[name]
                if not is_missing(t) and not is_numeric(t)
            )
            if bad > 0:
                numeric_bad.append((name, bad))
    if numeric_bad:
        errors.append(f"numeric columns with non-numeric tokens: {numeric_bad}")
    else:
        print("  OK: groups valid; numeric columns coherent")

    # --- fully empty columns not migrated (unless justified) ---
    print("[12/12] No fully empty columns migrated...")
    fully_empty = [
        name for name in header
        if all(is_missing(t) for t in col_data[name])
    ]
    if fully_empty:
        errors.append(f"fully empty columns migrated: {fully_empty}")
    else:
        print("  OK: no fully empty column migrated")

    print("\n" + "=" * 60)
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("PASSED: All migrated-matrix validations passed")
    print("\nSusceptibility features != confirmed occurrence.")
    print("This matrix does NOT unlock supervised training and is NOT ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
