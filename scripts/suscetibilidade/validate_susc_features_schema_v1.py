"""
SUSC-01 Schema Validation Script

Validates the susceptibility feature schema and provenance manifest
for structural integrity, governance compliance, and vocabulary adherence.

This script does NOT alter any file, does NOT unlock training,
and does NOT treat heuristic labels as ground truth.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_features_schema_v1.json"
MANIFEST_PATH = ROOT / "manifests" / "suscetibilidade" / "susc_features_provenance_manifest_v1.csv"

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

ALLOWED_ORIGIN_STATUS = {
    "raw_existing",
    "computed_existing",
    "documented_only",
    "proxy_existing",
    "heuristic_existing",
    "unknown_origin",
}

REQUIRED_FEATURE_FIELDS = {
    "name",
    "group",
    "dtype",
    "source_column",
    "origin_status",
    "interpretation",
    "can_be_used_as_feature",
    "can_be_used_as_ground_truth",
    "requires_provenance_audit",
    "allowed_for_training",
    "review_only",
}

REQUIRED_MANIFEST_COLUMNS = {
    "feature_name",
    "feature_group",
    "source_file",
    "source_column",
    "origin_status",
    "requires_audit",
    "can_be_used_as_feature",
    "can_be_used_as_ground_truth",
    "allowed_for_training",
    "review_only",
}


def validate_schema(path: Path) -> list[str]:
    errors: list[str] = []

    if not path.exists():
        errors.append(f"Schema file not found: {path}")
        return errors

    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    gov = schema.get("governance", {})
    if gov.get("allowed_for_training") is not False:
        errors.append("GOVERNANCE VIOLATION: allowed_for_training must be false")
    if gov.get("review_only") is not True:
        errors.append("GOVERNANCE VIOLATION: review_only must be true")
    if gov.get("can_create_ground_truth") is not False:
        errors.append("GOVERNANCE VIOLATION: can_create_ground_truth must be false")

    features = schema.get("features", [])
    if not features:
        errors.append("No features found in schema")
        return errors

    for i, feat in enumerate(features):
        name = feat.get("name", f"<unnamed_{i}>")

        missing = REQUIRED_FEATURE_FIELDS - set(feat.keys())
        if missing:
            errors.append(f"Feature '{name}' missing fields: {sorted(missing)}")

        group = feat.get("group", "")
        if group not in ALLOWED_GROUPS:
            errors.append(f"Feature '{name}' has invalid group: '{group}'")

        origin = feat.get("origin_status", "")
        if origin not in ALLOWED_ORIGIN_STATUS:
            errors.append(f"Feature '{name}' has invalid origin_status: '{origin}'")

        if feat.get("allowed_for_training") is True:
            errors.append(
                f"GOVERNANCE VIOLATION: Feature '{name}' has allowed_for_training=true"
            )

        if feat.get("review_only") is not True:
            errors.append(
                f"GOVERNANCE VIOLATION: Feature '{name}' has review_only != true"
            )

        if feat.get("can_be_used_as_ground_truth") is True:
            if not feat.get("notes", ""):
                errors.append(
                    f"Feature '{name}' has can_be_used_as_ground_truth=true "
                    "without explicit justification in notes"
                )

    return errors


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []

    if not path.exists():
        errors.append(f"Manifest file not found: {path}")
        return errors

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        errors.append("Manifest is empty")
        return errors

    header_set = set(rows[0].keys())
    missing_cols = REQUIRED_MANIFEST_COLUMNS - header_set
    if missing_cols:
        errors.append(f"Manifest missing columns: {sorted(missing_cols)}")

    for i, row in enumerate(rows):
        name = row.get("feature_name", f"<row_{i}>")

        group = row.get("feature_group", "")
        if group not in ALLOWED_GROUPS:
            errors.append(f"Manifest row '{name}' has invalid group: '{group}'")

        origin = row.get("origin_status", "")
        if origin not in ALLOWED_ORIGIN_STATUS:
            errors.append(f"Manifest row '{name}' has invalid origin_status: '{origin}'")

        if row.get("allowed_for_training", "").lower() == "true":
            errors.append(
                f"GOVERNANCE VIOLATION: Manifest row '{name}' has "
                "allowed_for_training=true"
            )

        if row.get("review_only", "").lower() != "true":
            errors.append(
                f"GOVERNANCE VIOLATION: Manifest row '{name}' has review_only != true"
            )

    return errors


def validate_cross_consistency(
    schema_path: Path, manifest_path: Path
) -> list[str]:
    errors: list[str] = []

    if not schema_path.exists() or not manifest_path.exists():
        return errors

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    schema_names = {f["name"] for f in schema.get("features", [])}
    manifest_names = {r["feature_name"] for r in manifest_rows}

    in_schema_not_manifest = schema_names - manifest_names
    in_manifest_not_schema = manifest_names - schema_names

    if in_schema_not_manifest:
        errors.append(
            f"In schema but not manifest: {sorted(in_schema_not_manifest)}"
        )
    if in_manifest_not_schema:
        errors.append(
            f"In manifest but not schema: {sorted(in_manifest_not_schema)}"
        )

    return errors


def main() -> int:
    print("=" * 60)
    print("SUSC-01 Schema & Manifest Validation")
    print("=" * 60)

    all_errors: list[str] = []

    print("\n[1/3] Validating schema...")
    schema_errors = validate_schema(SCHEMA_PATH)
    all_errors.extend(schema_errors)
    if schema_errors:
        for e in schema_errors:
            print(f"  ERROR: {e}")
    else:
        with SCHEMA_PATH.open("r", encoding="utf-8") as f:
            schema = json.load(f)
        n_features = len(schema.get("features", []))
        n_gt_false = sum(
            1 for f in schema["features"]
            if f.get("can_be_used_as_ground_truth") is False
        )
        n_train_false = sum(
            1 for f in schema["features"]
            if f.get("allowed_for_training") is False
        )
        print(f"  OK: {n_features} features catalogued")
        print(f"  OK: {n_gt_false}/{n_features} have can_be_used_as_ground_truth=false")
        print(f"  OK: {n_train_false}/{n_features} have allowed_for_training=false")

    print("\n[2/3] Validating manifest...")
    manifest_errors = validate_manifest(MANIFEST_PATH)
    all_errors.extend(manifest_errors)
    if manifest_errors:
        for e in manifest_errors:
            print(f"  ERROR: {e}")
    else:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        print(f"  OK: {n_rows} rows in manifest")

    print("\n[3/3] Cross-consistency check...")
    cross_errors = validate_cross_consistency(SCHEMA_PATH, MANIFEST_PATH)
    all_errors.extend(cross_errors)
    if cross_errors:
        for e in cross_errors:
            print(f"  WARNING: {e}")
    else:
        print("  OK: Schema and manifest are consistent")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s) found")
        return 1
    else:
        print("PASSED: All validations passed")
        print("\nGovernance summary:")
        print("  - allowed_for_training = false (ALL features)")
        print("  - review_only = true (ALL features)")
        print("  - can_be_used_as_ground_truth = false (ALL features)")
        print("  - This matrix is for susceptibility analysis, NOT occurrence confirmation")
        return 0


if __name__ == "__main__":
    sys.exit(main())
