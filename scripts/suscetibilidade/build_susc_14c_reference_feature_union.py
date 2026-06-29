"""SUSC-14C FASE 3b - build deduplicated reference feature union."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import parse_susc_14b_official_spatial_references as p14b  # noqa: E402
from susc_io import read_csv, sha256_file, write_csv, write_json  # noqa: E402

FEATURES_14B = ROOT / "datasets" / "suscetibilidade" / "susc_14b_official_spatial_reference_features_v1.csv"
FEATURES_14C_ADD = ROOT / "datasets" / "suscetibilidade" / "susc_14c_additional_official_reference_features_v1.csv"
UNION = ROOT / "datasets" / "suscetibilidade" / "susc_14c_official_spatial_reference_features_union_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14c_reference_feature_union_manifest_v1.json"
MAX_UNION = 130000


def _key(row: dict) -> tuple:
    geom = row.get("bbox") or row.get("wkt") or row.get("geojson_ref")
    if not geom and (row.get("lat") or row.get("lon")):
        geom = f"{row.get('lat','')},{row.get('lon','')}"
    return (
        row.get("region", ""),
        row.get("feature_type", ""),
        row.get("normalized_name", ""),
        row.get("normalized_neighborhood", "") or row.get("neighborhood", ""),
        geom,
        row.get("source_id", ""),
    )


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Reference Feature Union")
    print("=" * 60)
    rows = []
    for source, path in (("14b", FEATURES_14B), ("14c_additional", FEATURES_14C_ADD)):
        for row in read_csv(path) if path.exists() else []:
            row = dict(row)
            row["union_source"] = source
            rows.append(row)
    dedup = {}
    for row in rows:
        dedup.setdefault(_key(row), row)
    union = list(dedup.values())[:MAX_UNION]
    for idx, row in enumerate(union):
        row["reference_feature_id"] = f"S14CUNION_{idx:06d}"
    fields = list(p14b.FIELDS)
    if "union_source" not in fields:
        fields.append("union_source")
    write_csv(UNION, [{k: row.get(k, "") for k in fields} for row in union], fields)
    manifest = {
        "artifact": "SUSC-14C reference feature union",
        "input_14b_rows": len(read_csv(FEATURES_14B)),
        "input_14c_additional_rows": len(read_csv(FEATURES_14C_ADD)) if FEATURES_14C_ADD.exists() else 0,
        "union_rows": len(union),
        "deduplicated_rows": len(rows) - len(union),
        "feature_type_counts": dict(Counter(r.get("feature_type", "") for r in union)),
        "sha256": sha256_file(UNION),
        "allowed_for_training": False,
        "can_be_ground_truth": False,
        "can_be_used_as_ground_truth": False,
        "review_only": True,
    }
    write_json(MANIFEST, manifest)
    print(f"union rows: {len(union)} | {manifest['feature_type_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
