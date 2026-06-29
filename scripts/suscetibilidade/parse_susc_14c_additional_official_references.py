"""SUSC-14C FASE 3a - parse additional official reference downloads."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import parse_susc_14b_official_spatial_references as p14b  # noqa: E402
from susc_io import read_csv, write_csv  # noqa: E402

MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14c_additional_reference_download_manifest_v1.csv"
FEATURES = ROOT / "datasets" / "suscetibilidade" / "susc_14c_additional_official_reference_features_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_additional_reference_parse_audit.csv"
MAX_FEATURES_TOTAL = 50000


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Additional Official Reference Parser")
    print("=" * 60)
    features, audit = [], []
    for row in read_csv(MANIFEST) if MANIFEST.exists() else []:
        fp = row.get("file_path", "")
        sid = row.get("reference_id", "")
        if not fp:
            audit.append({"source_id": sid, "source_file": "", "parse_status": "no_file", "n_features": 0,
                          "by_feature_type": "{}", "review_only": "true", "notes": row.get("blocked_reason", "")})
            continue
        path = ROOT / fp
        if not path.exists():
            audit.append({"source_id": sid, "source_file": fp, "parse_status": "missing", "n_features": 0,
                          "by_feature_type": "{}", "review_only": "true", "notes": "file not found"})
            continue
        if path.suffix.lower() in {".geojson", ".json"}:
            parsed = p14b._parse_geojson(path, sid, row.get("source_title", ""))
        elif path.suffix.lower() in {".csv", ".tsv"}:
            parsed = p14b._parse_csv(path, sid, row.get("source_title", ""), row.get("region", ""))
        else:
            parsed = []
        slots = MAX_FEATURES_TOTAL - len(features)
        if slots > 0:
            features.extend(parsed[:slots])
        counts = Counter(r["feature_type"] for r in parsed)
        audit.append({"source_id": sid, "source_file": fp,
                      "parse_status": "parsed" if parsed else "no_supported_features",
                      "n_features": len(parsed), "by_feature_type": dict(counts),
                      "review_only": "true", "notes": "additional official reference parsing review-only"})
        if len(features) >= MAX_FEATURES_TOTAL:
            break
    for idx, row in enumerate(features):
        row["reference_feature_id"] = f"S14CFEAT_{idx:06d}"
    write_csv(FEATURES, [{k: row.get(k, "") for k in p14b.FIELDS} for row in features], p14b.FIELDS)
    write_csv(AUDIT, audit, p14b.AUDIT_FIELDS)
    print(f"features: {len(features)} | {dict(Counter(r['feature_type'] for r in features))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
