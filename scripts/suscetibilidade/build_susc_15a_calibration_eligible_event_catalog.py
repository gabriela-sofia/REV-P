"""SUSC-15A FASE 8 - build calibration-eligible precision event catalog."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, sha256_file, write_csv, write_json  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
GEOCODED = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
CATALOG = DAT / "susc_15a_calibration_eligible_event_catalog_v1.csv"
MANIFEST = MAN / "susc_15a_calibration_eligible_event_catalog_manifest_v1.json"
AUDIT = OUT / "SUSC_15A_calibration_eligible_event_catalog_audit.csv"

FIELDS = [
    "event_id", "region", "event_date", "event_type", "precision_tier", "lat", "lon",
    "wkt", "uncertainty_m", "calibration_eligible", "eligibility_reason", "review_only",
]
ELIGIBLE_TIERS = {
    "T0_official_event_polygon",
    "T1_official_event_point",
    "T2_official_address_point_or_parcel",
    "T3_official_intersection_point",
}


def _eligible(row: dict) -> tuple[bool, str]:
    tier = row.get("precision_tier", "")
    if row.get("calibration_eligible") != "true":
        return False, "source_geocoding_not_eligible"
    if tier in ELIGIBLE_TIERS:
        if row.get("lat") and row.get("lon") or row.get("wkt") or row.get("bbox") or row.get("geojson_ref"):
            return True, "precise_official_geometry_tier"
        return False, "eligible_tier_missing_geometry"
    if tier == "T4_official_house_number_linear_reference":
        try:
            if float(row.get("uncertainty_m", "9999")) <= 100 and row.get("is_ambiguous") != "true":
                return True, "linear_reference_uncertainty_lte_100m"
        except ValueError:
            pass
    return False, "tier_excluded_from_calibration"


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Calibration-Eligible Event Catalog")
    print("=" * 60)
    selected, audit = [], []
    for row in read_csv(GEOCODED) if GEOCODED.exists() else []:
        ok, reason = _eligible(row)
        audit.append({"event_id": row.get("event_id", ""), "precision_tier": row.get("precision_tier", ""),
                      "calibration_eligible_input": row.get("calibration_eligible", ""),
                      "included": str(ok).lower(), "eligibility_reason": reason, "review_only": "true"})
        if ok:
            selected.append({
                "event_id": row.get("event_id", ""),
                "region": row.get("region", ""),
                "event_date": row.get("event_date", ""),
                "event_type": row.get("event_type", ""),
                "precision_tier": row.get("precision_tier", ""),
                "lat": row.get("lat", ""),
                "lon": row.get("lon", ""),
                "wkt": row.get("wkt", ""),
                "uncertainty_m": row.get("uncertainty_m", ""),
                "calibration_eligible": "true",
                "eligibility_reason": reason,
                "review_only": "true",
            })
    write_csv(CATALOG, selected, FIELDS)
    write_csv(AUDIT, audit, ["event_id", "precision_tier", "calibration_eligible_input", "included", "eligibility_reason", "review_only"])
    write_json(MANIFEST, {
        "artifact": "SUSC-15A calibration-eligible event catalog",
        "source_rows": len(audit),
        "eligible_rows": len(selected),
        "precision_tier_counts": dict(Counter(r["precision_tier"] for r in selected)),
        "sha256": sha256_file(CATALOG),
        "allowed_for_training": False,
        "can_be_ground_truth": False,
        "can_be_used_as_ground_truth": False,
        "review_only": True,
    })
    print(f"eligible events: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
