"""SUSC-15B step 6: build forensic precision event catalog."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402

FIELDS = [
    "event_id", "source_event_id", "region", "event_date", "event_type",
    "precision_tier", "lat", "lon", "wkt", "uncertainty_m",
    "forensic_source_status", "calibration_eligible", "eligibility_reason",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training",
    "review_only",
]


def main() -> int:
    join_by_event = {row.get("event_id", ""): row for row in c.read_csv(c.JOIN_AUDIT)}
    rows = []
    for row in c.read_csv(c.GEOCODED_15A):
        join = join_by_event.get(row.get("event_id", ""), {})
        eligible = c.is_calibration_eligible(row) and join.get("official_table_match") == "true"
        tier = row.get("precision_tier", "")
        if eligible:
            reason = "official_precise_geometry_confirmed_by_15b"
        elif tier in c.WEAK_TIERS:
            reason = "weak_or_context_tier_excluded_from_calibration"
        else:
            reason = "no_new_forensic_official_match"
        rows.append({
            "event_id": row.get("event_id", ""),
            "source_event_id": row.get("source_event_id", ""),
            "region": row.get("region", ""),
            "event_date": row.get("event_date", ""),
            "event_type": row.get("event_type", ""),
            "precision_tier": tier,
            "lat": row.get("lat", "") if eligible else "",
            "lon": row.get("lon", "") if eligible else "",
            "wkt": row.get("wkt", "") if eligible else "",
            "uncertainty_m": row.get("uncertainty_m", "") if eligible else "",
            "forensic_source_status": join.get("decision", "no_join_audit"),
            "calibration_eligible": c.bool_text(eligible),
            "eligibility_reason": reason,
            "can_be_ground_truth": "false",
            "can_be_used_as_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    c.write_csv(c.CATALOG, rows, FIELDS)
    print(f"15B event catalog rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
