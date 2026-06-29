"""SUSC-15B step 3: create a fail-closed deep-source acquisition manifest."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402

FIELDS = [
    "source_id", "source_family", "target", "download_attempted", "download_status",
    "local_artifact_path", "failure_or_blocker", "could_unlock_tier", "review_only",
]
TARGETS = [
    ("S15B_SRC_001", "internal_occurrence_ids", "CSV occurrence ids/codes", "T2/T3/T4"),
    ("S15B_SRC_002", "hidden_coordinate_columns", "coordinate/UTM columns in existing tables", "T0/T1/T2"),
    ("S15B_SRC_003", "official_auxiliary_tables", "auxiliary municipal/state event tables", "T2/T3/T4"),
    ("S15B_SRC_004", "pdf_annex_points", "PDF/report annexes with flood points", "T1/T3"),
    ("S15B_SRC_005", "nonobvious_arcgis_layers", "ArcGIS/FeatureServer/WFS layers", "T0/T1/T2/T3"),
    ("S15B_SRC_006", "critical_flood_points", "official critical flood points and 156/Defesa Civil occurrences", "T1/T3"),
    ("S15B_SRC_007", "official_intersections", "official intersection gazetteers", "T3"),
    ("S15B_SRC_008", "house_numbering", "official address number ranges/parcels", "T2/T4"),
    ("S15B_SRC_009", "external_footprints", "Copernicus/EMS/SGB/INEA/APAC footprints", "T0"),
]


def main() -> int:
    rows = []
    for source_id, family, target, tier in TARGETS:
        rows.append({
            "source_id": source_id,
            "source_family": family,
            "target": target,
            "download_attempted": "false",
            "download_status": "BLOCKED_MANUAL_OR_EXPLICIT_LIVE_ACCESS_REQUIRED",
            "local_artifact_path": "",
            "failure_or_blocker": "No precise official local file or explicit live acquisition result available in this run.",
            "could_unlock_tier": tier,
            "review_only": "true",
        })
    c.write_csv(c.DOWNLOAD_MANIFEST, rows, FIELDS)
    print(f"deep source manifest rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
