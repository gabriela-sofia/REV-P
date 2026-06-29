"""SUSC-15B step 4: parse only available deep-source manifests fail-closed."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402

FIELDS = [
    "source_id", "parse_status", "parsed_rows", "precise_geometry_rows",
    "calibration_eligible_rows", "decision", "review_only",
]


def main() -> int:
    rows = []
    for source in c.read_csv(c.DOWNLOAD_MANIFEST):
        rows.append({
            "source_id": source.get("source_id", ""),
            "parse_status": "NO_LOCAL_PRECISION_SOURCE_TO_PARSE",
            "parsed_rows": 0,
            "precise_geometry_rows": 0,
            "calibration_eligible_rows": 0,
            "decision": "blocked_until_official_source_materialized",
            "review_only": "true",
        })
    c.write_csv(c.PARSE_AUDIT, rows, FIELDS)
    print(f"parse audit rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
