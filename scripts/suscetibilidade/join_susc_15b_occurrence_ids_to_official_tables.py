"""SUSC-15B step 5: join occurrence IDs to official tables when possible."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402

FIELDS = [
    "event_id", "source_event_id", "identifier_available", "official_table_match",
    "matched_source_id", "precision_tier_unlocked", "decision", "review_only",
]


def main() -> int:
    rows = []
    for row in c.read_csv(c.GEOCODED_15A):
        source_event_id = row.get("source_event_id", "")
        has_id = bool(source_event_id)
        rows.append({
            "event_id": row.get("event_id", ""),
            "source_event_id": source_event_id,
            "identifier_available": c.bool_text(has_id),
            "official_table_match": "false",
            "matched_source_id": "",
            "precision_tier_unlocked": "",
            "decision": "identifier_retained_but_no_official_auxiliary_match" if has_id else "blocked_no_identifier",
            "review_only": "true",
        })
    c.write_csv(c.JOIN_AUDIT, rows, FIELDS)
    print(f"join audit rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
