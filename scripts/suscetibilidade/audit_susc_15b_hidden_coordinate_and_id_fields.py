"""SUSC-15B step 1: audit hidden coordinate and identifier fields."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402

FIELDS = [
    "source_path", "field_name", "field_kind", "non_empty_rows", "sample_value",
    "precision_candidate", "calibration_eligible", "decision", "review_only",
]
COORD_RE = re.compile(r"(lat|lon|lng|coord|utm|x_|y_|wkt|geo|bbox)", re.I)
ID_RE = re.compile(r"(id|codigo|cod_|protocolo|ocorr|chamado|processo)", re.I)
SCAN_FILES = [
    c.GEOCODED_15A,
    c.DAT / "susc_14a_normalized_official_occurrences_v1.csv",
    c.DAT / "susc_14b_matched_official_occurrences_v1.csv",
    c.DAT / "susc_14c_matched_official_occurrences_v1.csv",
    c.DAT / "susc_14c_resolved_ambiguous_occurrences_v1.csv",
]


def classify(name: str) -> str:
    if COORD_RE.search(name):
        return "coordinate_like"
    if ID_RE.search(name):
        return "identifier_like"
    return "other"


def main() -> int:
    rows = []
    for path in SCAN_FILES:
        table = c.read_csv(path)
        if not table:
            continue
        fields = list(table[0].keys())
        for field in fields:
            kind = classify(field)
            if kind == "other":
                continue
            values = [str(row.get(field, "")).strip() for row in table if str(row.get(field, "")).strip()]
            sample = values[0] if values else ""
            precision_candidate = kind == "coordinate_like" and bool(values)
            eligible = False
            decision = "blocked_no_precise_official_geometry"
            if precision_candidate and field in {"lat", "lon", "wkt", "bbox", "geojson_ref"}:
                decision = "already_audited_by_15a_not_new_hidden_field"
            elif kind == "identifier_like" and values:
                decision = "identifier_available_for_manual_official_join"
            rows.append({
                "source_path": str(path.relative_to(c.ROOT)).replace("\\", "/"),
                "field_name": field,
                "field_kind": kind,
                "non_empty_rows": len(values),
                "sample_value": sample[:120],
                "precision_candidate": c.bool_text(precision_candidate),
                "calibration_eligible": c.bool_text(eligible),
                "decision": decision,
                "review_only": "true",
            })
    c.write_csv(c.HIDDEN_AUDIT, rows, FIELDS)
    print(f"hidden/id audit rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
