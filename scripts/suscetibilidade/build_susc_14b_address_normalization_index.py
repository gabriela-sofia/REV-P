"""SUSC-14B FASE 4 - robust address normalization audit/index."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv  # noqa: E402

OCC = ROOT / "datasets" / "suscetibilidade" / "susc_14a_geocoded_official_occurrences_v1.csv"
FEATURES = ROOT / "datasets" / "suscetibilidade" / "susc_14b_official_spatial_reference_features_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14B_address_normalization_audit.csv"


def split_address(text: str) -> dict:
    raw = str(text or "")
    normalized = g14.normalize_street(raw)
    number_match = re.search(r"\b(\d+[A-Za-z]?)\b", g14.strip_accents(raw))
    number = number_match.group(1) if number_match else ""
    tokens = [t for t in normalized.split() if t not in {"rua", "avenida", "travessa", "estrada"}]
    complement = ""
    if "," in raw:
        complement = raw.split(",", 1)[1].strip()[:80]
    return {
        "raw": raw,
        "normalized_street": normalized,
        "number": number,
        "complement": complement,
        "tokens": "|".join(tokens),
        "first_tokens": " ".join(tokens[:3]),
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-14B Address Normalization Index")
    print("=" * 60)
    occs = read_csv(OCC) if OCC.exists() else []
    refs = read_csv(FEATURES) if FEATURES.exists() else []
    occ_keys = Counter()
    ref_keys = Counter()
    numbered = complements = 0
    for row in occs:
        parsed = split_address(row.get("raw_address") or row.get("normalized_street", ""))
        hood = g14.normalize_text(row.get("raw_neighborhood") or row.get("normalized_neighborhood", ""))
        if parsed["number"]:
            numbered += 1
        if parsed["complement"]:
            complements += 1
        occ_keys[f"{hood}|{parsed['first_tokens']}"] += 1
    for row in refs:
        key = f"{row.get('normalized_neighborhood','')}|{' '.join(str(row.get('normalized_name','')).split()[:3])}"
        ref_keys[key] += 1
    overlap = sum(1 for key in occ_keys if key in ref_keys)
    rows = [
        {"metric": "occurrence_rows", "value": len(occs), "review_only": "true"},
        {"metric": "reference_rows", "value": len(refs), "review_only": "true"},
        {"metric": "occurrence_blocking_keys", "value": len(occ_keys), "review_only": "true"},
        {"metric": "reference_blocking_keys", "value": len(ref_keys), "review_only": "true"},
        {"metric": "overlapping_blocking_keys", "value": overlap, "review_only": "true"},
        {"metric": "occurrences_with_number_detected", "value": numbered, "review_only": "true"},
        {"metric": "occurrences_with_complement_detected", "value": complements, "review_only": "true"},
        {"metric": "normalization_rules", "value": "strip_accents|uppercase_equivalent|street_abbrev|punctuation|number|complement|tokens|difflib", "review_only": "true"},
    ]
    write_csv(AUDIT, rows, ["metric", "value", "review_only"])
    print(f"audit -> {AUDIT.relative_to(ROOT)} | overlap keys {overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
