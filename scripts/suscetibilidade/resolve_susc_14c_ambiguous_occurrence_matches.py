"""SUSC-14C FASE 5 - strict auto-resolution for ambiguous official matches."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv  # noqa: E402

MATCHED = ROOT / "datasets" / "suscetibilidade" / "susc_14c_matched_official_occurrences_v1.csv"
RESOLVED = ROOT / "datasets" / "suscetibilidade" / "susc_14c_resolved_ambiguous_occurrences_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_ambiguous_resolution_audit.csv"


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _t(v) -> bool:
    return str(v or "").strip().lower() == "true"


def _status_for(feature_type: str) -> tuple[str, str]:
    if feature_type in {"address_point", "flood_point"}:
        return "official_address_point_match", "moderate_official_occurrence_geocoded"
    return "official_street_segment_match", "moderate_official_street_segment_match"


def _candidate(row: dict) -> dict | None:
    try:
        payload = json.loads(row.get("candidate_summary") or "[]")
    except json.JSONDecodeError:
        return None
    return payload[0] if payload else None


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Ambiguous Match Resolution")
    print("=" * 60)
    rows = read_csv(MATCHED) if MATCHED.exists() else []
    out, audit = [], []
    for row in rows:
        row = dict(row)
        action = "carried_forward"
        if row.get("georeferencing_status") == "blocked_ambiguous_address":
            cand = _candidate(row)
            allowed = (
                cand is not None
                and _f(row.get("name_similarity")) >= 0.92
                and _t(row.get("neighborhood_match"))
                and _f(row.get("score_gap")) >= 0.08
                and _t(row.get("geometry_available"))
            )
            if allowed:
                status, level = _status_for(cand.get("type", "street_segment"))
                row["georeferencing_status"] = status
                row["evidence_level"] = level
                row["matched_reference_feature_id"] = cand.get("feature_id", "")
                row["matched_feature_type"] = cand.get("type", "")
                row["matched_reference_name"] = cand.get("name", "")
                row["unique_candidate"] = "true"
                row["match_method"] = "resolved_ambiguous_strict_official_candidate"
                row["match_confidence"] = "medium"
                action = "resolved_strict"
            else:
                row["evidence_level"] = "rejected_not_observed_event"
                row["match_confidence"] = "none"
                action = "kept_blocked"
        row["requires_manual_review"] = "true"
        row["can_be_ground_truth"] = "false"
        row["can_be_used_as_ground_truth"] = "false"
        row["allowed_for_training"] = "false"
        row["review_only"] = "true"
        out.append(row)
        audit.append({
            "matched_id": row.get("matched_id", ""),
            "occurrence_id": row.get("occurrence_id", ""),
            "input_status": "blocked_ambiguous_address" if action in {"resolved_strict", "kept_blocked"} else row.get("georeferencing_status", ""),
            "resolution_action": action,
            "output_status": row.get("georeferencing_status", ""),
            "name_similarity": row.get("name_similarity", ""),
            "neighborhood_match": row.get("neighborhood_match", ""),
            "score_gap": row.get("score_gap", ""),
            "geometry_available": row.get("geometry_available", ""),
            "review_only": "true",
        })
    write_csv(RESOLVED, out, list(rows[0].keys()) if rows else None)
    write_csv(AUDIT, audit, ["matched_id", "occurrence_id", "input_status", "resolution_action", "output_status",
                             "name_similarity", "neighborhood_match", "score_gap", "geometry_available", "review_only"])
    print(f"resolved rows: {len(out)} | {dict(Counter(a['resolution_action'] for a in audit))}")
    print("review-only. Ambiguous rows resolve only under strict official-reference criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
