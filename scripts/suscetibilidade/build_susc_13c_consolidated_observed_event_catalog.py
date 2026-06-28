"""
SUSC-13C consolidated observed-event catalog (13A + 13B + 13C live, review-only).

Normalizes the three parsed catalogs into a common schema, flags likely
duplicates by region + date/period + approximate geometry, keeps the
higher-evidence record as preferred, and records lineage/upgrade. No record
becomes ground truth or training material.

Writes:
  - datasets/suscetibilidade/susc_13c_consolidated_observed_event_catalog_v1.csv
  - manifests/suscetibilidade/susc_13c_consolidated_catalog_manifest_v1.json
  - outputs_public/suscetibilidade/SUSC_13C_consolidation_audit.csv
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import build_susc_13b_auto_consolidated_event_catalog as c13b  # noqa: E402
from susc_io import read_csv, rel, write_csv, write_json  # noqa: E402

P13A = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_observed_events_parsed_v1.csv"
P13B = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_observed_events_parsed_v1.csv"
P13C = ROOT / "datasets" / "suscetibilidade" / "susc_13c_live_observed_events_parsed_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_13c_consolidated_observed_event_catalog_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13c_consolidated_catalog_manifest_v1.json"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_consolidation_audit.csv"

FIELDS = c13b.FIELDS[:-1] + ["evidence_upgrade_from_13a_or_13b", "consolidation_notes"]
AUDIT_FIELDS = c13b.AUDIT_FIELDS
RANK = c13b.EVIDENCE_RANK
LINEAGE_RANK = {"susc_13c_live": 3, "susc_13b_auto": 2, "susc_13a": 1}


def _norm_13c(r: dict) -> dict:
    d = c13b._norm_13b(r)
    d["source_lineage"] = "susc_13c_live"
    return d


def main() -> int:
    print("=" * 60)
    print("SUSC-13C Consolidated Event Catalog (13A+13B+13C)")
    print("=" * 60)
    recs: list[dict] = []
    if P13A.exists():
        recs.extend(c13b._norm_13a(r) for r in read_csv(P13A))
    if P13B.exists():
        recs.extend(c13b._norm_13b(r) for r in read_csv(P13B))
    if P13C.exists():
        recs.extend(_norm_13c(r) for r in read_csv(P13C))

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, rec in enumerate(recs):
        groups[c13b._dup_key(rec)].append(i)

    group_id_of: dict[int, str] = {}
    preferred_idx: set[int] = set()
    for gi, members in enumerate(m for _, m in sorted(groups.items(), key=lambda kv: str(kv[0]))):
        gid = f"DUPG_{gi:04d}"
        best = max(members, key=lambda i: (RANK.get(recs[i]["evidence_level"], 0),
                                           LINEAGE_RANK.get(recs[i]["source_lineage"], 0)))
        preferred_idx.add(best)
        for i in members:
            group_id_of[i] = gid

    rows: list[dict] = []
    audit: list[dict] = []
    for i, rec in enumerate(recs):
        gid = group_id_of[i]
        members = groups[c13b._dup_key(rec)]
        is_pref = i in preferred_idx
        is_dup = len(members) > 1
        upgrade = "false"
        if is_pref and is_dup and rec["source_lineage"] in {"susc_13b_auto", "susc_13c_live"}:
            if any(recs[j]["source_lineage"] == "susc_13a" and j != i for j in members):
                upgrade = "true"
        action = "keep_preferred" if is_pref else ("drop_duplicate_review" if is_dup else "keep_unique")
        event_id = f"SUSC13CC_{i:05d}"
        row = {
            "event_id": event_id, "source_lineage": rec["source_lineage"], "region": rec["region"],
            "municipality": rec["municipality"], "event_type": rec["event_type"],
            "event_date": rec["event_date"], "event_period_start": rec["event_period_start"],
            "event_period_end": rec["event_period_end"], "source_institution": rec["source_institution"],
            "source_title": rec["source_title"], "source_url": rec["source_url"],
            "local_file_path": rec["local_file_path"], "evidence_level": rec["evidence_level"],
            "geometry_type": rec["geometry_type"], "lat": rec["lat"], "lon": rec["lon"],
            "bbox": rec["bbox"], "wkt": rec["wkt"], "geojson_ref": rec["geojson_ref"],
            "crs": rec["crs"], "spatial_precision": rec["spatial_precision"],
            "temporal_precision": rec["temporal_precision"], "is_observed_event": rec["is_observed_event"],
            "source_confidence": rec["source_confidence"], "duplicate_group_id": gid,
            "duplicate_candidate": "true" if is_dup else "false", "deduplication_action": action,
            "preferred_record": "true" if is_pref else "false",
            "evidence_upgrade_from_13a_or_13b": upgrade,
            "requires_manual_review": "true", "can_be_ground_truth": "false",
            "allowed_for_training": "false", "review_only": "true",
            "consolidation_notes": f"origem={rec['source_lineage']}; orig_id={rec['orig_event_id']}",
        }
        rows.append(row)
        audit.append({
            "event_id": event_id, "source_lineage": rec["source_lineage"], "region": rec["region"],
            "evidence_level": rec["evidence_level"], "duplicate_group_id": gid,
            "duplicate_candidate": row["duplicate_candidate"], "deduplication_action": action,
            "preferred_record": row["preferred_record"], "review_only": "true",
            "notes": "upgrade_from_13a_or_13b" if upgrade == "true" else "consolidado review-only",
        })

    write_csv(CATALOG, rows, FIELDS)
    write_csv(AUDIT, audit, AUDIT_FIELDS)

    by_lineage = Counter(r["source_lineage"] for r in rows)
    by_level = Counter(r["evidence_level"] for r in rows)
    write_json(MANIFEST, {
        "artifact": "SUSC-13C consolidated observed event catalog (13A+13B+13C)",
        "n_records": len(rows), "n_groups": len(groups),
        "n_duplicate_candidates": sum(1 for r in rows if r["duplicate_candidate"] == "true"),
        "n_preferred_records": sum(1 for r in rows if r["preferred_record"] == "true"),
        "by_source_lineage": dict(by_lineage), "by_evidence_level": dict(by_level),
        "n_evidence_upgrades": sum(1 for r in rows if r["evidence_upgrade_from_13a_or_13b"] == "true"),
        "inputs": {"susc_13a": rel(P13A), "susc_13b_auto": rel(P13B), "susc_13c_live": rel(P13C)},
        "can_be_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "score_v7_created": False,
    })
    print(f"records: {len(rows)} | by lineage: {dict(by_lineage)}")
    print("by level:", dict(by_level))
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
