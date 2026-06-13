"""v2az - minimal real-geometry handoff tests."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "datasets" / "manual_intake" / "recife_p1"


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_minimal_handoff_selects_one_auditable_recife_pair_without_geometry():
    patch = read(MANUAL / "minimal_turning_point_candidate_patch.csv")[0]
    event = read(MANUAL / "minimal_turning_point_candidate_event.csv")[0]
    pair = read(MANUAL / "minimal_turning_point_pairing.csv")[0]
    assert patch["patch_id"] == pair["patch_id"] == "REC_00019"
    assert event["event_id"] == pair["event_id"] == "REC_2022_05_24_30"
    assert pair["package_id"] == "PKG_34713b8aab96"
    assert patch["source_type"] == event["source_type"] == "missing"
    assert not patch["geometry_value"] and not event["geometry_value"]
    assert pair["pair_ready"] == "false"
