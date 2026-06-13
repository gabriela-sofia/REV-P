"""v2bd recovered boundary candidate and feed tests."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_recovered_candidate_is_reprojected_and_requires_review():
    path = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "derived" / "patch_boundary_REC_00019_from_lineage.geojson"
    obj = json.loads(path.read_text(encoding="utf-8"))
    candidate = read("v2bd_REC_00019_patch_boundary_candidate_registry.csv")[0]
    feed = read("v2bd_ready_patch_boundary_feed.csv")[0]
    assert obj["properties"]["source_crs"] == "EPSG:32725"
    assert obj["properties"]["requires_human_review"] is True
    assert obj["properties"]["can_be_ground_truth"] is False
    assert candidate["crs"] == feed["crs"] == "EPSG:4326"
    assert feed["requires_human_review"] == "true"
    assert feed["ready_for_v2av"] == feed["ready_for_v2az"] == "true"


def test_certificate_records_tp1_candidate():
    cert = read("v2bd_REC_00019_footprint_recovery_certificate.csv")[0]
    assert cert["status"] == "BOUNDARY_RECOVERED_FROM_LINEAGE"
    assert cert["boundary_recovered"] == "true"
    assert cert["turning_point_unlocked"] == "true"
