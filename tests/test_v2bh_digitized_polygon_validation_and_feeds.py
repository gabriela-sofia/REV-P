"""v2bh candidate validation and feed tests."""

import csv
import json
import sys
from pathlib import Path

from shapely.geometry import Polygon, shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v2bh_charter758_recife_product_georeferencing_digitization as engine  # noqa: E402


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_candidate_is_valid_normalized_and_feeds_require_review():
    candidate = read("v2bh_georeferenced_event_polygon_candidate_registry.csv")[0]
    assert candidate["geometry_type"] == "MultiPolygon"
    assert candidate["crs"] == "EPSG:4326"
    assert candidate["is_patch_boundary_duplicate"] == "false"
    assert candidate["requires_human_review"] == "true"
    for suffix in ("v2ba", "v2aw", "v2au", "v2az"):
        feed = read(f"v2bh_ready_event_polygon_feed_for_{suffix}.csv")[0]
        assert feed["review_status"] == "provided_unreviewed"
        assert feed["source_method"] == "charter758_public_product_digitized_candidate"


def test_synthetic_polygon_passes_and_patch_boundary_duplicate_is_rejected():
    patch_path = ROOT / "datasets/external_sources/recife_minimal_tp/derived/patch_boundary_REC_00019_from_lineage.geojson"
    patch = shape(json.loads(patch_path.read_text(encoding="utf-8"))["geometry"])
    assert engine.validate_geometry(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), patch_path, {}) == (True, False)
    assert engine.validate_geometry(patch, patch_path, {}) == (True, True)
