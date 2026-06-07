import json

import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_geojson_null_when_no_explicit_geometry(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_event_geometry_candidate_builder(common.parse_args([]))
    index = common.run_geojson_candidate_exporter(common.parse_args([]))
    assert len(index) == 9
    # no candidate has explicit geometry in the synthetic stack -> all null
    assert all(r["geometry_present"] == "false" for r in index)
    for f in geojson.glob("*.geojson"):
        fc = json.load(open(f, encoding="utf-8"))
        for feat in fc["features"]:
            assert feat["geometry"] is None
            assert feat["properties"]["not_ground_truth"] is True
            assert feat["properties"]["patch_truth_allowed"] is False
