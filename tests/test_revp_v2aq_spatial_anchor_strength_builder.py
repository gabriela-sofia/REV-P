import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_anchor_strength_derivation_rules(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    rows = common.run_spatial_anchor_strength_builder(common.parse_args([]))
    assert len(rows) == 9
    by = {r["candidate_id"]: r for r in rows}
    # map/point anchor allows derivation; broad municipio does not
    assert by["PET_2022_02_15"]["geometry_derivation_allowed"] == "true"
    assert by["PET_2024_03_21_28"]["geometry_derivation_allowed"] == "false"
    assert all(r["manual_digitization_required"] == "true" for r in rows)
