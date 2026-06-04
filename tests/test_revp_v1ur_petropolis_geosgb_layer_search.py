import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env

def test_geosgb_layer_search_separates_context_from_candidate(tmp_path, monkeypatch):
    set_env(tmp_path,monkeypatch)
    rows=common.run_geosgb_layer_search(allow_web=True, services_fixture="tests/fixtures/v1ur/geosgb_services.json", layers_fixture="tests/fixtures/v1ur/geosgb_layers.json")
    assert any(r["layer_class"]=="OBSERVED_OR_FIELD_MAPPING_LAYER_CANDIDATE" for r in rows)
    assert any(r["layer_class"]=="RISK_CONTEXT_LAYER" for r in rows)
