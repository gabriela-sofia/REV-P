import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_public_query_runner_respects_allowlist(tmp_path, monkeypatch):
    data,_,_,_=set_env(tmp_path,monkeypatch)
    write_csv(data/"v1ur_petropolis_geodata_signal_seed_registry.csv", common.SEED_COLUMNS, [{"seed_id":"s1","event_id":"PET_2022_02_15","source_asset_id":"a","page_number":"1","signal_class":"DIGITAL_GEODATA_REFERENCE","signal_strength":"STRONG","referenced_artifact_type":"geodata","seed_terms":"Petropolis|SGB","source_context_hash":"h","priority":"1","target_source_family":"SGB_RIGEO","can_resolve_by_public_search":"true","can_create_ground_reference":"false","can_create_training_label":"false","notes":""}])
    rows=common.run_public_query_runner(allow_web=True, fixture_html="tests/fixtures/v1ur/public_query.html")
    assert rows
    assert all("evil.example" not in r["result_url"] for r in rows)
