import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_bitstream_resolver_detects_format_and_does_not_download(tmp_path, monkeypatch):
    data,_,_,_=set_env(tmp_path,monkeypatch)
    write_csv(data/"v1ur_petropolis_rigeo_related_item_registry.csv", common.RELATED_COLUMNS, [{"related_item_id":"r","event_id":"PET_2022_02_15","seed_id":"s","item_url":"https://rigeo.sgb.gov.br/handle/doc/22668","title":"Petropolis","year":"2022","relation_type":"KNOWN","matched_terms_hash":"h","is_public":"true","event_specificity":"EVENT_SPECIFIC_PETROPOLIS_2022","candidate_relevance":"HIGH","has_bitstreams":"true","notes":""}])
    rows=common.run_sgb_bitstream_deep_resolver(allow_web=True, fixture_html="tests/fixtures/v1ur/rigeo_related_item.html")
    assert any(r["format_hint"]=="zip" and r["is_geodata_candidate"]=="true" for r in rows)
