import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_candidate_url_classifier_deduplicates_urls(tmp_path, monkeypatch):
    data,_,_,_=set_env(tmp_path,monkeypatch)
    url="https://rigeo.sgb.gov.br/a.geojson"
    write_csv(data/"v1ur_petropolis_sgb_bitstream_deep_registry.csv", common.BITSTREAM_COLUMNS, [{"bitstream_record_id":"b1","event_id":"PET_2022_02_15","item_url":"","bitstream_url":url,"bitstream_name":"a.geojson","format_hint":"geojson","content_length":"","is_public":"true","is_geodata_candidate":"true","is_pdf_only":"false","event_specificity":"x","blocking_reason":"","notes":""},{"bitstream_record_id":"b2","event_id":"PET_2022_02_15","item_url":"","bitstream_url":url,"bitstream_name":"a.geojson","format_hint":"geojson","content_length":"","is_public":"true","is_geodata_candidate":"true","is_pdf_only":"false","event_specificity":"x","blocking_reason":"","notes":""}])
    write_csv(data/"v1ur_petropolis_geosgb_layer_search_registry.csv", common.LAYER_COLUMNS, [])
    write_csv(data/"v1ur_petropolis_public_query_registry.csv", common.QUERY_COLUMNS, [])
    rows=common.run_candidate_url_classifier()
    assert len(rows)==1
