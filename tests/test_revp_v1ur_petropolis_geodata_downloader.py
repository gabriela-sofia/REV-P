import os
import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_downloader_uses_safe_filename_and_no_overwrite(tmp_path, monkeypatch):
    data,_,_,raw=set_env(tmp_path,monkeypatch)
    write_csv(data/"v1ur_petropolis_candidate_url_registry.csv", common.CANDIDATE_COLUMNS, [{"candidate_url_id":"c","event_id":"PET_2022_02_15","source_registry":"test","source_record_id":"r","url":"https://rigeo.sgb.gov.br/a.geojson","url_sha1_12":"abc123","candidate_class":"GEODATA_PACKAGE_CANDIDATE","artifact_type":"geojson","event_specificity":"PET","phenomenon_specificity":"x","download_priority":"1","can_contain_geometry":"true","can_contain_observed_occurrence":"true","can_contain_context_only":"false","blocking_reason":"","notes":""}])
    monkeypatch.setattr(common,"fetch_bytes",lambda *a,**k:(b'{"type":"FeatureCollection","features":[]}',"application/geo+json",""))
    rows=common.run_geodata_downloader(allow_web=True, download=True)
    assert rows[0]["download_status"]=="DOWNLOADED"
    assert os.path.exists(raw/rows[0]["safe_filename"])
