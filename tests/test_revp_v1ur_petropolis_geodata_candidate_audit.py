import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_candidate_audit_does_not_promote_risk_to_occurrence(tmp_path, monkeypatch):
    data,_,_,_=set_env(tmp_path,monkeypatch)
    write_csv(data/"v1ur_petropolis_candidate_url_registry.csv", common.CANDIDATE_COLUMNS, [{"candidate_url_id":"c","event_id":"PET_2022_02_15","source_registry":"s","source_record_id":"r","url":"https://rigeo.sgb.gov.br/risco.geojson","url_sha1_12":"h","candidate_class":"RISK_CONTEXT_LAYER","artifact_type":"geojson","event_specificity":"PET","phenomenon_specificity":"x","download_priority":"1","can_contain_geometry":"true","can_contain_observed_occurrence":"false","can_contain_context_only":"true","blocking_reason":"","notes":""}])
    write_csv(data/"v1ur_petropolis_geodata_download_manifest.csv", common.DOWNLOAD_COLUMNS, [{"download_id":"d","event_id":"PET_2022_02_15","candidate_url_id":"c","url":"https://rigeo.sgb.gov.br/risco.geojson","safe_filename":"x","local_path_hash":"h","sha256":"","file_size_bytes":"","mime_type":"","extension":"geojson","download_status":"DOWNLOADED","duplicate_status":"","license_status":"","notes":""}])
    write_csv(data/"v1ur_petropolis_geodata_inventory.csv", common.INVENTORY_COLUMNS, [{"inventory_id":"i","event_id":"PET_2022_02_15","download_id":"d","asset_type":"geojson","container_type":"file","internal_path":"risco.geojson","has_geometry":"true","geometry_type":"Polygon","crs":"EPSG:4326","feature_count":"1","has_date_field":"false","has_phenomenon_field":"false","has_locality_field":"false","has_coordinate_fields":"false","is_pdf_only":"false","inventory_status":"INVENTORIED","notes":""}])
    rows=common.run_geodata_candidate_audit()
    assert rows[0]["candidate_class"]=="RISK_CONTEXT_LAYER"
    assert rows[0]["can_create_ground_reference"]=="false"
