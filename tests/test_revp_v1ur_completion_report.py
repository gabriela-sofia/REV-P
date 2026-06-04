import os
import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_completion_report_writes_docs_manifest_next_actions(tmp_path, monkeypatch):
    data,docs,cfg,_=set_env(tmp_path,monkeypatch)
    monkeypatch.setattr(common,"V1UR_ARTIFACTS",[])
    for name,cols in [("v1ur_petropolis_geodata_signal_seed_registry.csv",common.SEED_COLUMNS),("v1ur_petropolis_rigeo_related_item_registry.csv",common.RELATED_COLUMNS),("v1ur_petropolis_sgb_bitstream_deep_registry.csv",common.BITSTREAM_COLUMNS),("v1ur_petropolis_geosgb_layer_search_registry.csv",common.LAYER_COLUMNS),("v1ur_petropolis_public_query_registry.csv",common.QUERY_COLUMNS),("v1ur_petropolis_candidate_url_registry.csv",common.CANDIDATE_COLUMNS),("v1ur_petropolis_geodata_download_manifest.csv",common.DOWNLOAD_COLUMNS),("v1ur_petropolis_geodata_inventory.csv",common.INVENTORY_COLUMNS),("v1ur_petropolis_geodata_candidate_audit.csv",common.AUDIT_COLUMNS)]:
        write_csv(data/name,cols,[])
    write_csv(data/"v1ur_petropolis_event_status_registry.csv", common.EVENT_STATUS_COLUMNS, [{"event_id":"PET_2022_02_15","v1ur_status":"RISK_CONTEXT_LAYER_ONLY","has_public_path_candidate":"true","has_downloaded_artifact":"false","has_geodata_inventory":"false","has_observed_geometry":"false","has_context_layer_only":"true","ground_truth_operational":"false","can_create_ground_reference":"false","can_create_training_label":"false","can_advance_to_overlay_preflight":"false","main_blocker":"CONTEXT","recommended_next_action":"v1us - Petropolis Risk/Susceptibility Context Layer Consolidation","notes":""}])
    result=common.run_completion_report()
    assert result["next_action"]=="v1us - Petropolis Risk/Susceptibility Context Layer Consolidation"
    assert os.path.exists(docs/"protocolo_c_status_atual_v1ur.md")
