import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_event_status_does_not_release_overlay_without_observed_geometry(tmp_path, monkeypatch):
    data,_,_,_=set_env(tmp_path,monkeypatch)
    write_csv(data/"v1ur_petropolis_candidate_url_registry.csv", common.CANDIDATE_COLUMNS, [])
    write_csv(data/"v1ur_petropolis_geodata_download_manifest.csv", common.DOWNLOAD_COLUMNS, [])
    write_csv(data/"v1ur_petropolis_geodata_inventory.csv", common.INVENTORY_COLUMNS, [])
    write_csv(data/"v1ur_petropolis_geodata_candidate_audit.csv", common.AUDIT_COLUMNS, [])
    rows=common.run_event_status_updater()
    assert all(r["can_advance_to_overlay_preflight"]=="false" for r in rows)
