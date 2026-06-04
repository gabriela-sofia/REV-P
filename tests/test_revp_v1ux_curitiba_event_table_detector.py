from tests.test_revp_v1ux_curitiba_download_target_builder import seed_fixture_downloads, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_event_table_detector_marks_candidates_without_labels(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    seed_fixture_downloads(data, raw)
    common.run_artifact_inventory(common.parse_args([]))
    common.run_schema_audit(common.parse_args([]))
    rows = common.run_event_table_detector(common.parse_args([]))
    assert any(r["event_table_class"] == "EVENT_OCCURRENCE_TABLE_CANDIDATE" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
