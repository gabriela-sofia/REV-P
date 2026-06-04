from tests.test_revp_v2ae_canonical_region_registry_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ae_common as common


def test_event_registry_contains_expected_events(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_canonical_event_registry_builder(common.parse_args([]))
    by_event = {r["event_id"]: r for r in rows}
    for eid in ("REC_2022_05_24_30", "PET_2022_02_15", "PET_2024_03_21_28", "CUR_2022_01_15"):
        assert eid in by_event
    # Dates derived from event ids / linkage, never invented arbitrarily.
    assert by_event["REC_2022_05_24_30"]["start_date"] == "2022-05-24"
    assert by_event["REC_2022_05_24_30"]["end_date"] == "2022-05-30"
    assert by_event["PET_2024_03_21_28"]["start_date"] == "2024-03-21"
    # No event is promoted; occurrence/geometry support absent.
    assert all(r["canonical_event_status"] == "EVENT_CANDIDATE_NON_OPERATIONAL" for r in rows)
    assert all(r["occurrence_coordinate_support"] == "ABSENT" for r in rows)
    assert all(r["observed_geometry_support"] == "ABSENT" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
