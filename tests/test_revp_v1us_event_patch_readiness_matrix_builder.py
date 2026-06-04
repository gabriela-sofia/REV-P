import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def _dims(rows, event_id):
    return {r["dimension"]: r["classification"] for r in rows if r["event_id"] == event_id}


def test_readiness_blocks_ground_reference_without_geometry(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    common.run_event_temporal_window_linker()
    rows = common.run_event_patch_readiness_matrix_builder()
    assert rows
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["ground_truth_operational"] == "false" for r in rows)
    rec = _dims(rows, "REC_2022_05_24_30")
    assert rec["geometry_support"] == "BLOCKED"
    assert rec["overlay_readiness"] == "BLOCKED"
    assert rec["ground_reference_readiness"] == "BLOCKED"
    assert rec["training_readiness"] == "BLOCKED"
    assert rec["coordinate_support"] == "BLOCKED"


def test_readiness_has_ten_dimensions_per_candidate(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    cand = build_chain(data, root)
    common.run_event_temporal_window_linker()
    rows = common.run_event_patch_readiness_matrix_builder()
    assert len(rows) == len(cand) * 10
    rec = _dims(rows, "REC_2022_05_24_30")
    assert rec["dino_review_support"] == "MODERATE"  # patches resolved from DINO registry
