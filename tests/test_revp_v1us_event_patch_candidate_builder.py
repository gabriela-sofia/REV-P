from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def test_candidates_are_region_only_and_candidate_only(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    rows = build_chain(data, root)
    # REC(2) + PET_2022(2) + PET_2024(2) + CUR blocked(1)
    assert len(rows) == 7
    assert all(r["event_patch_candidate_only"] == "true" for r in rows)
    assert all(r["patch_bound_truth"] == "false" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
    linked = [r for r in rows if r["patch_id"]]
    assert all(r["linkage_basis"] == "REGION_ONLY_CANDIDATE_NO_SPATIAL_DISTANCE" for r in linked)
    assert all(r["linkage_status"] == "CANDIDATE_NON_OPERATIONAL" for r in linked)


def test_curitiba_blocked_without_clear_event(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    rows = build_chain(data, root)
    cur = [r for r in rows if r["region"] == "CUR"]
    assert len(cur) == 1
    assert cur[0]["patch_id"] == ""  # patches not linked to a missing event
    assert cur[0]["linkage_status"] == "BLOCKED_NO_CLEAR_EVENT"
    assert cur[0]["blocker"] == "CURITIBA_EVENT_REGISTRY_MISSING"
