import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def test_temporal_linker_never_invents_sentinel_date(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_event_temporal_window_linker()
    assert all(r["sentinel_scene_date"] == "" for r in rows)
    assert all(r["has_sentinel_date"] == "false" for r in rows)
    linked = [r for r in rows if r["patch_id"]]
    assert all(r["temporal_linkage_class"] == "SENTINEL_DATE_MISSING" for r in linked)
    blocked = [r for r in rows if not r["patch_id"]]
    assert all(r["temporal_linkage_class"] == "PATCH_EVENT_LINKAGE_NOT_AVAILABLE" for r in blocked)


def test_temporal_linker_carries_event_window(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_event_temporal_window_linker()
    rec = next(r for r in rows if r["event_id"] == "REC_2022_05_24_30")
    assert rec["event_start_date"] == "2022-05-24"
    assert rec["event_end_date"] == "2022-05-30"
