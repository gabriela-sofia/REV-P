import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def test_phenomenon_status_is_never_a_label(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_phenomenon_status_attacher()
    assert all(r["is_observed_label"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)


def test_petropolis_2022_partial_textual(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_phenomenon_status_attacher()
    pet22 = [r for r in rows if r["event_id"] == "PET_2022_02_15"]
    assert pet22 and all(r["phenomenon_support"] == "PARTIAL_TEXTUAL" for r in pet22)
    rec = [r for r in rows if r["event_id"] == "REC_2022_05_24_30"]
    assert rec and all(r["phenomenon_class"] == "URBAN_FLOOD" for r in rec)
