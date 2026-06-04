import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def test_dino_support_is_review_only(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_dino_review_support_attacher()
    assert all(r["dino_usage"] == "SUPPORT_ONLY" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
    linked = [r for r in rows if r["patch_id"]]
    assert all(r["dino_review_support_status"] == "DINO_REVIEW_SUPPORT_AVAILABLE" for r in linked)
    cur = [r for r in rows if not r["patch_id"]]
    assert all(r["dino_review_support_status"] == "DINO_NOT_APPLICABLE" for r in cur)


def test_dino_registry_missing_recorded(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    # build candidates from event registry only, without a DINO/patch registry
    from tests.test_revp_v1us_patch_registry_resolver import install_event_registry
    install_event_registry(data)
    common.run_patch_registry_resolver()      # PATCH_REGISTRY_MISSING
    common.run_event_patch_candidate_builder()
    rows = common.run_dino_review_support_attacher()
    assert all(r["dino_review_support_status"] in {"DINO_NOT_APPLICABLE", "DINO_REGISTRY_MISSING"} for r in rows)
