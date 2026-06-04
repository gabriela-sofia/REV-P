from tests.test_revp_v1uv_curitiba_source_target_builder import set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_geocuritiba_resolver_does_not_promote_context_layer(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    common.run_source_target_builder(common.parse_args([]))
    rows = common.run_geocuritiba_resolver(common.parse_args(["--dry-run"]))
    assert rows
    assert all(r["event_specificity"] == "CONTEXT_ONLY_NOT_EVENT" for r in rows)
    assert all(r["can_contain_observed_geometry"] == "false" for r in rows)
