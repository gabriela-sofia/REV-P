from tests.test_revp_v1uv_curitiba_source_target_builder import install_candidate_discovery, set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_defesa_civil_pr_resolver_separates_event_from_generic_alert(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_candidate_discovery(data, official=True, date="2022-01-15", hazard="alagamento")
    rows = common.run_defesa_civil_pr_resolver(common.parse_args([]))
    assert rows == []  # fixture source is municipal, not PR resolver
    install_candidate_discovery(data, official=True, date="", hazard="alagamento")
    rows = common.run_defesa_civil_pr_resolver(common.parse_args([]))
    assert rows == []
