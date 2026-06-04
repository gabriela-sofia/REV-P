from tests.test_revp_v1uv_curitiba_source_target_builder import install_candidate_discovery, set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_hydromet_resolver_never_treats_rain_as_observed_occurrence(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_candidate_discovery(data, official=True, date="2022-01-15", hazard="chuva")
    rows = common.run_hydromet_source_resolver(common.parse_args([]))
    assert rows
    assert all(r["can_be_observed_occurrence"] == "false" for r in rows)
