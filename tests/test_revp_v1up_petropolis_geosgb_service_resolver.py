import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env


def test_geosgb_resolver_fail_closed_and_classifies_context(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    rows = common.run_geosgb_service_resolver()
    assert rows[0]["blocking_reason"] == "DRY_RUN_ENDPOINT_NOT_QUERIED"
    rows = common.run_geosgb_service_resolver(
        allow_web=True,
        services_fixture="tests/fixtures/v1up/arcgis_services.json",
        layers_fixture="tests/fixtures/v1up/arcgis_layers.json",
    )
    assert any(r["is_observed_occurrence_candidate"] == "true" for r in rows)
    assert any(r["is_susceptibility_or_risk_context"] == "true" for r in rows)
