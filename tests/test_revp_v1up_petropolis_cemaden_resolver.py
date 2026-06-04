import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env


def test_cemaden_resolver_never_promotes_alert_as_geometry(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    rows = common.run_cemaden_resolver(
        allow_web=True,
        fixture_html="tests/fixtures/v1up/cemaden_fixture.html",
    )
    assert rows
    assert all(r["is_geometry_candidate"] == "false" for r in rows)
    assert all("CONTEXT_NOT_OBSERVED_GEOMETRY" in r["blocking_reason"] for r in rows)
