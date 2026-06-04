import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env


def test_rj_portal_resolver_does_not_authenticate(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    rows = common.run_rj_public_portal_resolver(
        allow_web=True,
        fixture_html="tests/fixtures/v1up/rj_portal_fixture.html",
    )
    assert rows
    assert all(r["requires_authentication"] == "false" for r in rows)
