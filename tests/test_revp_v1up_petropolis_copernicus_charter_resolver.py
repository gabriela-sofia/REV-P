import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env


def test_copernicus_charter_quickview_and_offtarget_do_not_promote(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    rows = common.run_copernicus_charter_resolver(
        allow_web=True,
        fixture_html="tests/fixtures/v1up/charter_quickview.html",
    )
    assert any(r["source_id"] == "CHARTER_751" and r["is_event_specific"] == "true" for r in rows)
    assert all(r["is_vector_package_candidate"] == "false" for r in rows)
    rows = common.run_copernicus_charter_resolver(
        allow_web=True,
        fixture_html="tests/fixtures/v1up/copernicus_activation_off_target.html",
    )
    assert any(r["is_off_target"] == "true" for r in rows)
