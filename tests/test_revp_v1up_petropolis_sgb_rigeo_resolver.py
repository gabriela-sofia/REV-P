import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env


def test_sgb_rigeo_resolver_uses_fixture_and_does_not_invent_2024_item(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    rows = common.run_sgb_rigeo_resolver(
        allow_web=True,
        fixture_html="tests/fixtures/v1up/rigeo_item_with_bitstreams.html",
    )
    assert any(r["format_hint"] == "zip" and r["is_geometry_candidate"] == "true" for r in rows)
    pet2024 = [r for r in rows if r["event_id"] == "PET_2024_03_21_28"][0]
    assert pet2024["blocking_reason"] == "NO_EVENT_SPECIFIC_PUBLIC_RIGEO_ITEM_RESOLVED"
