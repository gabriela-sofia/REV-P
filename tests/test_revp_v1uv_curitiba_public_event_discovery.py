from tests.test_revp_v1uv_curitiba_source_target_builder import set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_public_discovery_dry_run_keeps_candidate_blocked_when_no_fetch(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    common.run_source_target_builder(common.parse_args([]))
    rows = common.run_public_event_discovery(common.parse_args(["--dry-run"]))
    assert rows
    assert all(r["official_source_status"] == "OFFICIAL_PUBLIC_SOURCE" for r in rows)
    assert all(r["candidate_status"] in {"BLOCKED", "PUBLIC_OFFICIAL_EVENT_CANDIDATE_SIGNAL"} for r in rows)
