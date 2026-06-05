import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env


def test_uncertainty_registry_preserves_blocker_signatures(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    rows = common.run_uncertainty_registry_builder(common.parse_args([]))
    assert len(rows) == 2
    assert any(r["dominant_uncertainty"] == "temporal_crosswalk" for r in rows)
    assert any(r["dominant_uncertainty"] == "spatial" for r in rows)
    assert all("signature=" in r["uncertainty_reason"] for r in rows)
    assert all(r["effect_on_allowed_use"] == "review_only_no_automatic_resolution" for r in rows)
