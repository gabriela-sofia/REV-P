import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_review_guide_blocks_dino_and_context_only_promotion(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_review_guide_registry_builder(common.parse_args([]))
    assert len(rows) == 8
    basis = "|".join(r["disallowed_basis"] for r in rows)
    assert "DINO" in basis
    assert "GIS context alone" in basis
