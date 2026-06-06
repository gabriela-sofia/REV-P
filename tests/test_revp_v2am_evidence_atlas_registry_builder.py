import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_atlas_has_all_axes_and_is_safe(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_evidence_atlas_registry_builder(common.parse_args([]))
    axes = {r["axis"] for r in rows}
    for expected in ("candidatos review-only", "blockers de promocao",
                     "revisao humana pendente", "adjudicacao pendente",
                     "claims permitidos/proibidos", "limitacoes metodologicas",
                     "integracao segura no manuscrito", "captions e tabelas", "guardrails"):
        assert expected in axes
    md = (atlas / "v2am_protocol_c_evidence_atlas.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
    assert "nao e o capitulo final" in md.lower()
