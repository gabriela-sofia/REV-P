import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_claims_guardrails_registry_and_md(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_claims_guardrails_appendix_builder(common.parse_args([]))
    statuses = {r["status"] for r in rows}
    assert "allowed" in statuses
    assert "prohibited" in statuses
    assert "guardrail" in statuses
    md = (atlas / "v2am_claims_and_guardrails_appendix.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
    assert "exemplos negativos" in md.lower() or "unsafe wording" in md.lower()
