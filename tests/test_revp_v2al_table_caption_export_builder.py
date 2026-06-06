import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_table_captions_are_governance_framed(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    rows = common.run_table_caption_export_builder(common.parse_args([]))
    assert rows
    assert all(r["manual_review_required"] == "true" for r in rows)
    assert all("governanca" in r["safe_caption"].lower() for r in rows)
    md = (integration / "v2al_safe_table_captions.md").read_text(encoding="utf-8")
    assert "governanca" in md.lower()
    common.assert_safe_manuscript_language(md)


def test_table_captions_default_when_no_registry(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    (data / "v2aj_results_tables_export_registry.csv").unlink()
    rows = common.run_table_caption_export_builder(common.parse_args([]))
    assert rows
    assert rows[0]["manual_review_required"] == "true"
