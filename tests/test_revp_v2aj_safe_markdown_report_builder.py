import os

import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_safe_markdown_report_is_export_not_tcc_chapter(tmp_path, monkeypatch):
    data, docs = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_safe_markdown_report_builder(common.parse_args([]))
    path = docs / "protocolo_c_v2aj_safe_tcc_export.md"
    assert rows[0]["status"] == "WRITTEN_SAFE_EXPORT"
    assert os.path.exists(path)
    text = path.read_text(encoding="utf-8")
    assert "Nao pode dizer:" in text
    assert "## Estado consolidado" in text
