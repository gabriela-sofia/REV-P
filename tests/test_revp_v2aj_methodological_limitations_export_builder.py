import os

import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_limitations_export_frames_absence_as_controlled_limitation(tmp_path, monkeypatch):
    data, docs = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_methodological_limitations_export_builder(common.parse_args([]))
    assert len(rows) == 9
    assert any("Controlled limitation" in r["safe_tcc_wording"] for r in rows)
    assert os.path.exists(docs / "protocolo_c_v2aj_methodological_limitations_export.md")
