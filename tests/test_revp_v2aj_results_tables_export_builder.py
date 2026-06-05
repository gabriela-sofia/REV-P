import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_results_tables_do_not_recommend_accuracy_or_training(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_results_tables_export_builder(common.parse_args([]))
    assert len(rows) == 7
    joined = " ".join(r["safe_caption"] + " " + r["allowed_interpretation"] for r in rows).lower()
    assert "accuracy" not in joined
    assert "training" not in joined
