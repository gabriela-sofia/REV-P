import os

import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, read_csv, set_env


def test_completion_report_records_counts_and_safe_rank(tmp_path, monkeypatch):
    data, docs = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_all(common.parse_args([]))
    rows = read_csv(data / "v2aj_completion_report.csv")
    metrics = {r["metric"]: r for r in rows}
    assert metrics["claims_allowed"]["status"] == "SAFE_CLAIMS"
    assert metrics["guardrail_failures"]["value"] == "0"
    assert metrics["next_action_rank_1"]["value"] == "SAFE_TCC_PROTOCOL_C_WRITEUP"
    assert os.path.exists(docs / "protocolo_c_v2aj_completion_report.md")
