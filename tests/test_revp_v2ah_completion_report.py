import os

import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, read_csv, set_env


def test_completion_report_records_outputs_and_stop_decision(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_all(common.parse_args([]))
    rows = read_csv(data / "v2ah_completion_report.csv")
    metrics = {r["metric"]: r for r in rows}
    assert metrics["stop_gate"]["value"] == "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE"
    assert metrics["review_queue_rows"]["status"] == "REVIEW_ONLY"
    assert metrics["guardrail_failures"]["value"] == "0"
    doc = os.path.join(common.DOCS_DIR, "protocolo_c_v2ah_completion_report.md")
    assert os.path.exists(doc)
