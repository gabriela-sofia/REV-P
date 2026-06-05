import os

import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, read_csv, set_env


def test_completion_report_records_pending_review_package(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    common.run_all(common.parse_args([]))
    rows = read_csv(data / "v2ai_completion_report.csv")
    metrics = {r["metric"]: r for r in rows}
    assert metrics["candidates"]["value"] == "2"
    assert metrics["assignments"]["value"] == "4"
    assert metrics["decision_templates"]["status"] == "PENDING_HUMAN_REVIEW"
    assert metrics["human_review_status"]["status"] == "NOT_COMPLETED"
    assert metrics["adjudication_status"]["status"] == "NOT_COMPLETED"
    assert metrics["next_action_rank_1"]["value"] == "HUMAN_REVIEW_EXECUTION_OR_SAFE_TCC_EXPORT"
    assert os.path.exists(os.path.join(common.DOCS_DIR, "protocolo_c_v2ai_completion_report.md"))
