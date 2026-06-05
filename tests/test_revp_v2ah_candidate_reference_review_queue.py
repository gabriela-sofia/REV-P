import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, set_env


def test_review_queue_preserves_review_only_candidates(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_candidate_reference_review_queue(common.parse_args([]))
    assert len(rows) == 2
    assert all(r["candidate_status"] in {"REVIEW_ONLY_CANDIDATE", "BLOCKED_REFERENCE_CANDIDATE"} for r in rows)
    assert all("ground_reference" in r["forbidden_use"] for r in rows)
    assert [int(r["review_priority_rank"]) for r in rows] == [1, 2]
