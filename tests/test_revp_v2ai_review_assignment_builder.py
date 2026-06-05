from collections import Counter

import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env


def test_review_assignment_builder_creates_two_slots_per_candidate(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    rows = common.run_review_assignment_builder(common.parse_args([]))
    assert len(rows) == 4
    assert set(r["reviewer_slot"] for r in rows) == {"reviewer_a", "reviewer_b"}
    counts = Counter(r["package_id"] for r in rows)
    assert all(v == 2 for v in counts.values())
    assert all(r["assignment_status"] == "ASSIGNED_SLOT_PENDING_HUMAN_REVIEW" for r in rows)
