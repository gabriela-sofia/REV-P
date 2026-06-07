import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all

ALLOWED = {
    "REMAINS_DOCUMENTED_OBSERVED_CANDIDATE",
    "ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW",
    "NEEDS_MORE_SPATIAL_EVIDENCE", "NEEDS_SENTINEL_CROSSWALK",
    "NEEDS_SOURCE_STRENGTH_REVIEW", "BLOCKED_BY_PHENOMENON_AMBIGUITY",
}
FORBIDDEN = {"GROUND_TRUTH_VALIDATED", "LABEL_READY", "PROTOCOL_B_REOPENED",
             "TRAINING_READY", "OPERATIONAL_VALIDATION"}


def test_decisions_use_allowed_states_only(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    common.run_gate_closure_matrix_builder(common.parse_args([]))
    common.run_ground_reference_readiness_scorer(common.parse_args([]))
    rows = common.run_validation_decision_registry_builder(common.parse_args([]))
    assert len(rows) == 9
    for r in rows:
        assert r["decision_status"] in ALLOWED
        assert r["decision_status"] not in FORBIDDEN
        assert r["ground_truth_status"] == "NOT_ESTABLISHED"
        assert r["label_status"] == "NOT_CREATED"
        assert r["protocol_b_status"] == "BLOCKED"
