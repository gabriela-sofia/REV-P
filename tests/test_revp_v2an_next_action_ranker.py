import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all, write_csv


def test_rank1_execute_when_candidate_advances(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    common.run_gate_closure_matrix_builder(common.parse_args([]))
    common.run_ground_reference_readiness_scorer(common.parse_args([]))
    common.run_validation_decision_registry_builder(common.parse_args([]))
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "EXECUTE_HUMAN_GROUND_REFERENCE_REVIEW"


def test_rank1_collect_when_none_advances(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    write_csv(protocol / "v2an_validation_decision_registry.csv",
              ["decision_id", "candidate_id", "decision_status"],
              [{"decision_id": "VD", "candidate_id": "X",
                "decision_status": "NEEDS_MORE_SPATIAL_EVIDENCE"}])
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "COLLECT_MISSING_SPATIAL_GEOMETRY_AND_SENTINEL_CROSSWALK"


def test_no_forbidden_action_allowed(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    common.run_gate_closure_matrix_builder(common.parse_args([]))
    common.run_ground_reference_readiness_scorer(common.parse_args([]))
    common.run_validation_decision_registry_builder(common.parse_args([]))
    rows = common.run_next_action_ranker(common.parse_args([]))
    for r in rows:
        if r["allowed"] == "true":
            name = r["next_action"].lower()
            assert "training" not in name and "protocol_b" not in name
            assert "overlay" not in name and "label" not in name
