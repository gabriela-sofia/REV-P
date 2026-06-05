import csv
from pathlib import Path

import pytest

import scripts.protocolo_c.revp_v2aj_common as common


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    for p in (data, docs, cfg):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    return data, docs


def install_inputs(data):
    write_csv(data / "v2ah_ground_truth_search_stop_gate.csv", ["ground_truth_search_status"], [{"ground_truth_search_status": "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE"}])
    write_csv(data / "v2ah_candidate_reference_review_queue.csv", ["package_id", "event_id", "region", "patch_id"], [{"package_id": "PKG1", "event_id": "EV1", "region": "REC", "patch_id": "REC_00001"}, {"package_id": "PKG2", "event_id": "EV2", "region": "PET", "patch_id": "PET_00002"}])
    write_csv(data / "v2ah_safe_tcc_export_registry.csv", ["id"], [{"id": "x"}])
    write_csv(data / "v2ah_guardrail_regression.csv", ["id", "status"], [{"id": "x", "status": "PASS"}])
    write_csv(data / "v2ah_completion_report.csv", ["metric", "value"], [{"metric": "stop_gate", "value": "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE"}])
    write_csv(data / "v2ai_review_assignment_registry.csv", ["assignment_id", "package_id"], [{"assignment_id": "A1", "package_id": "PKG1"}, {"assignment_id": "A2", "package_id": "PKG1"}, {"assignment_id": "A3", "package_id": "PKG2"}, {"assignment_id": "A4", "package_id": "PKG2"}])
    write_csv(data / "v2ai_reviewer_decision_template.csv", ["assignment_id", "decision_status"], [{"assignment_id": "A1", "decision_status": "PENDING_HUMAN_REVIEW"}, {"assignment_id": "A2", "decision_status": "PENDING_HUMAN_REVIEW"}, {"assignment_id": "A3", "decision_status": "PENDING_HUMAN_REVIEW"}, {"assignment_id": "A4", "decision_status": "PENDING_HUMAN_REVIEW"}])
    write_csv(data / "v2ai_adjudication_queue.csv", ["package_id", "adjudication_status"], [{"package_id": "PKG1", "adjudication_status": "WAITING_FOR_HUMAN_REVIEW"}, {"package_id": "PKG2", "adjudication_status": "WAITING_FOR_HUMAN_REVIEW"}])
    write_csv(data / "v2ai_uncertainty_registry.csv", ["package_id"], [{"package_id": "PKG1"}, {"package_id": "PKG2"}])
    write_csv(data / "v2ai_safe_promotion_blockers.csv", ["package_id", "promotion_allowed"], [{"package_id": "PKG1", "promotion_allowed": "false"}, {"package_id": "PKG2", "promotion_allowed": "false"}])
    write_csv(data / "v2ai_guardrail_regression.csv", ["id", "status"], [{"id": "x", "status": "PASS"}])
    write_csv(data / "v2ai_next_actions_registry.csv", ["rank", "next_action"], [{"rank": "1", "next_action": "HUMAN_REVIEW_EXECUTION_OR_SAFE_TCC_EXPORT"}])
    write_csv(data / "v2ai_completion_report.csv", ["metric", "value"], [
        {"metric": "candidates", "value": "2"},
        {"metric": "assignments", "value": "4"},
        {"metric": "decision_templates", "value": "4"},
        {"metric": "adjudication_status", "value": "WAITING_FOR_HUMAN_REVIEW"},
        {"metric": "next_action_rank_1", "value": "HUMAN_REVIEW_EXECUTION_OR_SAFE_TCC_EXPORT"},
    ])


def test_common_requires_v2ah_and_v2ai_and_blocks_overclaim(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.assert_v2ah_v2ai_ready()
    install_inputs(data)
    assert common.assert_v2ah_v2ai_ready()
    assert common.safe_claim_id("x").startswith("CLM_")
    with pytest.raises(ValueError):
        common.assert_no_operational_claim([{"safe_wording": "ground truth validado"}])
    common.assert_no_operational_claim([{"unsafe_wording": "ground truth validado"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_review([{"human_review_completed": "true"}])
