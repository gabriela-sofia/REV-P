import csv
from pathlib import Path

import pytest

import scripts.protocolo_c.revp_v2ai_common as common


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
    return data


def install_v2ah(data):
    queue_cols = [
        "review_queue_id", "package_id", "event_id", "region", "patch_id",
        "hazard_type", "candidate_status", "evidence_strength",
        "dominant_blocker", "review_priority_score", "review_priority_rank",
        "review_action", "allowed_use", "forbidden_use",
    ]
    queue = [
        {
            "review_queue_id": "RQ_FIX_001", "package_id": "EPC_FIX_001",
            "event_id": "REC_2022_05_24_30", "region": "REC",
            "patch_id": "REC_00001", "hazard_type": "FLOOD_CONTEXT_REVIEW_ONLY",
            "candidate_status": "REVIEW_ONLY_CANDIDATE",
            "evidence_strength": "DOCUMENTARY_CONTEXT_ONLY",
            "dominant_blocker": "no explicit Sentinel date crosswalk",
            "review_priority_score": "45", "review_priority_rank": "1",
            "review_action": "review_candidate_context_only",
            "allowed_use": "review_queue_and_tcc_context_only",
            "forbidden_use": "ground_reference|label|training|overlay|prediction|protocol_b_reopen",
        },
        {
            "review_queue_id": "RQ_FIX_002", "package_id": "EPC_FIX_002",
            "event_id": "PET_2022_02_15", "region": "PET",
            "patch_id": "PET_00002", "hazard_type": "MASS_MOVEMENT_CONTEXT_REVIEW_ONLY",
            "candidate_status": "BLOCKED_REFERENCE_CANDIDATE",
            "evidence_strength": "DOCUMENTARY_CONTEXT_ONLY",
            "dominant_blocker": "no observed geometry",
            "review_priority_score": "35", "review_priority_rank": "2",
            "review_action": "review_candidate_context_only",
            "allowed_use": "review_queue_and_tcc_context_only",
            "forbidden_use": "ground_reference|label|training|overlay|prediction|protocol_b_reopen",
        },
    ]
    write_csv(data / "v2ah_candidate_reference_review_queue.csv", queue_cols, queue)
    write_csv(data / "v2ah_ground_truth_search_stop_gate.csv", ["ground_truth_search_status"], [{"ground_truth_search_status": "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE"}])
    for name in [
        "v2ah_candidate_evidence_dossier_index.csv",
        "v2ah_reopen_conditions_registry.csv",
        "v2ah_safe_tcc_export_registry.csv",
        "v2ah_guardrail_regression.csv",
        "v2ah_next_actions_registry.csv",
    ]:
        write_csv(data / name, ["id", "status"], [{"id": "x", "status": "PASS"}])
    return queue


def test_common_requires_v2ah_and_rejects_fake_review(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.assert_v2ah_ready()
    install_v2ah(data)
    assert common.assert_v2ah_ready()
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"label": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_human_review([{"human_review_completed": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_human_review([{"decision_timestamp": "2026-06-04"}])
