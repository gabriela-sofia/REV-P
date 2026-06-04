import csv
import os

import scripts.protocolo_c.revp_v1un_recife_common as common


def _write(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_base(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    data.mkdir(parents=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    _write(data / "v1um_recife_redacted_evidence_package_registry.csv", [
        "candidate_row_id", "hazard_class", "ground_truth_operational",
        "can_create_ground_reference", "can_create_training_label",
    ], [
        {"candidate_row_id": "c1", "hazard_class": "CIVIL_DEFENSE_GENERIC", "ground_truth_operational": "false", "can_create_ground_reference": "false", "can_create_training_label": "false"},
        {"candidate_row_id": "c2", "hazard_class": "RAIN_IMPACT", "ground_truth_operational": "false", "can_create_ground_reference": "false", "can_create_training_label": "false"},
    ])
    _write(data / "v1um_recife_human_review_batch_registry.csv", ["batch_id"], [{"batch_id": "b1"}, {"batch_id": "b2"}, {"batch_id": "b3"}])
    _write(data / "v1um_recife_neighborhood_signal_aggregation.csv", ["aggregation_id"], [{"aggregation_id": "a1"}])
    _write(data / "v1um_recife_non_overlay_readiness_matrix.csv", ["readiness_id"], [{"readiness_id": "r1"}])
    _write(data / "v1um_recife_locality_candidate_sample_registry.csv", ["sample_id"], [{"sample_id": "s1"}])
    return data


def test_consolidation_registry_is_created(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    rows = common.run_human_review_evidence_consolidator(str(data / "consolidation.csv"))
    assert rows[0]["locality_only_candidates"] == "2"
    assert rows[0]["overlay_status"] == "BLOCKED"
    assert rows[0]["can_create_ground_reference"] == "false"
