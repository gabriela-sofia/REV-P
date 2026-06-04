import scripts.protocolo_c.revp_v1un_recife_common as common


def test_protocol_status_final_does_not_allow_ground_reference(tmp_path):
    rows = common.run_protocol_c_status_updater(str(tmp_path / "status.csv"))
    assert rows[0]["new_status"] == "LOCALITY_ONLY_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED"
    assert rows[0]["can_advance_to_ground_reference"] == "false"
    assert rows[0]["can_create_training_label"] == "false"
