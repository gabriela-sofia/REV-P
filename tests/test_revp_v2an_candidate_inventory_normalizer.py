import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all, read_csv


def test_normalizes_nine_candidates(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_candidate_inventory_normalizer(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["operational_ground_truth_status"] == "NOT_ESTABLISHED" for r in rows)
    assert all(r["protocol_b_status"] == "BLOCKED" for r in rows)
    assert all(r["can_be_used_as_training_label"] == "false" for r in rows)
    assert (protocol / "v2an_observed_candidate_inventory_normalized.csv").exists()
    assert (docs / "v2an_observed_candidate_inventory_normalized.md").exists()


def test_fails_on_wrong_count(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch, "less_than_nine.csv")
    import pytest
    with pytest.raises(ValueError):
        common.run_candidate_inventory_normalizer(common.parse_args([]))
