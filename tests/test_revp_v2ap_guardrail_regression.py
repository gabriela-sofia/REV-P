import pytest

import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all, write_csv


def test_regression_passes_on_clean_outputs(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    common.run_master_orchestrator(common.parse_args([]))
    rows = common.run_guardrail_regression(common.parse_args([]))
    assert all(r["status"] == "PASS" for r in rows)


def test_regression_fails_on_patch_truth_allowed(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    write_csv(protocol / "v2ap_bad.csv", ["id", "note"],
              [{"id": "X", "note": "patch_truth_allowed=true"}])
    with pytest.raises(ValueError):
        common.run_guardrail_regression(common.parse_args([]))


def test_regression_fails_on_can_create_ground_truth(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    write_csv(protocol / "v2ap_bad.csv", ["candidate_id", "can_create_ground_truth"],
              [{"candidate_id": "X", "can_create_ground_truth": "true"}])
    with pytest.raises(ValueError):
        common.run_guardrail_regression(common.parse_args([]))


def test_regression_fails_on_local_only(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    (docs / "v2ap_bad.md").write_text("uses local" + "_only path\n", encoding="utf-8")
    with pytest.raises(ValueError):
        common.run_guardrail_regression(common.parse_args([]))
