import pytest

import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all, read_csv


def test_orchestrator_runs_all_steps(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_master_orchestrator(common.parse_args([]))
    assert len(rows) == 12
    assert all(r["status"] == "OK" for r in rows)
    assert rows[0]["step_name"] == "artifact_inventory"
    assert rows[-1]["step_name"] == "completion_report"
    assert (atlas / "v2am_orchestrator_run_manifest.md").exists()
    assert (data / "v2am_orchestrator_run_manifest.csv").exists()


def test_orchestrator_fail_fast(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)

    def boom(args=None):
        raise ValueError("injected failure")

    monkeypatch.setattr(common, "run_traceability_dag_builder", boom)
    with pytest.raises(ValueError):
        common.run_master_orchestrator(common.parse_args([]))
    manifest = read_csv(data / "v2am_orchestrator_run_manifest.csv")
    # records steps up to and including the failed one, last is FAIL
    assert manifest[-1]["status"] == "FAIL"
    assert manifest[-1]["step_name"] == "traceability_dag"
