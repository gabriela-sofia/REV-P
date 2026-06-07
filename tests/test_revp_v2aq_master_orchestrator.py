import pytest

import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all, read_csv


def test_orchestrator_runs_all_steps(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    rows = common.run_master_orchestrator(common.parse_args([]))
    assert len(rows) == 13
    assert all(r["status"] == "OK" for r in rows)
    assert rows[0]["step_name"] == "priority_candidate"
    assert rows[-1]["step_name"] == "completion_report"


def test_orchestrator_fail_fast(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)

    def boom(args=None):
        raise ValueError("injected")

    monkeypatch.setattr(common, "run_event_geometry_candidate_builder", boom)
    with pytest.raises(ValueError):
        common.run_master_orchestrator(common.parse_args([]))
    manifest = read_csv(protocol / "v2aq_orchestrator_run_manifest.csv")
    assert manifest[-1]["status"] == "FAIL"
    assert manifest[-1]["step_name"] == "event_geometry_candidate"
