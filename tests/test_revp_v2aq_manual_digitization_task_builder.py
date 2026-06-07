import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_digitization_tasks_carry_do_not_infer(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_event_geometry_candidate_builder(common.parse_args([]))
    rows = common.run_manual_digitization_task_builder(common.parse_args([]))
    assert rows
    for r in rows:
        low = r["do_not_infer"].lower()
        assert "nao inferir" in low or "nao inventar" in low
        assert r["priority"] in ("HIGH", "MEDIUM", "LOW")
