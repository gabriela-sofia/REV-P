from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env, pkg, missing_pkg
import scripts.protocolo_c.revp_v2ad_common as common


def test_temporal_qa_clean_and_detects_unlinkable_date_applied(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_temporal_safety_qa(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0

    # Inject an unlinkable date that was wrongly applied -> must FAIL.
    bad = pkg(sentinel_scene_date="2022-02-02", date_linkability_status="UNLINKABLE_NAMESPACE")
    install_base_inputs(data, packages=[bad, missing_pkg()])
    rows = common.run_temporal_safety_qa(common.parse_args([]))
    assert any(r["check_name"] == "unlinkable_date_not_applied" and r["status"] == "FAIL" for r in rows)


def test_temporal_qa_detects_inferred_date(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    bad = pkg(sentinel_date_inferred="true")
    install_base_inputs(data, packages=[bad, missing_pkg()])
    rows = common.run_temporal_safety_qa(common.parse_args([]))
    assert any(r["check_name"] == "sentinel_date_not_inferred" and r["status"] == "FAIL" for r in rows)
