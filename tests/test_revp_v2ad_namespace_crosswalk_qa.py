from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env, pkg, missing_pkg
import scripts.protocolo_c.revp_v2ad_common as common


def test_namespace_qa_clean_and_detects_inferred_crosswalk(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_namespace_crosswalk_qa(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0

    # Inject an inferred crosswalk -> must FAIL.
    bad = pkg(crosswalk_inferred="true", crosswalk_status="INFERRED_BY_REGION")
    install_base_inputs(data, packages=[bad, missing_pkg()])
    rows = common.run_namespace_crosswalk_qa(common.parse_args([]))
    assert any(r["check_name"] == "crosswalk_not_inferred" and r["status"] == "FAIL" for r in rows)


def test_namespace_qa_detects_anchor_crosswalk_leak(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    bad = pkg(refpatch_id="REFPATCH_REC_001")
    install_base_inputs(data, packages=[bad, missing_pkg()])
    rows = common.run_namespace_crosswalk_qa(common.parse_args([]))
    assert any(r["check_name"] == "no_anchor_or_refpatch_crosswalk" and r["status"] == "FAIL" for r in rows)
