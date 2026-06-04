import csv

from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ad_common as common


def test_readiness_qa_clean_and_detects_overlay_inconsistency(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_readiness_consistency_qa(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0

    # Tamper readiness: overlay_readiness STRONG while geometry absent -> FAIL.
    path = str(data / "v2ac_v2_readiness_matrix.csv")
    with open(path, newline="", encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
        cols = list(reg[0].keys())
    for r in reg:
        if r["dimension"] == "overlay_readiness":
            r["classification"] = "STRONG"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reg)
    rows = common.run_readiness_consistency_qa(common.parse_args([]))
    assert any(r["check_name"] == "overlay_blocked_when_geometry_absent" and r["status"] == "FAIL" for r in rows)
