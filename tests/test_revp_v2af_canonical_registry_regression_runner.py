import csv

from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2af_common as common


def test_registry_regression_clean(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_canonical_registry_regression_runner(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0


def test_registry_regression_detects_region_status_change(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Tamper a canonical region status -> regression must FAIL.
    path = str(data / "v2ae_canonical_region_registry.csv")
    with open(path, newline="", encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
        cols = list(reg[0].keys())
    for r in reg:
        if r["region"] == "CUR":
            r["canonical_region_status"] = "REGION_PROMOTED_OPERATIONAL"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reg)
    rows = common.run_canonical_registry_regression_runner(common.parse_args([]))
    assert any(r["check_name"] == "region_status_CUR" and r["status"] == "FAIL" for r in rows)
