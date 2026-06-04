import csv

from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2af_common as common


def test_event_patch_regression_clean(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_event_patch_v2_regression_runner(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0
    by_name = {r["check_name"]: r for r in rows}
    assert by_name["packages_172"]["status"] == "PASS"
    assert by_name["no_anchor_crosswalk"]["status"] == "PASS"


def test_event_patch_regression_detects_applied_unlinkable_date(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Apply a date on an unlinkable package -> regression must FAIL.
    path = str(data / "v2ac_event_patch_v2_package_registry.csv")
    with open(path, newline="", encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
        cols = list(reg[0].keys())
    reg[0]["sentinel_scene_date"] = "2022-02-02"  # date_linkability_status is UNLINKABLE_NAMESPACE
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reg)
    rows = common.run_event_patch_v2_regression_runner(common.parse_args([]))
    assert any(r["check_name"] == "no_unlinkable_date_applied" and r["status"] == "FAIL" for r in rows)
