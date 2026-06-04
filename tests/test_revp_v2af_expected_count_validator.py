import csv

from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2af_common as common


def test_expected_count_clean(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_expected_count_validator(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0
    by_name = {r["check_name"]: r for r in rows}
    assert by_name["canonical_regions"]["actual_count"] == "3"
    assert by_name["canonical_events"]["actual_count"] == "4"
    assert by_name["v2ac_packages"]["actual_count"] == "172"


def test_expected_count_fails_on_lost_package(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Drop one package from v2ac registry -> count 171 fails.
    path = str(data / "v2ac_event_patch_v2_package_registry.csv")
    with open(path, newline="", encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
        cols = list(reg[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reg[:-1])
    rows = common.run_expected_count_validator(common.parse_args([]))
    assert any(r["check_name"] == "v2ac_packages" and r["status"] == "FAIL" for r in rows)
