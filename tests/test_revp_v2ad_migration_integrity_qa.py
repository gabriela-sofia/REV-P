from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env, pkg
import scripts.protocolo_c.revp_v2ad_common as common


def test_migration_qa_clean(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_migration_integrity_qa(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0


def test_migration_qa_detects_lost_package(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Rebuild registry WITHOUT the missing-patch package -> v1us has it, v2ac lost it.
    import csv
    from tests.test_revp_v2ad_package_contract_qa import PKG_COLUMNS
    path = str(data / "v2ac_event_patch_v2_package_registry.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PKG_COLUMNS)
        w.writeheader()
        w.writerow(pkg())  # only EPC0, EPC2 dropped
    rows = common.run_migration_integrity_qa(common.parse_args([]))
    assert any(r["check_name"] == "all_ids_preserved" and r["status"] == "FAIL" for r in rows)
