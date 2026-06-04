import csv

from tests.test_revp_v2ac_event_patch_v2_package_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ac_common as common


def test_schema_validator_passes_valid_and_blocks_missing_patch(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    rows = common.run_schema_contract_validator(common.parse_args([]))
    by_epc = {r["event_patch_candidate_id"]: r for r in rows}
    assert by_epc["EPC0"]["validation_status"] == "SCHEMA_VALID_NON_OPERATIONAL"
    assert by_epc["EPC2"]["validation_status"] == "SCHEMA_INVALID_MISSING_PATCH_ID"
    assert all(r["guardrail_violation_count"] == "0" for r in rows)
    assert all(r["forbidden_value_count"] == "0" for r in rows)


def test_schema_validator_flags_guardrail_violation(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    # Tamper one package: overlay no longer BLOCKED (a guardrail violation).
    path = str(data / "v2ac_event_patch_v2_package_registry.csv")
    with open(path, newline="", encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
        cols = reg[0].keys()
    reg[0]["overlay_status"] = "ALLOWED"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reg)
    rows = common.run_schema_contract_validator(common.parse_args([]))
    tampered = rows[0]
    assert int(tampered["guardrail_violation_count"]) >= 1
    assert tampered["validation_status"] == "SCHEMA_INVALID"
