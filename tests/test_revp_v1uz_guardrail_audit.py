from pathlib import Path

from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v1uz"


def _counts(rows, artifact):
    return {r["check_type"]: int(r["violation_count"]) for r in rows if r["artifact"] == artifact}


def test_guardrail_audit_detects_violations_in_fixture(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    violation = FIXTURE_DIR / "guardrail_violation_fixture.csv"
    rows = common.run_guardrail_audit(common.parse_args([]), artifacts=[str(violation)])
    counts = _counts(rows, "guardrail_violation_fixture.csv")
    assert counts["forbidden_true_value"] >= 1
    assert counts["forbidden_status"] >= 1
    assert counts["absolute_path"] >= 1
    assert counts["local_only_leak"] >= 1
    assert counts["tool_name_leak"] >= 1
    assert any(r["status"] == "FAIL" for r in rows)


def test_guardrail_audit_passes_clean_fixture(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    clean = FIXTURE_DIR / "guardrail_clean_fixture.csv"
    rows = common.run_guardrail_audit(common.parse_args([]), artifacts=[str(clean)])
    assert all(r["status"] == "PASS" for r in rows)
    assert all(int(r["violation_count"]) == 0 for r in rows)


def test_guardrail_audit_passes_real_v1uz_outputs(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_all(common.parse_args([]))
    # Re-run audit over the freshly generated v1uz artifacts.
    rows = common.run_guardrail_audit(common.parse_args([]))
    assert rows, "audit produced no rows"
    assert all(r["status"] == "PASS" for r in rows)
