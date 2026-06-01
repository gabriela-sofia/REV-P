"""REV-P v1ry — Integration test runner wrapper.

Generates a test plan CSV. By default (REVP_RUN_INTEGRATION_TESTS != true)
only writes the plan without executing pytest. When env is true, executes
each test suite with a per-command timeout and records results.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from revp_v1rs_v1rz_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_PLAN = _p("REVP_V1RY_OUT_PLAN", DATASETS / "protocol_c_integration_test_plan_v1ry.csv")
OUT_SUMMARY = _p("REVP_V1RY_OUT_SUMMARY", DATASETS / "protocol_c_integration_test_summary_v1ry.csv")
SCHEMA_PLAN = _p("REVP_V1RY_SCHEMA_PLAN", SCHEMAS / "protocol_c_integration_test_plan_v1ry_schema.csv")
SCHEMA_SUM = _p("REVP_V1RY_SCHEMA_SUM", SCHEMAS / "protocol_c_integration_test_summary_v1ry_schema.csv")
DOC = _p("REVP_V1RY_DOC", DOCS / "revp_v1ry_integration_test_runner_wrapper.md")

PLAN_FIELDS = ["plan_id", "test_suite", "command", "timeout_sec", "expected_status",
               "actual_status", "test_count_passed", "test_count_failed", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

_PLAN: list[tuple[str, ...]] = [
    ("T01", "P0_ground_reference", "pytest tests/test_revp_v1qu_v1qz_ground_reference_partial_validation.py -q", "120", "PASS"),
    ("T02", "P1_external_intake", "pytest tests/test_revp_v1ra_v1rf_external_intake_workflow.py -q", "90", "PASS"),
    ("T03", "P2_review_gate", "pytest tests/test_revp_v1rg_v1rm_review_supervisor_gate.py -q", "90", "PASS"),
    ("T04", "P3_dashboard_roadmap", "pytest tests/test_revp_v1rn_v1rr_protocol_c_dashboard_roadmap.py -q", "60", "PASS"),
    ("T05", "INT_hardening", "pytest tests/test_revp_v1rs_v1rz_integration_hardening.py -q", "90", "PASS"),
    ("T06", "DINO_local_readiness", "pytest tests/test_revp_v1qn_v1qt_local_dino_readiness.py -q", "180", "PASS_OR_TIMEOUT"),
    ("T07", "DINO_smoke", "pytest tests/test_revp_v1qg_v1qm_dino_smoke_embeddings.py -q", "120", "PASS_OR_TIMEOUT"),
    ("T08", "git_diff_check", "git diff --check", "30", "CLEAN"),
    ("T09", "guardrail_scan", "python scripts/protocolo_c/revp_v1ru_cross_block_guardrail_audit.py", "60", "GUARDRAIL_CLEAN"),
]


def _run_cmd(cmd: str, timeout: int) -> tuple[str, str, str]:
    """Return (status, stdout_tail, stderr_tail)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=ROOT,
        )
        out = (r.stdout + r.stderr)[-800:].strip()
        if r.returncode == 0:
            return "PASS", out, ""
        return "FAIL", out, ""
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", ""
    except Exception as e:
        return "ERROR", str(e), ""


def run(datasets: Path | None = None) -> dict[str, Any]:
    execute = os.environ.get("REVP_RUN_INTEGRATION_TESTS", "false").strip().lower() == "true"
    plan_rows: list[dict[str, Any]] = []
    for p in _PLAN:
        pid, suite, cmd, timeout, expected = p
        actual = "NOT_EXECUTED"
        passed = ""
        failed = ""
        if execute:
            status, out, _ = _run_cmd(cmd, int(timeout))
            actual = status
            import re
            m = re.search(r"(\d+) passed", out)
            passed = m.group(1) if m else ""
            m2 = re.search(r"(\d+) failed", out)
            failed = m2.group(1) if m2 else ""
        plan_rows.append({
            "plan_id": pid, "test_suite": suite, "command": cmd,
            "timeout_sec": timeout, "expected_status": expected,
            "actual_status": actual, "test_count_passed": passed,
            "test_count_failed": failed, "notes": "execute_only_if_REVP_RUN_INTEGRATION_TESTS=true",
        })

    write_csv_with_header(OUT_PLAN, plan_rows, PLAN_FIELDS)
    write_schema_safe(SCHEMA_PLAN, PLAN_FIELDS, "v1ry_plan")

    executed = sum(1 for r in plan_rows if r["actual_status"] != "NOT_EXECUTED")
    passed_n = sum(1 for r in plan_rows if r["actual_status"] == "PASS")
    failed_n = sum(1 for r in plan_rows if r["actual_status"] == "FAIL")
    summary = [
        {"stat_key": "test_execution_mode", "stat_value": "EXECUTE" if execute else "PLAN_ONLY"},
        {"stat_key": "suites_planned", "stat_value": str(len(plan_rows))},
        {"stat_key": "suites_executed", "stat_value": str(executed)},
        {"stat_key": "suites_passed", "stat_value": str(passed_n)},
        {"stat_key": "suites_failed", "stat_value": str(failed_n)},
        {"stat_key": "stage", "stat_value": "v1ry"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_safe(SCHEMA_SUM, SUM_FIELDS, "v1ry_summary")

    write_doc(DOC, "v1ry — Integration Test Runner Wrapper", [
        "## Objetivo",
        "Gera plano de testes; por padrão (REVP_RUN_INTEGRATION_TESTS != true) não executa "
        "pytest. Quando env=true, executa cada suite com timeout individual.",
        "## Resultado",
        f"Suites planejadas: {len(plan_rows)}. Executadas: {executed}.",
        "## Como executar",
        "`$env:REVP_RUN_INTEGRATION_TESTS='true'; python revp_v1ry_integration_test_runner_wrapper.py`",
    ])

    print(f"[v1ry] mode={'EXECUTE' if execute else 'PLAN_ONLY'} planned={len(plan_rows)} executed={executed}")
    return {"execute": execute, "planned": len(plan_rows), "executed": executed}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ry integration test runner").parse_args()
    run()
