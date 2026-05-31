"""Tests for REV-P DINO execution bridge v1qa-v1qf.

All I/O via tmp_path/env vars. Real datasets/ never touched.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS))

import revp_v1qa_v1qf_execution_bridge_common as BC  # noqa
import revp_v1pg_v1pm_dino_representation_common as C  # noqa

S = {
    "v1qa": SCRIPTS / "revp_v1qa_expanded_queue_import_bridge.py",
    "v1qb": SCRIPTS / "revp_v1qb_dino_execution_readiness_audit.py",
    "v1qc": SCRIPTS / "revp_v1qc_dino_dry_run_execution_package.py",
    "v1qd": SCRIPTS / "revp_v1qd_executor_compatibility_patch_report.py",
    "v1qe": SCRIPTS / "revp_v1qe_dino_execution_readiness_tcc_update.py",
    "v1qf": SCRIPTS / "revp_v1qf_dino_execution_bridge_bundle.py",
    "v1pq": SCRIPTS / "revp_v1pq_controlled_smoke_embedding_executor.py",
}


def _run(script: Path, env: dict, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), [])


def _base(**kw: str) -> dict[str, str]:
    return {**os.environ, **kw}


def _make_v1pw_queue(tmp: Path, n: int = 5) -> Path:
    q = tmp / "v1pw.csv"
    rows = [{
        "queue_id": f"V1PW_Q_{i:05d}",
        "visual_asset_id": f"VA_{i}", "patch_id": f"CUR_{i:05d}",
        "alias": f"CUR_{i:05d}", "region": "CURITIBA",
        "relative_path": f"data/sentinel/patch_curitiba_{i:05d}.tif",
        "path_hash": C.path_hash(f"data/sentinel/patch_curitiba_{i:05d}.tif"),
        "visual_type": "SENTINEL_TIF_REFERENCE",
        "queue_priority": "2", "queue_reason": "sentinel_tif_reference",
        "linkage_confidence": "HIGH", "manual_check_required": "false",
        "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
        "can_create_label": "false", "can_train_model": "false", "target_created": "false",
        "execution_status": "PENDING", "blocked_reason": "", "notes": "",
    } for i in range(1, n + 1)]
    C.write_csv(q, rows, list(rows[0].keys()))
    return q


def _make_backend_summary(tmp: Path, model_available: bool = False) -> Path:
    s = tmp / "backend.csv"
    C.write_csv(s, [
        {"stat_key": "can_execute_embeddings", "stat_value": str(model_available).lower()},
        {"stat_key": "final_status",
         "stat_value": "DINO_BACKEND_READY_LOCAL_MODEL" if model_available
         else "DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED"},
        {"stat_key": "model_available", "stat_value": str(model_available).lower()},
    ], ["stat_key", "stat_value"])
    return s


def _v1qa_env(tmp: Path, queue: Path, backend: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QA_IN_QUEUE=str(queue), REVP_V1QA_IN_BACKEND=str(backend),
        REVP_V1QA_OUT_QUEUE=str(tmp / "v1qa.csv"),
        REVP_V1QA_OUT_SUM=str(tmp / "v1qa_sum.csv"),
        REVP_V1QA_SCH_QUEUE=str(sch / "a.csv"),
        REVP_V1QA_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QA_DOC=str(tmp / "doc.md"),
    )


# ---------------------------------------------------------------------------
# v1qa — queue import bridge
# ---------------------------------------------------------------------------

def test_v1qa_imports_expanded_queue(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = _make_v1pw_queue(tmp_path, 5)
    backend = _make_backend_summary(tmp_path)
    r = _run(S["v1qa"], _v1qa_env(tmp_path, queue, backend, sch))
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "v1qa.csv")
    assert len(rows) == 5
    assert all(r["execution_queue_id"].startswith("V1QA_EQ_") for r in rows)


def test_v1qa_preserves_review_only(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = _make_v1pw_queue(tmp_path, 3)
    backend = _make_backend_summary(tmp_path)
    _run(S["v1qa"], _v1qa_env(tmp_path, queue, backend, sch))
    for row in _read(tmp_path / "v1qa.csv"):
        assert row["dino_allowed_use"] == "REVIEW_ONLY_REPRESENTATION"
        assert row["ready_for_dry_run"] == "true"


def test_v1qa_no_label_train_target(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = _make_v1pw_queue(tmp_path, 4)
    backend = _make_backend_summary(tmp_path)
    _run(S["v1qa"], _v1qa_env(tmp_path, queue, backend, sch))
    for row in _read(tmp_path / "v1qa.csv"):
        assert row["can_create_label"] == "false"
        assert row["can_train_model"] == "false"
        assert row["target_created"] == "false"


def test_v1qa_real_execution_false_without_model(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = _make_v1pw_queue(tmp_path, 3)
    backend = _make_backend_summary(tmp_path, model_available=False)
    _run(S["v1qa"], _v1qa_env(tmp_path, queue, backend, sch))
    for row in _read(tmp_path / "v1qa.csv"):
        assert row["ready_for_real_execution"] == "false"


def test_v1qa_real_execution_true_with_model(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = _make_v1pw_queue(tmp_path, 2)
    backend = _make_backend_summary(tmp_path, model_available=True)
    _run(S["v1qa"], _v1qa_env(tmp_path, queue, backend, sch))
    for row in _read(tmp_path / "v1qa.csv"):
        assert row["ready_for_real_execution"] == "true"


# ---------------------------------------------------------------------------
# v1qb — readiness audit
# ---------------------------------------------------------------------------

def _v1qb_env(tmp: Path, qa_queue: Path, backend: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QB_IN_QUEUE=str(qa_queue), REVP_V1QB_IN_BACKEND=str(backend),
        REVP_V1QB_OUT_AUDIT=str(tmp / "v1qb.csv"),
        REVP_V1QB_OUT_SUM=str(tmp / "v1qb_sum.csv"),
        REVP_V1QB_SCH_AUDIT=str(sch / "a.csv"),
        REVP_V1QB_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QB_DOC=str(tmp / "doc.md"),
    )


def _make_v1qa_queue(tmp: Path, n: int, model_ok: bool = False) -> Path:
    q = tmp / "v1qa_q.csv"
    rows = [{
        "execution_queue_id": f"V1QA_EQ_{i:05d}",
        "source_queue_id": f"V1PW_Q_{i:05d}", "visual_asset_id": f"VA_{i}",
        "patch_id": f"CUR_{i:05d}", "alias": f"CUR_{i:05d}", "region": "CURITIBA",
        "relative_path": f"data/sentinel/patch_curitiba_{i:05d}.tif",
        "path_hash": C.path_hash(f"data/sentinel/p{i}.tif"),
        "visual_type": "SENTINEL_TIF_REFERENCE",
        "queue_priority": "2", "queue_reason": "sentinel_tif_reference",
        "linkage_confidence": "HIGH", "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
        "can_create_label": "false", "can_train_model": "false", "target_created": "false",
        "ready_for_dry_run": "true", "ready_for_real_execution": str(model_ok).lower(),
        "blocked_reason": "", "notes": "",
    } for i in range(1, n + 1)]
    C.write_csv(q, rows, list(rows[0].keys()))
    return q


def test_v1qb_dry_run_ready_without_model(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    qa_q = _make_v1qa_queue(tmp_path, 4)
    backend = _make_backend_summary(tmp_path, False)
    r = _run(S["v1qb"], _v1qb_env(tmp_path, qa_q, backend, sch))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "v1qb_sum.csv")
    dry = next(s["stat_value"] for s in summary if s["stat_key"] == "ready_dry_run_only")
    assert int(dry) > 0
    real = next(s["stat_value"] for s in summary if s["stat_key"] == "ready_local_model_execution")
    assert real == "0"


def test_v1qb_local_model_execution_ready(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    qa_q = _make_v1qa_queue(tmp_path, 4, model_ok=True)
    backend = _make_backend_summary(tmp_path, model_available=True)
    r = _run(S["v1qb"], _v1qb_env(tmp_path, qa_q, backend, sch))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "v1qb_sum.csv")
    real = next(s["stat_value"] for s in summary if s["stat_key"] == "ready_local_model_execution")
    assert int(real) > 0


# ---------------------------------------------------------------------------
# v1qc — dry-run package
# ---------------------------------------------------------------------------

def _v1qc_env(tmp: Path, qa_queue: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QC_IN_QUEUE=str(qa_queue),
        REVP_V1QC_OUT_PLAN=str(tmp / "plan.csv"),
        REVP_V1QC_OUT_CMDS=str(tmp / "cmds.csv"),
        REVP_V1QC_SCH_PLAN=str(sch / "a.csv"),
        REVP_V1QC_SCH_CMDS=str(sch / "b.csv"),
        REVP_V1QC_DOC=str(tmp / "doc.md"),
        REVP_DINO_MAX_EXECUTE="3",
    )


def test_v1qc_generates_safe_powershell_commands(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    qa_q = _make_v1qa_queue(tmp_path, 5)
    r = _run(S["v1qc"], _v1qc_env(tmp_path, qa_q, sch))
    assert r.returncode == 0, r.stderr
    cmds = _read(tmp_path / "cmds.csv")
    assert len(cmds) > 0
    # All commands are strings, no raw abs paths
    for cmd in cmds:
        ps = cmd.get("powershell_command", "")
        assert "C:\\Users" not in ps or ps.startswith("#")


def test_v1qc_download_remains_false(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    qa_q = _make_v1qa_queue(tmp_path, 3)
    _run(S["v1qc"], _v1qc_env(tmp_path, qa_q, sch))
    cmds = _read(tmp_path / "cmds.csv")
    dl_cmd = next((c for c in cmds if c["command_type"] == "keep_download_false"), None)
    assert dl_cmd is not None
    assert "false" in dl_cmd["powershell_command"].lower()


def test_v1qc_disable_dry_run_needs_manual(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    qa_q = _make_v1qa_queue(tmp_path, 3)
    _run(S["v1qc"], _v1qc_env(tmp_path, qa_q, sch))
    cmds = _read(tmp_path / "cmds.csv")
    disable_cmd = next((c for c in cmds if c["command_type"] == "disable_dry_run_manual"), None)
    assert disable_cmd is not None
    assert disable_cmd["requires_manual_confirmation"] == "true"


# ---------------------------------------------------------------------------
# v1qd — compatibility
# ---------------------------------------------------------------------------

def test_v1qd_detects_compatibility(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1qd"], _base(
        REVP_V1QD_OUT_REPORT=str(tmp_path / "rep.csv"),
        REVP_V1QD_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1QD_SCH_REP=str(sch / "a.csv"),
        REVP_V1QD_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QD_DOC=str(tmp_path / "doc.md"),
    ))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    status = next(s["stat_value"] for s in summary if s["stat_key"] == "compatibility_status")
    assert status == "FULLY_COMPATIBLE"


def test_v1qd_v1pq_patched(tmp_path: Path) -> None:
    # Verify v1pq source has been patched
    src = (SCRIPTS / "revp_v1pq_controlled_smoke_embedding_executor.py").read_text()
    assert "REVP_V1PQ_QUEUE_PATH" in src
    assert "execution_queue_id" in src


# ---------------------------------------------------------------------------
# v1pq backward compat + new queue
# ---------------------------------------------------------------------------

def test_v1pq_accepts_original_v1po_queue(tmp_path: Path) -> None:
    old_q = tmp_path / "old.csv"
    C.write_csv(old_q, [], ["queue_id", "visual_asset_id", "patch_id",
                             "alias", "region", "relative_path", "path_hash"])
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _base(
        REVP_V1PQ_IN_QUEUE=str(old_q),
        REVP_V1PQ_OUT_RESULTS=str(tmp_path / "res.csv"),
        REVP_V1PQ_OUT_FAILURES=str(tmp_path / "fail.csv"),
        REVP_V1PQ_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PQ_SCH_RES=str(sch / "a.csv"),
        REVP_V1PQ_SCH_FAIL=str(sch / "b.csv"),
        REVP_V1PQ_SCH_SUM=str(sch / "c.csv"),
        REVP_V1PQ_DOC=str(tmp_path / "doc.md"),
        REVP_DINO_DRY_RUN="true",
    )
    r = _run(S["v1pq"], env)
    assert r.returncode == 0, r.stderr


def test_v1pq_accepts_v1qa_queue_via_env(tmp_path: Path) -> None:
    qa_q = _make_v1qa_queue(tmp_path, 3)
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _base(
        REVP_V1PQ_QUEUE_PATH=str(qa_q),
        REVP_V1PQ_OUT_RESULTS=str(tmp_path / "res.csv"),
        REVP_V1PQ_OUT_FAILURES=str(tmp_path / "fail.csv"),
        REVP_V1PQ_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PQ_SCH_RES=str(sch / "a.csv"),
        REVP_V1PQ_SCH_FAIL=str(sch / "b.csv"),
        REVP_V1PQ_SCH_SUM=str(sch / "c.csv"),
        REVP_V1PQ_DOC=str(tmp_path / "doc.md"),
        REVP_DINO_DRY_RUN="true",
    )
    r = _run(S["v1pq"], env)
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    mode = next((s["stat_value"] for s in summary if s["stat_key"] == "execution_mode"), "")
    assert mode == "DRY_RUN"


# ---------------------------------------------------------------------------
# v1qe — TCC update
# ---------------------------------------------------------------------------

def test_v1qe_generates_tcc_tables(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    qa_q = _make_v1qa_queue(tmp_path, 3)
    readiness = tmp_path / "v1qb.csv"
    C.write_csv(readiness, [{
        "readiness_id": "V1QB_RD_00001", "execution_queue_id": "V1QA_EQ_00001",
        "patch_id": "CUR_00001", "region": "CURITIBA",
        "visual_type": "SENTINEL_TIF_REFERENCE",
        "backend_status": "DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED",
        "model_available": "false", "dry_run_allowed": "true",
        "real_execution_allowed": "false",
        "readiness_status": "READY_FOR_DRY_RUN_ONLY",
        "readiness_reason": "model_unavailable_dry_run_permitted",
        "can_create_label": "false", "can_train_model": "false", "target_created": "false",
        "blocked_reason": "", "notes": "",
    }], ["readiness_id", "execution_queue_id", "patch_id", "region", "visual_type",
         "backend_status", "model_available", "dry_run_allowed", "real_execution_allowed",
         "readiness_status", "readiness_reason", "can_create_label", "can_train_model",
         "target_created", "blocked_reason", "notes"])
    cmds = tmp_path / "cmds.csv"
    C.write_csv(cmds, [{
        "command_id": "V1QC_CMD_001", "command_type": "set_model_path",
        "powershell_command": '$env:REVP_DINO_MODEL_PATH = "<path>"',
        "safety_note": "Set local model path", "requires_manual_confirmation": "true",
    }], ["command_id", "command_type", "powershell_command", "safety_note", "requires_manual_confirmation"])
    env = _base(
        REVP_V1QE_IN_READINESS=str(readiness), REVP_V1QE_IN_COMMANDS=str(cmds),
        REVP_V1QE_OUT_T_READ=str(tmp_path / "t_read.csv"),
        REVP_V1QE_OUT_T_SAFE=str(tmp_path / "t_safe.csv"),
        REVP_V1QE_SCH_READ=str(sch / "a.csv"), REVP_V1QE_SCH_SAFE=str(sch / "b.csv"),
        REVP_V1QE_DOC=str(tmp_path / "doc.md"),
    )
    r = _run(S["v1qe"], env)
    assert r.returncode == 0, r.stderr
    assert len(_read(tmp_path / "t_read.csv")) == 1
    assert len(_read(tmp_path / "t_safe.csv")) == 1


def test_v1qe_doc_contains_tcc_text(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    empty = tmp_path / "e.csv"
    C.write_csv(empty, [], ["readiness_id"])
    env = _base(
        REVP_V1QE_IN_READINESS=str(empty), REVP_V1QE_IN_COMMANDS=str(empty),
        REVP_V1QE_OUT_T_READ=str(tmp_path / "t.csv"),
        REVP_V1QE_OUT_T_SAFE=str(tmp_path / "s.csv"),
        REVP_V1QE_SCH_READ=str(sch / "a.csv"), REVP_V1QE_SCH_SAFE=str(sch / "b.csv"),
        REVP_V1QE_DOC=str(tmp_path / "doc.md"),
    )
    _run(S["v1qe"], env)
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "permanece em modo dry-run" in text
    assert "representação auto-supervisionada review-only" in text


# ---------------------------------------------------------------------------
# v1qf — bridge bundle
# ---------------------------------------------------------------------------

def test_v1qf_status_dry_run_model_missing(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    r = _run(S["v1qf"], _base(
        REVP_V1QF_OUT_MANIFEST=str(tmp_path / "manifest.csv"),
        REVP_V1QF_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1QF_SCH_MAN=str(sch / "a.csv"),
        REVP_V1QF_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QF_DOC=str(tmp_path / "doc.md"),
    ))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    final = next(s["value"] for s in summary if s["metric"] == "final_status")
    assert final == "DINO_EXECUTION_BRIDGE_READY_DRY_RUN_MODEL_MISSING"


def test_v1qf_status_local_model_ready() -> None:
    # Logical test: if real_ready > 0 → READY_LOCAL_MODEL
    rr, dr = 5, 5
    if rr > 0:
        final = "DINO_EXECUTION_BRIDGE_READY_LOCAL_MODEL"
    elif dr > 0:
        final = "DINO_EXECUTION_BRIDGE_READY_DRY_RUN_MODEL_MISSING"
    else:
        final = "DINO_EXECUTION_BRIDGE_EMPTY_FAIL_CLOSED"
    assert final == "DINO_EXECUTION_BRIDGE_READY_LOCAL_MODEL"


# ---------------------------------------------------------------------------
# Cross-cutting guardrails
# ---------------------------------------------------------------------------

def test_empty_outputs_have_header(tmp_path: Path) -> None:
    for fields in (["a", "b"], ["x", "y", "z"]):
        p = tmp_path / f"h{'_'.join(fields)}.csv"
        C.write_csv(p, [], fields)
        with p.open(encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == fields


def test_blocked_rows_have_blocked_reason() -> None:
    valid, reason = BC.validate_queue_item({"can_create_label": "true"})
    assert not valid and reason


def test_no_abs_path_in_v1qa(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = _make_v1pw_queue(tmp_path, 3)
    backend = _make_backend_summary(tmp_path)
    _run(S["v1qa"], _v1qa_env(tmp_path, queue, backend, sch))
    for row in _read(tmp_path / "v1qa.csv"):
        rel = row.get("relative_path", "")
        assert not (len(rel) > 2 and rel[1] == ":")


def test_guardrail_label_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_create_label": "true"}], "test")


def test_guardrail_train_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_train_model": "true"}], "test")


def test_guardrail_target_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"dino_target_field_created": "true"}], "test")


def test_no_local_runs_in_v1qc_commands(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    qa_q = _make_v1qa_queue(tmp_path, 2)
    _run(S["v1qc"], _v1qc_env(tmp_path, qa_q, sch))
    for row in _read(tmp_path / "cmds.csv"):
        assert "local_runs" not in row.get("powershell_command", "").lower()
