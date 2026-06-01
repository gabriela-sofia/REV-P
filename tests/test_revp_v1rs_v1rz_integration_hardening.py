"""Tests for REV-P Protocol C v1rs-v1rz integration/hardening block.

All script outputs are redirected to tmp_path; real datasets/ is never written.
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1rs_v1rz_integration_common as IC  # noqa: E402
import revp_v1qu_v1qz_ground_reference_common as G  # noqa: E402

v1rs = importlib.import_module("revp_v1rs_integrated_artifact_inventory")
v1rt = importlib.import_module("revp_v1rt_dependency_graph_execution_order")
v1ru = importlib.import_module("revp_v1ru_cross_block_guardrail_audit")
v1rv = importlib.import_module("revp_v1rv_commit_readiness_package")
v1rw = importlib.import_module("revp_v1rw_local_execution_runbook_generator")
v1rx = importlib.import_module("revp_v1rx_manual_evidence_collection_runbook")
v1ry = importlib.import_module("revp_v1ry_integration_test_runner_wrapper")
v1rz = importlib.import_module("revp_v1rz_integration_hardening_bundle")


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


def _redirect(monkeypatch, mod, tmp: Path) -> None:
    for name in dir(mod):
        if name.startswith(("OUT_", "SCHEMA_", "DOC")):
            val = getattr(mod, name)
            if isinstance(val, Path):
                monkeypatch.setattr(mod, name, tmp / val.name)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ===========================================================================
# Common helpers
# ===========================================================================

def test_common_empty_csv_header(tmp_path):
    IC.write_csv_with_header(tmp_path / "x.csv", [], ["a", "b"])
    assert _header(tmp_path / "x.csv") == ["a", "b"]

def test_common_detects_absolute_path():
    assert IC.detect_absolute_path(r"C:\Users\x\file.tif")

def test_common_masks_absolute_path():
    out = G.mask_path(r"C:\Users\gabriela\file.tif")
    assert "C:" not in out and "gabriela" not in out

def test_common_detects_forbidden_literal():
    assert IC.detect_forbidden_literal_exposure("local_runs/x/y")

def test_common_guardrail_raises():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_create_operational_label": "true"}], "t")

def test_common_classify_artifact_status_missing(tmp_path):
    assert IC.classify_artifact_status(tmp_path / "ghost.csv") == "MISSING"

def test_common_classify_artifact_status_header_only(tmp_path):
    p = tmp_path / "x.csv"
    _write_csv(p, [], ["a", "b"])
    assert IC.classify_artifact_status(p) == "HEADER_ONLY"

def test_common_infer_block_p0():
    assert "GROUND_REF" in IC.infer_block_from_filename("protocol_c_partial_v1qu.csv")

def test_common_infer_block_p1():
    assert "EXTERNAL_INTAKE" in IC.infer_block_from_filename("protocol_c_task_board_v1ra.csv")

def test_common_infer_block_p2():
    assert "REVIEW_GATE" in IC.infer_block_from_filename("protocol_c_review_v1rg.csv")

def test_common_infer_block_p3():
    assert "DASHBOARD" in IC.infer_block_from_filename("protocol_c_state_machine_v1rn.csv")


# ===========================================================================
# v1rs — artifact inventory
# ===========================================================================

def test_v1rs_includes_p0(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rs, tmp_path)
    v1rs.run()
    rows = _read(tmp_path / v1rs.OUT_INVENTORY.name)
    blocks = {r["block"] for r in rows}
    assert any("GROUND_REF" in b for b in blocks) or any("P0" in b for b in blocks)

def test_v1rs_includes_p1(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rs, tmp_path)
    v1rs.run()
    rows = _read(tmp_path / v1rs.OUT_INVENTORY.name)
    blocks = {r["block"] for r in rows}
    assert any("EXTERNAL_INTAKE" in b for b in blocks)

def test_v1rs_includes_p2(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rs, tmp_path)
    v1rs.run()
    rows = _read(tmp_path / v1rs.OUT_INVENTORY.name)
    blocks = {r["block"] for r in rows}
    assert any("REVIEW_GATE" in b for b in blocks)

def test_v1rs_includes_p3(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rs, tmp_path)
    v1rs.run()
    rows = _read(tmp_path / v1rs.OUT_INVENTORY.name)
    blocks = {r["block"] for r in rows}
    assert any("DASHBOARD" in b for b in blocks)

def test_v1rs_includes_dino(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rs, tmp_path)
    v1rs.run()
    rows = _read(tmp_path / v1rs.OUT_INVENTORY.name)
    blocks = {r["block"] for r in rows}
    assert any("DINO" in b for b in blocks)

def test_v1rs_schema_and_doc(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rs, tmp_path)
    v1rs.run()
    assert (tmp_path / v1rs.SCHEMA_INV.name).exists()
    assert (tmp_path / v1rs.DOC.name).exists()


# ===========================================================================
# v1rt — dependency graph
# ===========================================================================

def test_v1rt_generates_edges(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rt, tmp_path)
    out = v1rt.run()
    assert out["edges"] >= 5

def test_v1rt_execution_order_has_p0(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rt, tmp_path)
    v1rt.run()
    rows = _read(tmp_path / v1rt.OUT_ORDER.name)
    blocks = {r["block"] for r in rows}
    assert any("GROUND_REF" in b for b in blocks)

def test_v1rt_execution_order_has_p1_p2_p3(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rt, tmp_path)
    v1rt.run()
    rows = _read(tmp_path / v1rt.OUT_ORDER.name)
    blocks = " ".join(r["block"] for r in rows)
    assert "EXTERNAL_INTAKE" in blocks
    assert "REVIEW_GATE" in blocks
    assert "DASHBOARD" in blocks


# ===========================================================================
# v1ru — cross-block guardrail audit
# ===========================================================================

def test_v1ru_detects_bad_fixture(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_csv(ds / "protocol_c_bad_v1qu.csv",
               [{"can_train_model": "true"}], ["can_train_model"])
    _redirect(monkeypatch, v1ru, tmp_path)
    with monkeypatch.context() as m:
        m.setattr(v1ru, "_csv_paths", lambda: list(ds.glob("*.csv")))
        m.setattr(v1ru, "_doc_paths", lambda: [])
        v1ru.run()
    rows = _read(tmp_path / v1ru.OUT_AUDIT.name)
    viol = [r for r in rows if r["status"] == "VIOLATION"]
    assert viol

def test_v1ru_passes_safe_fixture(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_csv(ds / "protocol_c_safe_v1qz.csv",
               [{"review_only": "true", "can_train_model": "false"}],
               ["review_only", "can_train_model"])
    _redirect(monkeypatch, v1ru, tmp_path)
    with monkeypatch.context() as m:
        m.setattr(v1ru, "_csv_paths", lambda: list(ds.glob("*.csv")))
        m.setattr(v1ru, "_doc_paths", lambda: [])
        v1ru.run()
    rows = _read(tmp_path / v1ru.OUT_AUDIT.name)
    viol = [r for r in rows if r["status"] == "VIOLATION"]
    assert not viol


# ===========================================================================
# v1rv — commit readiness
# ===========================================================================

def test_v1rv_classifies_recommended(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rv, tmp_path)
    v1rv.run()
    assert (tmp_path / v1rv.OUT_RECOMMENDED.name).exists()

def test_v1rv_classifies_excluded(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rv, tmp_path)
    v1rv.run()
    assert (tmp_path / v1rv.OUT_EXCLUDED.name).exists()

def test_v1rv_no_git_add_in_code():
    src = Path(SCRIPTS / "revp_v1rv_commit_readiness_package.py").read_text(encoding="utf-8")
    # must not execute "git add" — only read operations
    import re
    # git add as a subprocess call (not in comment or string for doc)
    dangerous = re.findall(r'subprocess\.run\(.*git add', src)
    assert not dangerous

def test_v1rv_generates_powershell_commands(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rv, tmp_path)
    v1rv.run()
    doc = (tmp_path / v1rv.DOC.name).read_text(encoding="utf-8")
    assert "git add" in doc  # in doc as instructions, not executed


# ===========================================================================
# v1rw — local execution runbook
# ===========================================================================

def test_v1rw_contains_dino_model_path(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rw, tmp_path)
    monkeypatch.setattr(v1rw, "OUT_STEPS", tmp_path / v1rw.OUT_STEPS.name)
    monkeypatch.setattr(v1rw, "OUT_ENVVARS", tmp_path / v1rw.OUT_ENVVARS.name)
    v1rw.run()
    rows = _read(tmp_path / v1rw.OUT_ENVVARS.name)
    envvars = {r["env_var"] for r in rows}
    assert "REVP_DINO_MODEL_PATH" in envvars

def test_v1rw_contains_hf_hub_offline(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rw, tmp_path)
    monkeypatch.setattr(v1rw, "OUT_STEPS", tmp_path / v1rw.OUT_STEPS.name)
    monkeypatch.setattr(v1rw, "OUT_ENVVARS", tmp_path / v1rw.OUT_ENVVARS.name)
    v1rw.run()
    rows = _read(tmp_path / v1rw.OUT_ENVVARS.name)
    envvars = {r["env_var"] for r in rows}
    assert "HF_HUB_OFFLINE" in envvars

def test_v1rw_dry_run_first(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rw, tmp_path)
    monkeypatch.setattr(v1rw, "OUT_STEPS", tmp_path / v1rw.OUT_STEPS.name)
    monkeypatch.setattr(v1rw, "OUT_ENVVARS", tmp_path / v1rw.OUT_ENVVARS.name)
    v1rw.run()
    doc = (tmp_path / v1rw.OUT_RUNBOOK.name).read_text(encoding="utf-8")
    assert "dry" in doc.lower() and "first" in doc.lower()

def test_v1rw_warns_no_commit_rasters(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rw, tmp_path)
    monkeypatch.setattr(v1rw, "OUT_STEPS", tmp_path / v1rw.OUT_STEPS.name)
    monkeypatch.setattr(v1rw, "OUT_ENVVARS", tmp_path / v1rw.OUT_ENVVARS.name)
    v1rw.run()
    doc = (tmp_path / v1rw.OUT_RUNBOOK.name).read_text(encoding="utf-8")
    assert ".tif" in doc or ".npy" in doc


# ===========================================================================
# v1rx — evidence collection runbook
# ===========================================================================

def _rx_run(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rx, tmp_path)
    monkeypatch.setattr(v1rx, "OUT_EV_CHECKLIST", tmp_path / v1rx.OUT_EV_CHECKLIST.name)
    monkeypatch.setattr(v1rx, "OUT_RESP_CHECKLIST", tmp_path / v1rx.OUT_RESP_CHECKLIST.name)
    v1rx.run()

def test_v1rx_cemaden(monkeypatch, tmp_path):
    _rx_run(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1rx.OUT_EV_CHECKLIST.name)
    assert any("CEMADEN" in r["source_name"].upper() for r in rows)

def test_v1rx_ana_hidroweb(monkeypatch, tmp_path):
    _rx_run(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1rx.OUT_EV_CHECKLIST.name)
    assert any("ANA" in r["source_name"].upper() for r in rows)

def test_v1rx_inmet(monkeypatch, tmp_path):
    _rx_run(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1rx.OUT_EV_CHECKLIST.name)
    assert any("INMET" in r["source_name"].upper() for r in rows)

def test_v1rx_sgb_cprm(monkeypatch, tmp_path):
    _rx_run(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1rx.OUT_EV_CHECKLIST.name)
    assert any("SGB" in r["source_name"].upper() for r in rows)

def test_v1rx_defesa_civil(monkeypatch, tmp_path):
    _rx_run(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1rx.OUT_EV_CHECKLIST.name)
    assert any("DEFESA CIVIL" in r["source_name"].upper() for r in rows)

def test_v1rx_diario_oficial(monkeypatch, tmp_path):
    _rx_run(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1rx.OUT_EV_CHECKLIST.name)
    assert any("DIÁRIO" in r["source_name"].upper() or "OFICIAL" in r["source_name"].upper() for r in rows)


# ===========================================================================
# v1ry — integration test runner
# ===========================================================================

def test_v1ry_plan_only_by_default(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ry, tmp_path)
    monkeypatch.delenv("REVP_RUN_INTEGRATION_TESTS", raising=False)
    out = v1ry.run()
    assert not out["execute"]
    rows = _read(tmp_path / v1ry.OUT_PLAN.name)
    assert all(r["actual_status"] == "NOT_EXECUTED" for r in rows)

def test_v1ry_plan_has_all_suites(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ry, tmp_path)
    monkeypatch.delenv("REVP_RUN_INTEGRATION_TESTS", raising=False)
    v1ry.run()
    rows = _read(tmp_path / v1ry.OUT_PLAN.name)
    suites = " ".join(r["test_suite"] for r in rows)
    for expected in ("P0", "P1", "P2", "P3"):
        assert expected in suites


# ===========================================================================
# v1rz — integration hardening bundle
# ===========================================================================

def test_v1rz_creates_manifest(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    assert (tmp_path / v1rz.OUT_MANIFEST.name).exists()

def test_v1rz_creates_qc(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    qc = _read(tmp_path / v1rz.OUT_QC.name)
    assert len(qc) >= 4

def test_v1rz_creates_summary(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    assert (tmp_path / v1rz.OUT_SUMMARY.name).exists()

def test_v1rz_creates_next_actions(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    rows = _read(tmp_path / v1rz.OUT_ACTIONS.name)
    assert len(rows) >= 3

def _summary(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    return {r["metric"]: r["value"] for r in _read(tmp_path / v1rz.OUT_SUMMARY.name)}

def test_v1rz_labels_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["labels_created"] == "0"

def test_v1rz_targets_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["targets_created"] == "0"

def test_v1rz_ground_truth_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["ground_truth_operational_created"] == "0"

def test_v1rz_formal_negative_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["c4_formal_negatives"] == "0"

def test_v1rz_final_status_coherent(monkeypatch, tmp_path):
    out_val = _summary(monkeypatch, tmp_path)["final_status"]
    assert out_val in (v1rz.ST_READY, v1rz.ST_BLOCKED_GUARD, v1rz.ST_BLOCKED_ART,
                       v1rz.ST_WAIT_MODEL, v1rz.ST_WAIT_EV)

def test_v1rz_mandatory_sentence(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    doc = (tmp_path / v1rz.DOC.name).read_text(encoding="utf-8")
    assert "nao adiciona novos claims cientificos" in doc
    assert "permanece review-only" in doc


# ===========================================================================
# Guardrails (negative)
# ===========================================================================

def test_label_stays_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    rows = _read(tmp_path / v1rz.OUT_ACTIONS.name)
    assert all(r.get("can_create_operational_label") == "false" for r in rows)

def test_train_stays_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    rows = _read(tmp_path / v1rz.OUT_ACTIONS.name)
    assert all(r.get("can_train_model") == "false" for r in rows)

def test_target_stays_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    rows = _read(tmp_path / v1rz.OUT_ACTIONS.name)
    assert all(r.get("target_created") == "false" for r in rows)

def test_ground_truth_stays_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    rows = _read(tmp_path / v1rz.OUT_ACTIONS.name)
    assert all(r.get("ground_truth_operational") == "false" for r in rows)

def test_guardrail_dino_validates_true_raises():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"dino_validates_event": "true"}], "t")

def test_guardrail_absence_as_negative_raises():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"absence_as_negative": "true"}], "t")


# ===========================================================================
# Hygiene
# ===========================================================================

def test_empty_outputs_have_headers(tmp_path):
    for f in ["a.csv", "b.csv", "c.csv"]:
        IC.write_csv_with_header(tmp_path / f, [], ["x", "y"])
        assert _header(tmp_path / f) == ["x", "y"]

def test_schemas_and_docs_in_tmp(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rt, tmp_path)
    v1rt.run()
    assert (tmp_path / v1rt.SCHEMA_EDGES.name).exists()
    assert (tmp_path / v1rt.DOC.name).exists()

def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1rz*")}
    _redirect(monkeypatch, v1rz, tmp_path)
    v1rz.run()
    after = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1rz*")}
    assert before == after

def test_git_staged_remains_empty():
    import subprocess
    r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == ""
