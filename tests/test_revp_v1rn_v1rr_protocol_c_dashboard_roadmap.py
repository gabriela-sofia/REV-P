"""Tests for REV-P Protocol C v1rn-v1rr dashboard + roadmap (P3).

Script outputs are redirected to tmp_path; the real datasets/ tree is never
written. Inputs are read from the real (read-only) datasets, except the v1rq
claims-audit tests which use isolated fixture datasets in tmp_path.
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

import revp_v1qu_v1qz_ground_reference_common as G  # noqa: E402

v1rn = importlib.import_module("revp_v1rn_protocol_c_state_machine_registry")
v1ro = importlib.import_module("revp_v1ro_ground_reference_evidence_backlog")
v1rp = importlib.import_module("revp_v1rp_tcc_protocol_c_results_tables")
v1rq = importlib.import_module("revp_v1rq_methodological_claims_audit")
v1rr = importlib.import_module("revp_v1rr_scientific_roadmap_bundle")


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
# v1rn — state machine registry
# ===========================================================================

def test_v1rn_creates_registry(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rn, tmp_path)
    v1rn.run()
    assert (tmp_path / v1rn.OUT_REGISTRY.name).exists()

def _states(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rn, tmp_path)
    v1rn.run()
    return {r["state_id"]: r for r in _read(tmp_path / v1rn.OUT_REGISTRY.name)}

def test_v1rn_has_c1(monkeypatch, tmp_path):
    assert "C1" in _states(monkeypatch, tmp_path)

def test_v1rn_has_c2(monkeypatch, tmp_path):
    assert "C2" in _states(monkeypatch, tmp_path)

def test_v1rn_has_c3_candidate(monkeypatch, tmp_path):
    assert "C3_CANDIDATE" in _states(monkeypatch, tmp_path)

def test_v1rn_has_c4_blocked(monkeypatch, tmp_path):
    assert "C4_BLOCKED" in _states(monkeypatch, tmp_path)

def test_v1rn_blocks_c3_auto(monkeypatch, tmp_path):
    st = _states(monkeypatch, tmp_path)
    assert "AUTO_PROMOTE" in st["C3_CANDIDATE"]["forbidden_transitions"]

def test_v1rn_blocks_c4_without_formal(monkeypatch, tmp_path):
    st = _states(monkeypatch, tmp_path)
    forb = st["C4_BLOCKED"]["forbidden_transitions"]
    assert "C4_AUTO_OPEN" in forb and "ABSENCE_AS_NEGATIVE" in forb

def test_v1rn_no_state_is_operational_label(monkeypatch, tmp_path):
    st = _states(monkeypatch, tmp_path)
    assert all(r["is_operational_label"] == "false" for r in st.values())


# ===========================================================================
# v1ro — backlog
# ===========================================================================

def test_v1ro_creates_backlog(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ro, tmp_path)
    v1ro.run()
    assert (tmp_path / v1ro.OUT_BACKLOG.name).exists()

def test_v1ro_prioritizes_missing_source(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ro, tmp_path)
    v1ro.run()
    rows = _read(tmp_path / v1ro.OUT_BACKLOG.name)
    assert any(r["blocker"] == "EXTERNAL_SOURCE_NOT_LOCAL" for r in rows)

def test_v1ro_blocks_c3_for_critical(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ro, tmp_path)
    v1ro.run()
    rows = _read(tmp_path / v1ro.OUT_BACKLOG.name)
    assert any(r["blocks_c3"] == "true" for r in rows)

def test_v1ro_all_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ro, tmp_path)
    v1ro.run()
    rows = _read(tmp_path / v1ro.OUT_BACKLOG.name)
    assert all(r["review_only"] == "true" for r in rows)
    assert all(r["can_create_operational_label"] == "false" for r in rows)


# ===========================================================================
# v1rp — TCC tables
# ===========================================================================

def test_v1rp_creates_c_level(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rp, tmp_path)
    v1rp.run()
    rows = _read(tmp_path / v1rp.OUT_C_LEVEL.name)
    levels = {r["c_level"] for r in rows}
    assert any("C1" in l for l in levels) and any("C3" in l for l in levels)

def test_v1rp_dino_role_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rp, tmp_path)
    v1rp.run()
    rows = {r["dino_aspect"]: r["value"] for r in _read(tmp_path / v1rp.OUT_DINO.name)}
    assert rows["dino_validates_event"] == "false"

def test_v1rp_dino_not_proof(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rp, tmp_path)
    v1rp.run()
    rows = {r["dino_aspect"]: r["value"] for r in _read(tmp_path / v1rp.OUT_DINO.name)}
    assert rows["dino_can_create_label"] == "false"
    assert "REVIEW_ONLY" in rows["dino_role"]

def test_v1rp_c_level_not_label(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rp, tmp_path)
    v1rp.run()
    rows = _read(tmp_path / v1rp.OUT_C_LEVEL.name)
    assert all(r["is_operational_label"] == "false" for r in rows)


# ===========================================================================
# v1rq — methodological claims audit
# ===========================================================================

def _seed_disclaimer_docs(tmp: Path):
    docs = {
        "revp_v1qz_ground_reference_partial_validation_bundle.md": "embeddings DINO nao validam evento",
        "revp_v1rf_external_intake_bundle.md": "negativos formais por ausencia",
        "revp_v1rm_review_supervisor_gate_bundle.md": "permanece review-only",
    }
    for fname, phrase in docs.items():
        (tmp / fname).write_text(f"# doc\n\n{phrase}\n", encoding="utf-8")

def test_v1rq_detects_forbidden_claim(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_csv(ds / "protocol_c_bad_v1qz.csv",
               [{"x": "1", "ground_truth_operational": "true"}],
               ["x", "ground_truth_operational"])
    _redirect(monkeypatch, v1rq, tmp_path)
    _seed_disclaimer_docs(tmp_path)
    v1rq.run(datasets=ds)
    rows = _read(tmp_path / v1rq.OUT_AUDIT.name)
    viol = [r for r in rows if r["claim_type"] == "GROUND_TRUTH_OPERATIONAL" and r["status"] == "VIOLATION"]
    assert viol

def test_v1rq_safe_phrase_not_flagged(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    # safe methodological text in a cell, not a field=true
    _write_csv(ds / "protocol_c_safe_v1rz.csv",
               [{"note": "nao cria ground truth operacional", "ground_truth_operational": "false"}],
               ["note", "ground_truth_operational"])
    _redirect(monkeypatch, v1rq, tmp_path)
    _seed_disclaimer_docs(tmp_path)
    v1rq.run(datasets=ds)
    rows = _read(tmp_path / v1rq.OUT_AUDIT.name)
    viol = [r for r in rows if r["status"] == "VIOLATION"]
    assert not viol

def test_v1rq_summary_fail_on_violation(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_csv(ds / "protocol_c_bad_v1ra.csv",
               [{"can_train_model": "true"}], ["can_train_model"])
    _redirect(monkeypatch, v1rq, tmp_path)
    _seed_disclaimer_docs(tmp_path)
    v1rq.run(datasets=ds)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rq.OUT_SUMMARY.name)}
    assert summ["audit_status"] == "CLAIMS_AUDIT_VIOLATIONS_FOUND"

def test_v1rq_summary_pass_when_clean(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_csv(ds / "protocol_c_ok_v1rz.csv",
               [{"review_only": "true", "ground_truth_operational": "false"}],
               ["review_only", "ground_truth_operational"])
    _redirect(monkeypatch, v1rq, tmp_path)
    _seed_disclaimer_docs(tmp_path)
    v1rq.run(datasets=ds)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rq.OUT_SUMMARY.name)}
    assert summ["audit_status"] == "CLAIMS_AUDIT_CLEAN"

def test_v1rq_detects_absolute_path(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_csv(ds / "protocol_c_path_v1rz.csv",
               [{"p": "C:/Users/x/f.tif"}], ["p"])
    _redirect(monkeypatch, v1rq, tmp_path)
    _seed_disclaimer_docs(tmp_path)
    v1rq.run(datasets=ds)
    rows = _read(tmp_path / v1rq.OUT_AUDIT.name)
    viol = [r for r in rows if r["claim_type"] == "ABSOLUTE_PATH" and r["status"] == "VIOLATION"]
    assert viol


# ===========================================================================
# v1rr — roadmap bundle
# ===========================================================================

def test_v1rr_creates_manifest(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    assert (tmp_path / v1rr.OUT_MANIFEST.name).exists()

def test_v1rr_creates_summary(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    assert (tmp_path / v1rr.OUT_SUMMARY.name).exists()

def test_v1rr_creates_queue(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    assert (tmp_path / v1rr.OUT_QUEUE.name).exists()

def test_v1rr_creates_programming_steps(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    rows = _read(tmp_path / v1rr.OUT_STEPS.name)
    assert len(rows) >= 1

def test_v1rr_final_status_waiting(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    out = v1rr.run()
    assert out["final_status"] in (v1rr.ST_WAIT_INTAKE, v1rr.ST_WAIT_REVIEW)

def _summary(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    return {r["metric"]: r["value"] for r in _read(tmp_path / v1rr.OUT_SUMMARY.name)}

def test_v1rr_labels_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["labels_created"] == "0"

def test_v1rr_targets_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["targets_created"] == "0"

def test_v1rr_ground_truth_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["ground_truth_operational_created"] == "0"

def test_v1rr_formal_negative_zero(monkeypatch, tmp_path):
    assert _summary(monkeypatch, tmp_path)["c4_formal_negatives"] == "0"

def test_v1rr_next_actions_include_external(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    rows = _read(tmp_path / v1rr.OUT_QUEUE.name)
    types = {r["action_type"] for r in rows}
    assert "COLLECT_EXTERNAL_SOURCE" in types or "INTAKE_EXTERNAL_DOCUMENTS" in types

def test_v1rr_next_actions_include_double_review(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    rows = _read(tmp_path / v1rr.OUT_QUEUE.name)
    types = {r["action_type"] for r in rows}
    assert "RUN_DOUBLE_REVIEW" in types

def test_v1rr_programming_steps_have_script_range(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    rows = _read(tmp_path / v1rr.OUT_STEPS.name)
    assert all(r["next_script_range"] for r in rows)

def test_v1rr_mandatory_sentence(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    doc = (tmp_path / v1rr.DOC.name).read_text(encoding="utf-8")
    assert "aguarda documentos externos e respostas humanas" in doc
    assert "Nenhum rotulo operacional" in doc

def test_v1rr_queue_rows_have_notes_or_blocker(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    rows = _read(tmp_path / v1rr.OUT_QUEUE.name)
    assert all(r["blocker"] or r["notes"] for r in rows)


# ===========================================================================
# Generic: schemas / headers / hygiene
# ===========================================================================

def test_empty_output_has_header(tmp_path):
    G.write_csv_with_header(tmp_path / "x.csv", [], ["a", "b", "c"])
    assert _header(tmp_path / "x.csv") == ["a", "b", "c"]

def test_v1rn_schema_and_doc(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rn, tmp_path)
    v1rn.run()
    assert (tmp_path / v1rn.SCHEMA_REGISTRY.name).exists()
    assert (tmp_path / v1rn.DOC.name).exists()

def test_v1rr_schemas_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    assert (tmp_path / v1rr.SCHEMA_QUEUE.name).exists()
    assert (tmp_path / v1rr.SCHEMA_STEPS.name).exists()

def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1rr*")}
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    after = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1rr*")}
    assert before == after

def test_no_absolute_path_in_outputs(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rr, tmp_path)
    v1rr.run()
    for name in (v1rr.OUT_MANIFEST.name, v1rr.OUT_QUEUE.name, v1rr.OUT_STEPS.name):
        for r in _read(tmp_path / name):
            for v in r.values():
                assert not G.detect_absolute_path(str(v))

def test_no_forbidden_literal_in_outputs(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ro, tmp_path)
    v1ro.run()
    for r in _read(tmp_path / v1ro.OUT_BACKLOG.name):
        for v in r.values():
            assert not G.detect_local_runs_exposure(str(v))


# ===========================================================================
# Guardrails (negative)
# ===========================================================================

def test_guardrail_label_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_create_operational_label": "true"}], "t")

def test_guardrail_train_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_train_model": "true"}], "t")

def test_guardrail_target_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"target_created": "true"}], "t")

def test_guardrail_ground_truth_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"ground_truth_operational": "true"}], "t")

def test_guardrail_dino_validates_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"dino_validates_event": "true"}], "t")

def test_guardrail_absence_as_negative_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"absence_as_negative": "true"}], "t")
