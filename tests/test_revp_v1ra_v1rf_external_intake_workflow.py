"""Tests for REV-P Protocol C v1ra-v1rf external intake workflow.

All script outputs are redirected to tmp_path; the real datasets/ tree is
never written. Manual intake is simulated via fixture CSVs and env vars.
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

import revp_v1ra_v1rf_external_intake_common as IC  # noqa: E402
import revp_v1qu_v1qz_ground_reference_common as G  # noqa: E402

v1ra = importlib.import_module("revp_v1ra_external_collection_task_board")
v1rb = importlib.import_module("revp_v1rb_external_document_intake_template")
v1rc = importlib.import_module("revp_v1rc_external_document_intake_validator")
v1rd = importlib.import_module("revp_v1rd_event_candidate_builder_from_external_intake")
v1re = importlib.import_module("revp_v1re_external_event_patch_candidate_linker")
v1rf = importlib.import_module("revp_v1rf_external_intake_bundle")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _doc(**over) -> dict:
    base = {
        "document_id": "DOC_0001", "source_name": "Defesa Civil Petropolis",
        "source_family": "", "region": "PET", "hazard_type": "LANDSLIDE",
        "event_date_text": "15/02/2022", "event_location_text": "Alto da Serra",
        "url_or_reference": "https://defesacivil.example.gov.br/doc",
        "local_document_hash": "abc123", "access_date": "2026-05-31",
        "license_note": "dominio publico", "evidence_type": "occurrence_bulletin",
        "temporal_precision_claim": "DAY", "spatial_precision_claim": "ADDRESS",
        "reviewer_notes": "", "intake_status": "PENDING_VALIDATION",
    }
    base.update(over)
    return base


def _write_intake(tmp: Path, docs: list[dict]) -> Path:
    p = tmp / "intake.csv"
    _write_csv(p, docs, IC.INTAKE_FIELDS)
    return p


def _make_priorities(tmp: Path) -> Path:
    fields = ["priority_id", "region", "hazard_type", "evidence_need",
              "preferred_source_family", "preferred_source_name", "source_priority",
              "collection_status", "blocks_c3", "blocks_c4", "review_only", "notes"]
    rows = [
        {"priority_id": "P0", "region": "RECIFE", "hazard_type": "FLOOD",
         "evidence_need": "rainfall_alert", "preferred_source_family": G.OFFICIAL_HYDROMETEOROLOGICAL,
         "preferred_source_name": "CEMADEN", "source_priority": "P0",
         "collection_status": "SOURCE_REQUIRED_NOT_LOCAL", "blocks_c3": "true",
         "blocks_c4": "false", "review_only": "true", "notes": ""},
        {"priority_id": "P1", "region": "PET", "hazard_type": "LANDSLIDE",
         "evidence_need": "mass_movement_mapping", "preferred_source_family": G.OFFICIAL_GEOLOGICAL,
         "preferred_source_name": "SGB / CPRM", "source_priority": "P0",
         "collection_status": "SOURCE_REQUIRED_NOT_LOCAL", "blocks_c3": "true",
         "blocks_c4": "false", "review_only": "true", "notes": ""},
    ]
    p = tmp / "priorities.csv"
    _write_csv(p, rows, fields)
    return p


def _run_validate(monkeypatch, tmp, docs):
    _redirect(monkeypatch, v1rc, tmp)
    intake = _write_intake(tmp, docs)
    monkeypatch.setenv("REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH", str(intake))
    return v1rc.run()


# ===========================================================================
# Common helpers
# ===========================================================================

def test_common_writes_empty_csv_with_header(tmp_path):
    p = tmp_path / "x.csv"
    IC.write_csv_with_header(p, [], ["a", "b"])
    assert _header(p) == ["a", "b"]

def test_common_masks_absolute_path():
    out = IC.mask_path(r"C:\Users\gabriela\file.tif")
    assert "C:" not in out

def test_common_detects_local_runs():
    assert IC.detect_local_runs_exposure("local_runs/x")

def test_common_classify_cemaden():
    assert IC.classify_source_family("CEMADEN alerta") == G.OFFICIAL_HYDROMETEOROLOGICAL

def test_common_classify_inmet():
    assert IC.classify_source_family("INMET BDMEP") == G.OFFICIAL_HYDROMETEOROLOGICAL

def test_common_classify_ana():
    assert IC.classify_source_family("ANA HidroWeb") == G.OFFICIAL_HYDROMETEOROLOGICAL

def test_common_classify_sgb():
    assert IC.classify_source_family("SGB CPRM") == G.OFFICIAL_GEOLOGICAL

def test_common_classify_defesa_civil():
    assert IC.classify_source_family("Defesa Civil COMPDEC") == G.OFFICIAL_CIVIL_DEFENSE

def test_common_classify_diario_oficial():
    assert IC.classify_source_family("Diario Oficial decreto") == G.OFFICIAL_GOVERNMENT_PUBLICATION

def test_common_normalize_url_strips_tracking():
    assert "utm_source" not in IC.normalize_url("https://x.gov.br/a?utm_source=foo")

def test_common_detect_document_type_pdf():
    assert IC.detect_document_type("boletim.pdf") == "PDF_REPORT"

def test_common_validate_license_open():
    status, usable = IC.validate_license_access("dominio publico")
    assert usable and status == "LICENSE_OPEN"

def test_common_validate_license_unknown():
    status, usable = IC.validate_license_access("")
    assert not usable and status == "LICENSE_NOT_VERIFIED"


# ===========================================================================
# v1ra — task board
# ===========================================================================

def test_v1ra_builds_board_from_priorities(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ra, tmp_path)
    monkeypatch.setattr(v1ra, "IN_PRIORITIES", _make_priorities(tmp_path))
    v1ra.run()
    rows = _read(tmp_path / v1ra.OUT_BOARD.name)
    assert len(rows) == 2

def test_v1ra_suggests_queries_no_internet(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ra, tmp_path)
    monkeypatch.setattr(v1ra, "IN_PRIORITIES", _make_priorities(tmp_path))
    v1ra.run()
    rows = _read(tmp_path / v1ra.OUT_BOARD.name)
    assert all(r["search_query_suggested"] for r in rows)
    assert all(r["collection_status"] == "PENDING_MANUAL_COLLECTION" for r in rows)

def test_v1ra_all_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ra, tmp_path)
    monkeypatch.setattr(v1ra, "IN_PRIORITIES", _make_priorities(tmp_path))
    v1ra.run()
    rows = _read(tmp_path / v1ra.OUT_BOARD.name)
    assert all(r["review_only"] == "true" for r in rows)

def test_v1ra_doc_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ra, tmp_path)
    monkeypatch.setattr(v1ra, "IN_PRIORITIES", _make_priorities(tmp_path))
    v1ra.run()
    assert (tmp_path / v1ra.DOC.name).exists()


# ===========================================================================
# v1rb — template
# ===========================================================================

def test_v1rb_creates_template(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rb, tmp_path)
    monkeypatch.setattr(v1rb, "OUT_SCHEMA", tmp_path / v1rb.OUT_SCHEMA.name)
    v1rb.run()
    assert (tmp_path / v1rb.OUT_TEMPLATE.name).exists()

def test_v1rb_template_has_all_fields(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rb, tmp_path)
    monkeypatch.setattr(v1rb, "OUT_SCHEMA", tmp_path / v1rb.OUT_SCHEMA.name)
    v1rb.run()
    header = _header(tmp_path / v1rb.OUT_TEMPLATE.name)
    for f in ("document_id", "source_name", "region", "hazard_type",
              "event_date_text", "event_location_text", "url_or_reference",
              "license_note", "temporal_precision_claim", "spatial_precision_claim"):
        assert f in header

def test_v1rb_template_is_empty(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rb, tmp_path)
    monkeypatch.setattr(v1rb, "OUT_SCHEMA", tmp_path / v1rb.OUT_SCHEMA.name)
    v1rb.run()
    assert _read(tmp_path / v1rb.OUT_TEMPLATE.name) == []


# ===========================================================================
# v1rc — validator
# ===========================================================================

def test_v1rc_fail_closed_without_env(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rc, tmp_path)
    monkeypatch.delenv("REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH", raising=False)
    out = v1rc.run()
    assert out["status"] == "EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS"
    assert _header(tmp_path / v1rc.OUT_VALIDATION.name)

def test_v1rc_validates_fixture(monkeypatch, tmp_path):
    out = _run_validate(monkeypatch, tmp_path, [_doc()])
    assert out["status"] == "EXTERNAL_INTAKE_VALIDATION_PASS_REVIEW_ONLY"

def test_v1rc_blocks_missing_date(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, [_doc(event_date_text="")])
    rows = _read(tmp_path / v1rc.OUT_VALIDATION.name)
    date_checks = [r for r in rows if r["check_name"] == "required_field_event_date_text"]
    assert date_checks and date_checks[0]["status"] == "FAIL"

def test_v1rc_blocks_missing_location(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, [_doc(event_location_text="")])
    rows = _read(tmp_path / v1rc.OUT_VALIDATION.name)
    loc = [r for r in rows if r["check_name"] == "required_field_event_location_text"]
    assert loc and loc[0]["status"] == "FAIL"

def test_v1rc_blocks_license_unknown(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, [_doc(license_note="")])
    rows = _read(tmp_path / v1rc.OUT_VALIDATION.name)
    lic = [r for r in rows if r["check_name"] == "license_access_unknown"]
    assert lic and lic[0]["status"] == "FAIL"

def test_v1rc_missing_temporal_precision(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, [_doc(temporal_precision_claim="")])
    rows = _read(tmp_path / v1rc.OUT_VALIDATION.name)
    t = [r for r in rows if r["check_name"] == "missing_temporal_precision"]
    assert t and t[0]["status"] == "FAIL" and t[0]["blocks_c3"] == "true"

def test_v1rc_blocked_rows_have_reason(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, [_doc(event_date_text="")])
    rows = _read(tmp_path / v1rc.OUT_VALIDATION.name)
    for r in rows:
        if r["status"] == "FAIL":
            assert r["blocked_reason"]

def test_v1rc_formal_negative_false(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, [_doc()])
    rows = _read(tmp_path / v1rc.OUT_VALIDATION.name)
    assert all(r["formal_negative"] == "false" for r in rows)


# ===========================================================================
# v1rd — event candidate builder
# ===========================================================================

def _validate_then(monkeypatch, tmp, docs):
    _run_validate(monkeypatch, tmp, docs)
    _redirect(monkeypatch, v1rd, tmp)
    monkeypatch.setattr(v1rd, "IN_VALIDATION", tmp / v1rc.OUT_VALIDATION.name)
    return v1rd.run()

def test_v1rd_builds_candidates_review_only(monkeypatch, tmp_path):
    out = _validate_then(monkeypatch, tmp_path, [_doc()])
    assert out["candidates"] == 1
    rows = _read(tmp_path / v1rd.OUT_CANDIDATES.name)
    assert rows[0]["candidate_status"] == "REVIEW_ONLY_EXTERNAL_CANDIDATE"

def test_v1rd_no_confirmed_event(monkeypatch, tmp_path):
    _validate_then(monkeypatch, tmp_path, [_doc()])
    rows = _read(tmp_path / v1rd.OUT_CANDIDATES.name)
    assert all(r["ground_truth_operational"] == "false" for r in rows)
    assert all(r["can_create_operational_label"] == "false" for r in rows)

def test_v1rd_skips_blocked_document(monkeypatch, tmp_path):
    out = _validate_then(monkeypatch, tmp_path, [_doc(event_date_text="")])
    assert out["candidates"] == 0

def test_v1rd_waiting_without_intake(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rd, tmp_path)
    # empty validation
    IC.write_csv_with_header(tmp_path / v1rc.OUT_VALIDATION.name, [], v1rc.VALIDATION_FIELDS)
    monkeypatch.setattr(v1rd, "IN_VALIDATION", tmp_path / v1rc.OUT_VALIDATION.name)
    monkeypatch.delenv("REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH", raising=False)
    out = v1rd.run()
    assert out["status"] == "EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS"


# ===========================================================================
# v1re — event/patch linker
# ===========================================================================

def _make_patches(tmp: Path, region="PET") -> Path:
    fields = ["patch_id", "region"]
    rows = [{"patch_id": f"{region}_{10000+i:05d}", "region": region} for i in range(3)]
    p = tmp / "patches.csv"
    _write_csv(p, rows, fields)
    return p

def _candidates_then_link(monkeypatch, tmp, docs, region="PET"):
    _validate_then(monkeypatch, tmp, docs)
    _redirect(monkeypatch, v1re, tmp)
    monkeypatch.setattr(v1re, "IN_CANDIDATES", tmp / v1rd.OUT_CANDIDATES.name)
    monkeypatch.setattr(v1re, "IN_PATCHES", _make_patches(tmp, region))
    return v1re.run()

def test_v1re_builds_link_candidates_review_only(monkeypatch, tmp_path):
    _candidates_then_link(monkeypatch, tmp_path, [_doc()])
    rows = _read(tmp_path / v1re.OUT_LINKS.name)
    assert rows
    assert all(r["link_status"] in ("LINK_CANDIDATE_REVIEW_ONLY", "NO_PATCH_AVAILABLE_REVIEW_ONLY") for r in rows)

def test_v1re_no_scene_date_required(monkeypatch, tmp_path):
    # patches fixture has no scene_date column; linking still works
    _candidates_then_link(monkeypatch, tmp_path, [_doc()])
    rows = _read(tmp_path / v1re.OUT_LINKS.name)
    assert any(r["link_status"] == "LINK_CANDIDATE_REVIEW_ONLY" for r in rows)

def test_v1re_no_automatic_c3(monkeypatch, tmp_path):
    _candidates_then_link(monkeypatch, tmp_path, [_doc()])
    rows = _read(tmp_path / v1re.OUT_LINKS.name)
    assert all("C3" not in r["link_status"] for r in rows)
    assert all(r["dino_validates_event"] == "false" for r in rows)

def test_v1re_low_confidence_only(monkeypatch, tmp_path):
    _candidates_then_link(monkeypatch, tmp_path, [_doc()])
    rows = _read(tmp_path / v1re.OUT_LINKS.name)
    real = [r for r in rows if r["link_status"] == "LINK_CANDIDATE_REVIEW_ONLY"]
    assert real and all(r["link_confidence"] == "REVIEW_ONLY_LOW" for r in real)


# ===========================================================================
# v1rf — bundle
# ===========================================================================

def _full_pipeline(monkeypatch, tmp, *, with_intake=False):
    _redirect(monkeypatch, v1ra, tmp)
    monkeypatch.setattr(v1ra, "IN_PRIORITIES", _make_priorities(tmp))
    v1ra.run()
    _redirect(monkeypatch, v1rb, tmp)
    monkeypatch.setattr(v1rb, "OUT_SCHEMA", tmp / v1rb.OUT_SCHEMA.name)
    v1rb.run()
    if with_intake:
        _validate_then(monkeypatch, tmp, [_doc()])
        _redirect(monkeypatch, v1re, tmp)
        monkeypatch.setattr(v1re, "IN_CANDIDATES", tmp / v1rd.OUT_CANDIDATES.name)
        monkeypatch.setattr(v1re, "IN_PATCHES", _make_patches(tmp, "PET"))
        v1re.run()
    else:
        _redirect(monkeypatch, v1rc, tmp)
        monkeypatch.delenv("REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH", raising=False)
        v1rc.run()
        _redirect(monkeypatch, v1rd, tmp)
        monkeypatch.setattr(v1rd, "IN_VALIDATION", tmp / v1rc.OUT_VALIDATION.name)
        v1rd.run()
        _redirect(monkeypatch, v1re, tmp)
        monkeypatch.setattr(v1re, "IN_CANDIDATES", tmp / v1rd.OUT_CANDIDATES.name)
        monkeypatch.setattr(v1re, "IN_PATCHES", _make_patches(tmp, "PET"))
        v1re.run()
    _redirect(monkeypatch, v1rf, tmp)
    monkeypatch.setattr(v1rf, "IN_BOARD", tmp / v1ra.OUT_BOARD.name)
    monkeypatch.setattr(v1rf, "IN_TEMPLATE", tmp / v1rb.OUT_TEMPLATE.name)
    monkeypatch.setattr(v1rf, "IN_VALIDATION", tmp / v1rc.OUT_VALIDATION.name)
    monkeypatch.setattr(v1rf, "IN_CANDIDATES", tmp / v1rd.OUT_CANDIDATES.name)
    monkeypatch.setattr(v1rf, "IN_LINKS", tmp / v1re.OUT_LINKS.name)
    return v1rf.run()

def test_v1rf_works_without_intake(monkeypatch, tmp_path):
    out = _full_pipeline(monkeypatch, tmp_path, with_intake=False)
    assert out["qc_failed"] == 0
    assert out["final_status"] in (v1rf.ST_BOARD_READY, v1rf.ST_WAITING)

def test_v1rf_works_with_intake(monkeypatch, tmp_path):
    out = _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    assert out["final_status"] == v1rf.ST_CANDIDATES_READY
    assert out["candidates"] == 1

def test_v1rf_summary_labels_zero(monkeypatch, tmp_path):
    _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rf.OUT_SUMMARY.name)}
    assert summ["labels_created"] == "0"

def test_v1rf_summary_targets_zero(monkeypatch, tmp_path):
    _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rf.OUT_SUMMARY.name)}
    assert summ["targets_created"] == "0"

def test_v1rf_summary_ground_truth_zero(monkeypatch, tmp_path):
    _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rf.OUT_SUMMARY.name)}
    assert summ["ground_truth_operational_created"] == "0"

def test_v1rf_summary_formal_negative_zero(monkeypatch, tmp_path):
    _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rf.OUT_SUMMARY.name)}
    assert summ["c4_formal_negatives"] == "0"

def test_v1rf_qc_all_pass(monkeypatch, tmp_path):
    _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    qc = _read(tmp_path / v1rf.OUT_QC.name)
    assert all(c["passed"] == "true" for c in qc)

def test_v1rf_mandatory_sentence(monkeypatch, tmp_path):
    _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    doc = (tmp_path / v1rf.DOC.name).read_text(encoding="utf-8")
    assert "nao cria ground truth operacional" in doc
    assert "negativos formais por ausencia" in doc

def test_v1rf_tcc_table_exists(monkeypatch, tmp_path):
    _full_pipeline(monkeypatch, tmp_path, with_intake=True)
    tcc = _read(tmp_path / v1rf.OUT_TCC.name)
    metrics = {r["metric"] for r in tcc}
    assert "final_status" in metrics and "labels_created" in metrics


# ===========================================================================
# Guardrail detection (negative tests)
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


# ===========================================================================
# Hygiene
# ===========================================================================

def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1ra*")}
    _redirect(monkeypatch, v1ra, tmp_path)
    monkeypatch.setattr(v1ra, "IN_PRIORITIES", _make_priorities(tmp_path))
    v1ra.run()
    after = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1ra*")}
    assert before == after

def test_schemas_written_in_tmp(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ra, tmp_path)
    monkeypatch.setattr(v1ra, "IN_PRIORITIES", _make_priorities(tmp_path))
    v1ra.run()
    assert (tmp_path / v1ra.SCHEMA_BOARD.name).exists()
