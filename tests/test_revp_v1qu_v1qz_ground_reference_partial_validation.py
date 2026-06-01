"""Tests for REV-P Protocol C v1qu-v1qz ground reference partial validation.

All script outputs are redirected to tmp_path; the real datasets/ tree is
never written. Pure helpers are tested by direct import.
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

v1qu = importlib.import_module("revp_v1qu_official_evidence_source_requirement_registry")
v1qv = importlib.import_module("revp_v1qv_event_patch_review_sampling_frame")
v1qw = importlib.import_module("revp_v1qw_double_review_packet_generator")
v1qx = importlib.import_module("revp_v1qx_observational_evidence_scoring_model")
v1qy = importlib.import_module("revp_v1qy_ground_reference_adjudication_decision_registry")
v1qz = importlib.import_module("revp_v1qz_ground_reference_partial_validation_bundle")


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
    """Point every OUT_*/SCHEMA_*/DOC* global of a module into tmp."""
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


def _sample_rows(n: int = 6) -> list[dict]:
    rows = []
    regions = ["RECIFE", "PET", "CURITIBA"]
    for i in range(n):
        r = {f: "" for f in v1qv.SAMPLE_FIELDS}
        r.update({
            "review_sample_id": f"V1QV_SMP_{i:04d}",
            "event_id": f"EV_{i}", "patch_id": f"REC_{10000+i:05d}",
            "alias": f"alias_{i}", "region": regions[i % 3],
            "hazard_type": "FLOOD", "evidence_status": "C2_REVIEW",
            "temporal_status": "UNKNOWN", "spatial_status": "UNKNOWN",
            "dino_queue_status": "NOT_IN_DINO_QUEUE",
            "source_requirement_status": "SEE_V1QU", "sampling_stratum": "C2_REVIEW_ONLY",
            "sample_priority": "5", "sample_reason": "test",
        })
        r.update(G.guardrail_row())
        rows.append(r)
    return rows


def _filled_responses(rsid: str, *, agree=True, official=True,
                      day=True, address=True, independent=True) -> list[dict]:
    """Build filled double-review responses for one sample."""
    quality = "fonte oficial SGB CPRM" if official else "blog noticia jornal"
    dec_a = "C3-candidate"
    dec_b = "C3-candidate" if agree else "C2"
    out = []
    for slot, dec in (("REVIEWER_A", dec_a), ("REVIEWER_B", dec_b)):
        answers = {
            "evidence_visible": "sim",
            "event_supported": "sim",
            "location_supported": "sim" if address else "nao",
            "timing_supported": "sim" if day else "nao",
            "source_quality": quality,
            "independent_source_present": "sim" if independent else "nao",
            "uncertainty_level": "baixo",
            "recommended_decision": dec,
            "uncertainty_notes": "",
        }
        for q, v in answers.items():
            out.append({
                "form_id": f"F_{rsid}_{slot}_{q}", "packet_id": f"P_{rsid}_{slot}",
                "review_sample_id": rsid, "reviewer_slot": slot,
                "question_key": q, "question_text": "", "answer_placeholder": "",
                "response_value": v, "review_only": "true",
                "dino_validates_event": "false", "notes": "",
            })
    return out


# ===========================================================================
# Common helpers — normalization
# ===========================================================================

def test_normalize_region_recife():
    assert G.normalize_region("Recife") == "RECIFE"

def test_normalize_region_petropolis():
    assert G.normalize_region("petrópolis") == "PET"

def test_normalize_region_curitiba():
    assert G.normalize_region("cwb") == "CURITIBA"

def test_normalize_region_unknown():
    assert G.normalize_region("") == "UNKNOWN"

def test_normalize_event_id():
    assert G.normalize_event_id(" ev 1 ") == "EV_1"

def test_normalize_patch_id():
    assert G.normalize_patch_id("rec_00001") == "REC_00001"

def test_normalize_alias_default():
    assert G.normalize_alias("") == "UNKNOWN_ALIAS"

def test_normalize_source_id():
    assert G.normalize_source_id("SGB / CPRM!") == "SGB_CPRM"


# ===========================================================================
# Source family classification
# ===========================================================================

def test_classify_cemaden():
    assert G.classify_source_family("CEMADEN alerta") == G.OFFICIAL_HYDROMETEOROLOGICAL

def test_classify_inmet():
    assert G.classify_source_family("INMET BDMEP chuva") == G.OFFICIAL_HYDROMETEOROLOGICAL

def test_classify_ana_hidroweb():
    assert G.classify_source_family("ANA HidroWeb estacao") == G.OFFICIAL_HYDROMETEOROLOGICAL

def test_classify_sgb_cprm():
    assert G.classify_source_family("SGB CPRM movimento de massa") == G.OFFICIAL_GEOLOGICAL

def test_classify_defesa_civil():
    assert G.classify_source_family("Defesa Civil COMPDEC") == G.OFFICIAL_CIVIL_DEFENSE

def test_classify_diario_oficial():
    assert G.classify_source_family("Diario Oficial decreto situacao de emergencia") == G.OFFICIAL_GOVERNMENT_PUBLICATION

def test_classify_news_secondary():
    assert G.classify_source_family("portal G1 noticia") == G.NEWS_MEDIA_SECONDARY

def test_classify_social_secondary():
    assert G.classify_source_family("twitter post") == G.SOCIAL_MEDIA_SECONDARY

def test_classify_unknown():
    assert G.classify_source_family("xyz") == G.UNKNOWN_SOURCE

def test_classify_mapbiomas_scientific():
    assert G.classify_source_family("MapBiomas uso e cobertura") == G.SCIENTIFIC_DATASET


# ===========================================================================
# Scoring
# ===========================================================================

def test_reliability_official_high():
    assert G.score_source_reliability(G.OFFICIAL_GEOLOGICAL) >= 0.9

def test_reliability_news_low():
    assert G.score_source_reliability(G.NEWS_MEDIA_SECONDARY) < 0.5

def test_temporal_day_full():
    assert G.score_temporal_precision("DAY") == 1.0

def test_temporal_year_low():
    assert G.score_temporal_precision("YEAR") == 0.3

def test_temporal_unknown_zero():
    assert G.score_temporal_precision("UNKNOWN") == 0.0

def test_spatial_point_full():
    assert G.score_spatial_precision("POINT") == 1.0

def test_spatial_none_zero():
    assert G.score_spatial_precision("NONE") == 0.0

def test_independence_levels():
    assert G.score_independence(0) == 0.0
    assert G.score_independence(2) == 0.8

def test_agreement_match():
    assert G.score_review_agreement("C3", "C3") == 1.0

def test_agreement_mismatch():
    assert G.score_review_agreement("C3", "C2") == 0.3

def test_composite_in_range():
    s = {k: 1.0 for k in G._COMPOSITE_WEIGHTS}
    assert abs(G.composite_score(s) - 1.0) < 1e-6


# ===========================================================================
# decision_from_scores
# ===========================================================================

def _strong_scores():
    s = {
        "source_reliability_score": 0.95, "temporal_precision_score": 1.0,
        "spatial_precision_score": 0.8, "provenance_score": 0.9,
        "independence_score": 0.8, "review_agreement_score": 1.0,
    }
    s["composite"] = G.composite_score(s)
    return s

def test_decision_secondary_never_c3():
    s = _strong_scores()
    d = G.decision_from_scores(s, G.NEWS_MEDIA_SECONDARY)
    assert "C3" not in d

def test_decision_unknown_blocked():
    assert G.decision_from_scores(_strong_scores(), G.UNKNOWN_SOURCE) == G.BLOCKED_INSUFFICIENT_EVIDENCE

def test_decision_official_strong_ready_supervisor():
    d = G.decision_from_scores(_strong_scores(), G.OFFICIAL_GEOLOGICAL)
    assert d == G.C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW

def test_decision_low_temporal_not_c3():
    s = _strong_scores()
    s["temporal_precision_score"] = 0.3
    s["composite"] = G.composite_score(s)
    assert G.decision_from_scores(s, G.OFFICIAL_GEOLOGICAL) == G.C2_REVIEW_ONLY_CANDIDATE

def test_decision_low_spatial_not_c3():
    s = _strong_scores()
    s["spatial_precision_score"] = 0.0
    s["composite"] = G.composite_score(s)
    assert G.decision_from_scores(s, G.OFFICIAL_GEOLOGICAL) == G.C2_REVIEW_ONLY_CANDIDATE


# ===========================================================================
# Path / guardrail helpers
# ===========================================================================

def test_detect_absolute_path_true():
    assert G.detect_absolute_path(r"C:\Users\x\file.tif")

def test_detect_absolute_path_false():
    assert not G.detect_absolute_path("datasets/x.csv")

def test_detect_local_runs():
    assert G.detect_local_runs_exposure("local_runs/foo")

def test_mask_path_redacts_absolute():
    out = G.mask_path(r"C:\Users\gabriela\file.tif")
    assert "C:" not in out and "gabriela" not in out

def test_mask_path_redacts_local_runs():
    assert "local_runs" not in G.mask_path("local_runs/x/y").lower() or "REDACTED" in G.mask_path("local_runs/x")

def test_guardrail_row_all_safe():
    r = G.guardrail_row()
    assert r["review_only"] == "true"
    for f in G.FORBIDDEN_TRUE_FIELDS:
        assert r[f] == "false"

def test_guardrail_formal_negative_false():
    assert G.guardrail_row()["formal_negative"] == "false"

def test_guardrail_absence_as_negative_false():
    assert G.guardrail_row()["absence_as_negative"] == "false"

def test_guardrail_dino_validates_false():
    assert G.guardrail_row()["dino_validates_event"] == "false"

def test_assert_no_forbidden_true_raises():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_train_model": "true"}], "t")

def test_write_csv_rejects_absolute_path(tmp_path):
    with pytest.raises(ValueError):
        G.write_csv_with_header(tmp_path / "x.csv", [{"a": r"C:\x"}], ["a"])

def test_write_csv_rejects_local_runs(tmp_path):
    with pytest.raises(ValueError):
        G.write_csv_with_header(tmp_path / "x.csv", [{"a": "local_runs/y"}], ["a"])

def test_write_csv_empty_has_header(tmp_path):
    p = tmp_path / "x.csv"
    G.write_csv_with_header(p, [], ["a", "b"])
    assert _header(p) == ["a", "b"]

def test_hash_short_stable():
    assert G.hash_short("x", 8) == G.hash_short("x", 8)

def test_safe_relpath_no_absolute():
    rp = G.safe_relpath(SCRIPTS / "revp_v1qu_v1qz_ground_reference_common.py")
    assert not G.detect_absolute_path(rp)


# ===========================================================================
# v1qu — source requirement registry
# ===========================================================================

def test_v1qu_runs_and_has_regions(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    rows = _read(tmp_path / v1qu.OUT_REQUIREMENTS.name)
    regions = {r["region"] for r in rows}
    assert {"RECIFE", "PET", "CURITIBA"}.issubset(regions)

def test_v1qu_requirement_blocks_c3(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    rows = _read(tmp_path / v1qu.OUT_REQUIREMENTS.name)
    assert any(r["blocks_c3"] == "true" for r in rows)

def test_v1qu_marks_source_required_not_local(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    rows = _read(tmp_path / v1qu.OUT_REQUIREMENTS.name)
    assert any(r["collection_status"] == "SOURCE_REQUIRED_NOT_LOCAL" for r in rows)

def test_v1qu_all_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    rows = _read(tmp_path / v1qu.OUT_REQUIREMENTS.name)
    assert all(r["review_only"] == "true" for r in rows)
    assert all(r["can_create_operational_label"] == "false" for r in rows)

def test_v1qu_schema_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    assert (tmp_path / v1qu.SCHEMA_REQUIREMENTS.name).exists()

def test_v1qu_doc_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    assert (tmp_path / v1qu.DOC.name).exists()

def test_v1qu_secondary_does_not_block_gate(monkeypatch, tmp_path):
    # News/social families are not present as blocks_c4 requirements
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    rows = _read(tmp_path / v1qu.OUT_REQUIREMENTS.name)
    assert all(r["preferred_source_family"] not in (G.NEWS_MEDIA_SECONDARY, G.SOCIAL_MEDIA_SECONDARY) for r in rows)


# ===========================================================================
# v1qv — sampling frame
# ===========================================================================

def test_v1qv_runs(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qv, tmp_path)
    v1qv.run()
    assert (tmp_path / v1qv.OUT_SAMPLE.name).exists()

def test_v1qv_stratifies_region(monkeypatch, tmp_path):
    frame = [{"region": "RECIFE", "frame_id": f"f{i}", "frame_priority": "5",
              "sampling_stratum": "C2_REVIEW_ONLY", "event_id": "e", "patch_id": "p",
              "alias": "a", "hazard_type": "FLOOD", "evidence_status": "C2",
              "temporal_status": "U", "spatial_status": "U", "dino_queue_status": "N",
              "source_requirement_status": "S", "notes": ""} for i in range(3)]
    frame += [dict(frame[0], region="PET", frame_id=f"g{i}") for i in range(3)]
    sample = v1qv.draw_sample(frame, sample_n=4, min_per_region=2)
    regions = {s["region"] for s in sample}
    assert "RECIFE" in regions and "PET" in regions

def test_v1qv_respects_min_per_region(monkeypatch, tmp_path):
    frame = []
    for reg in ("RECIFE", "PET"):
        for i in range(5):
            frame.append({"region": reg, "frame_id": f"{reg}{i}", "frame_priority": "3",
                          "sampling_stratum": "GENERIC_CANDIDATE", "event_id": "e",
                          "patch_id": "p", "alias": "a", "hazard_type": "FLOOD",
                          "evidence_status": "x", "temporal_status": "U", "spatial_status": "U",
                          "dino_queue_status": "N", "source_requirement_status": "S", "notes": ""})
    sample = v1qv.draw_sample(frame, sample_n=4, min_per_region=2)
    per = {}
    for s in sample:
        per[s["region"]] = per.get(s["region"], 0) + 1
    assert per.get("RECIFE", 0) >= 2 and per.get("PET", 0) >= 2

def test_v1qv_includes_blocked_control():
    frame = [{"region": "RECIFE", "frame_id": "b1", "frame_priority": "1",
              "sampling_stratum": "BLOCKED_CONTROL", "event_id": "e", "patch_id": "p",
              "alias": "a", "hazard_type": "FLOOD", "evidence_status": "BLOCKED",
              "temporal_status": "U", "spatial_status": "U", "dino_queue_status": "N",
              "source_requirement_status": "S", "notes": "blocked_control"}]
    sample = v1qv.draw_sample(frame, sample_n=4, min_per_region=1)
    assert any(s["sampling_stratum"] == "BLOCKED_CONTROL" for s in sample)

def test_v1qv_includes_c2():
    frame = [{"region": "RECIFE", "frame_id": "c1", "frame_priority": "5",
              "sampling_stratum": "C2_REVIEW_ONLY", "event_id": "e", "patch_id": "p",
              "alias": "a", "hazard_type": "FLOOD", "evidence_status": "C2",
              "temporal_status": "U", "spatial_status": "U", "dino_queue_status": "N",
              "source_requirement_status": "S", "notes": ""}]
    sample = v1qv.draw_sample(frame, sample_n=2, min_per_region=1)
    assert any(s["sampling_stratum"] == "C2_REVIEW_ONLY" for s in sample)

def test_v1qv_includes_contextual_gap():
    frame = [{"region": "PET", "frame_id": "g1", "frame_priority": "4",
              "sampling_stratum": "C1_CONTEXTUAL_GAP", "event_id": "e", "patch_id": "p",
              "alias": "a", "hazard_type": "LANDSLIDE", "evidence_status": "C1",
              "temporal_status": "U", "spatial_status": "U", "dino_queue_status": "N",
              "source_requirement_status": "S", "notes": ""}]
    sample = v1qv.draw_sample(frame, sample_n=2, min_per_region=1)
    assert any(s["sampling_stratum"] == "C1_CONTEXTUAL_GAP" for s in sample)

def test_v1qv_dino_not_required():
    # No DINO-queue rows; sampling still succeeds
    frame = [{"region": "RECIFE", "frame_id": "x1", "frame_priority": "2",
              "sampling_stratum": "GENERIC_CANDIDATE", "event_id": "e", "patch_id": "p",
              "alias": "a", "hazard_type": "FLOOD", "evidence_status": "x",
              "temporal_status": "U", "spatial_status": "U", "dino_queue_status": "NOT_IN_DINO_QUEUE",
              "source_requirement_status": "S", "notes": ""}]
    sample = v1qv.draw_sample(frame, sample_n=1, min_per_region=1)
    assert len(sample) == 1

def test_v1qv_dino_not_used_as_proof():
    frame = [{"region": "RECIFE", "frame_id": "d1", "frame_priority": "3",
              "sampling_stratum": "DINO_REVIEW_QUEUE", "event_id": "e", "patch_id": "p",
              "alias": "a", "hazard_type": "FLOOD", "evidence_status": "x",
              "temporal_status": "U", "spatial_status": "U",
              "dino_queue_status": "REVIEW_ONLY_REPRESENTATION",
              "source_requirement_status": "S", "notes": ""}]
    sample = v1qv.draw_sample(frame, sample_n=1, min_per_region=1)
    assert sample[0]["dino_validates_event"] == "false"
    assert "not_proof" in sample[0]["sample_reason"]

def test_v1qv_sample_all_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qv, tmp_path)
    v1qv.run()
    rows = _read(tmp_path / v1qv.OUT_SAMPLE.name)
    assert all(r["review_only"] == "true" for r in rows)


# ===========================================================================
# v1qw — double review packets
# ===========================================================================

def test_v1qw_creates_ab_slots(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qw, tmp_path)
    sample = _sample_rows(3)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, sample, v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qw, "IN_SAMPLE", sp)
    v1qw.run()
    manifest = _read(tmp_path / v1qw.OUT_MANIFEST.name)
    slots = {m["reviewer_slot"] for m in manifest}
    assert slots == {"REVIEWER_A", "REVIEWER_B"}
    assert len(manifest) == 6

def test_v1qw_forms_have_placeholders(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qw, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qw, "IN_SAMPLE", sp)
    v1qw.run()
    forms = _read(tmp_path / v1qw.OUT_FORMS.name)
    assert all(f["answer_placeholder"] == "<TO_BE_FILLED_BY_HUMAN_REVIEWER>" for f in forms)

def test_v1qw_forms_decision_not_filled(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qw, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qw, "IN_SAMPLE", sp)
    v1qw.run()
    forms = _read(tmp_path / v1qw.OUT_FORMS.name)
    decisions = [f for f in forms if f["question_key"] == "recommended_decision"]
    assert decisions and all(f["response_value"] == "" for f in decisions)

def test_v1qw_has_all_questions(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qw, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qw, "IN_SAMPLE", sp)
    v1qw.run()
    forms = _read(tmp_path / v1qw.OUT_FORMS.name)
    qkeys = {f["question_key"] for f in forms}
    for q in ("evidence_visible", "event_supported", "location_supported",
              "timing_supported", "source_quality", "independent_source_present",
              "uncertainty_level", "recommended_decision", "uncertainty_notes"):
        assert q in qkeys


# ===========================================================================
# v1qx — scoring model
# ===========================================================================

def test_v1qx_fail_closed_without_responses(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qx, tmp_path)
    monkeypatch.delenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", raising=False)
    out = v1qx.run()
    assert out["status"] == "REVIEW_NOT_COMPLETED_FAIL_CLOSED"
    assert _header(tmp_path / v1qx.OUT_SCORES.name)  # header present

def test_v1qx_scores_with_fixture(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qx, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qx, "IN_SAMPLE", sp)
    resp = tmp_path / "responses.csv"
    _write_csv(resp, _filled_responses("V1QV_SMP_0000"), v1qw.FORM_FIELDS)
    monkeypatch.setenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", str(resp))
    out = v1qx.run()
    assert out["scored"] == 1
    rows = _read(tmp_path / v1qx.OUT_SCORES.name)
    assert rows[0]["scoring_status"] == "SCORED_REVIEW_ONLY"

def test_v1qx_detects_agreement(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qx, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qx, "IN_SAMPLE", sp)
    resp = tmp_path / "responses.csv"
    _write_csv(resp, _filled_responses("V1QV_SMP_0000", agree=True), v1qw.FORM_FIELDS)
    monkeypatch.setenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", str(resp))
    v1qx.run()
    rows = _read(tmp_path / v1qx.OUT_SCORES.name)
    assert rows[0]["disagreement_flag"] == "false"
    assert float(rows[0]["review_agreement_score"]) == 1.0

def test_v1qx_detects_disagreement(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qx, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qx, "IN_SAMPLE", sp)
    resp = tmp_path / "responses.csv"
    _write_csv(resp, _filled_responses("V1QV_SMP_0000", agree=False), v1qw.FORM_FIELDS)
    monkeypatch.setenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", str(resp))
    v1qx.run()
    rows = _read(tmp_path / v1qx.OUT_SCORES.name)
    assert rows[0]["disagreement_flag"] == "true"
    dis = _read(tmp_path / v1qx.OUT_DISAGREE.name)
    assert len(dis) == 1 and dis[0]["needs_third_reviewer"] == "true"

def test_v1qx_blocks_weak_source(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qx, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qx, "IN_SAMPLE", sp)
    resp = tmp_path / "responses.csv"
    _write_csv(resp, _filled_responses("V1QV_SMP_0000", official=False), v1qw.FORM_FIELDS)
    monkeypatch.setenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", str(resp))
    v1qx.run()
    rows = _read(tmp_path / v1qx.OUT_SCORES.name)
    assert "C3" not in rows[0]["recommended_protocol_c_decision"]

def test_v1qx_blocks_low_temporal(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qx, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qx, "IN_SAMPLE", sp)
    resp = tmp_path / "responses.csv"
    _write_csv(resp, _filled_responses("V1QV_SMP_0000", day=False), v1qw.FORM_FIELDS)
    monkeypatch.setenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", str(resp))
    v1qx.run()
    rows = _read(tmp_path / v1qx.OUT_SCORES.name)
    assert "C3" not in rows[0]["recommended_protocol_c_decision"]

def test_v1qx_blocks_low_spatial(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qx, tmp_path)
    sp = tmp_path / "sample.csv"
    _write_csv(sp, _sample_rows(1), v1qv.SAMPLE_FIELDS)
    monkeypatch.setattr(v1qx, "IN_SAMPLE", sp)
    resp = tmp_path / "responses.csv"
    _write_csv(resp, _filled_responses("V1QV_SMP_0000", address=False), v1qw.FORM_FIELDS)
    monkeypatch.setenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", str(resp))
    v1qx.run()
    rows = _read(tmp_path / v1qx.OUT_SCORES.name)
    assert "C3" not in rows[0]["recommended_protocol_c_decision"]


# ===========================================================================
# v1qy — adjudication
# ===========================================================================

def _score_row(decision, *, rel=0.95, temp=1.0, spat=0.8, comp=0.9):
    r = {f: "" for f in v1qx.SCORE_FIELDS}
    r.update({
        "review_sample_id": "V1QV_SMP_0000", "event_id": "e", "patch_id": "p",
        "region": "PET", "source_reliability_score": f"{rel}",
        "temporal_precision_score": f"{temp}", "spatial_precision_score": f"{spat}",
        "composite_observational_score": f"{comp}",
        "recommended_protocol_c_decision": decision, "blocked_reason": "",
    })
    r.update(G.guardrail_row())
    return r

def test_v1qy_keeps_c1(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C1_CONTEXTUAL_ONLY)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    v1qy.run()
    rows = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    assert rows[0]["adjudication_decision"] == v1qy.KEEP_C1

def test_v1qy_keeps_c2(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C2_REVIEW_ONLY_CANDIDATE)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    v1qy.run()
    rows = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    assert rows[0]["adjudication_decision"] == v1qy.KEEP_C2

def test_v1qy_promotes_c3_needs_supervisor(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    v1qy.run()
    rows = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    assert rows[0]["adjudication_decision"] == v1qy.PROMOTE_C3
    assert rows[0]["supervisor_review_required"] == "true"

def test_v1qy_blocks_c3_low_temporal(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C3_REFERENCE_CANDIDATE_NEEDS_ADJUDICATION, temp=0.3)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    v1qy.run()
    rows = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    assert rows[0]["adjudication_decision"] == v1qy.BLOCK_C3_TEMPORAL
    assert rows[0]["blocked_reason"]

def test_v1qy_blocks_c3_low_spatial(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C3_REFERENCE_CANDIDATE_NEEDS_ADJUDICATION, spat=0.0)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    v1qy.run()
    rows = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    assert rows[0]["adjudication_decision"] == v1qy.BLOCK_C3_SPATIAL

def test_v1qy_formal_negative_default_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    v1qy.run()
    rows = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    assert rows[0]["formal_negative"] == "false"

def test_v1qy_absence_as_negative_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C2_REVIEW_ONLY_CANDIDATE)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    v1qy.run()
    rows = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    assert rows[0]["absence_as_negative"] == "false"

def test_v1qy_no_c4_opened(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qy, tmp_path)
    sp = tmp_path / "scores.csv"
    _write_csv(sp, [_score_row(G.C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW)], v1qx.SCORE_FIELDS)
    monkeypatch.setattr(v1qy, "IN_SCORES", sp)
    out = v1qy.run()
    assert out["c4"] == 0


# ===========================================================================
# v1qz — bundle
# ===========================================================================

def _run_full_pipeline(monkeypatch, tmp_path, *, with_responses=False):
    # v1qu
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    # v1qv
    _redirect(monkeypatch, v1qv, tmp_path)
    v1qv.run()
    # v1qw
    _redirect(monkeypatch, v1qw, tmp_path)
    monkeypatch.setattr(v1qw, "IN_SAMPLE", tmp_path / v1qv.OUT_SAMPLE.name)
    v1qw.run()
    # v1qx
    _redirect(monkeypatch, v1qx, tmp_path)
    monkeypatch.setattr(v1qx, "IN_SAMPLE", tmp_path / v1qv.OUT_SAMPLE.name)
    if with_responses:
        sample = _read(tmp_path / v1qv.OUT_SAMPLE.name)
        resp_rows = []
        for s in sample:
            resp_rows += _filled_responses(s["review_sample_id"])
        resp = tmp_path / "responses.csv"
        _write_csv(resp, resp_rows, v1qw.FORM_FIELDS)
        monkeypatch.setenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", str(resp))
    else:
        monkeypatch.delenv("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", raising=False)
    v1qx.run()
    # v1qy
    _redirect(monkeypatch, v1qy, tmp_path)
    monkeypatch.setattr(v1qy, "IN_SCORES", tmp_path / v1qx.OUT_SCORES.name)
    v1qy.run()
    # v1qz
    _redirect(monkeypatch, v1qz, tmp_path)
    monkeypatch.setattr(v1qz, "IN_REQUIREMENTS", tmp_path / v1qu.OUT_REQUIREMENTS.name)
    monkeypatch.setattr(v1qz, "IN_SAMPLE", tmp_path / v1qv.OUT_SAMPLE.name)
    monkeypatch.setattr(v1qz, "IN_PACKETS", tmp_path / v1qw.OUT_MANIFEST.name)
    monkeypatch.setattr(v1qz, "IN_SCORES", tmp_path / v1qx.OUT_SCORES.name)
    monkeypatch.setattr(v1qz, "IN_DISAGREE", tmp_path / v1qx.OUT_DISAGREE.name)
    monkeypatch.setattr(v1qz, "IN_ADJUDICATION", tmp_path / v1qy.OUT_REGISTRY.name)
    return v1qz.run()

def test_v1qz_generates_priorities(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path)
    pri = _read(tmp_path / v1qz.OUT_PRIORITIES.name)
    assert len(pri) >= 1

def test_v1qz_summary_labels_zero(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1qz.OUT_SUMMARY.name)}
    assert summ["labels_created"] == "0"

def test_v1qz_summary_targets_zero(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1qz.OUT_SUMMARY.name)}
    assert summ["targets_created"] == "0"

def test_v1qz_summary_ground_truth_zero(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1qz.OUT_SUMMARY.name)}
    assert summ["ground_truth_operational"] == "0"

def test_v1qz_fail_closed_without_reviews(monkeypatch, tmp_path):
    out = _run_full_pipeline(monkeypatch, tmp_path, with_responses=False)
    assert out["final_status"] == v1qz.ST_NOT_COMPLETED

def test_v1qz_c3_needs_supervisor_status(monkeypatch, tmp_path):
    out = _run_full_pipeline(monkeypatch, tmp_path, with_responses=True)
    # With strong fixture responses the official-source samples promote to C3
    if out["c3_supervisor"] > 0:
        assert out["final_status"] == v1qz.ST_C3_SUPERVISOR

def test_v1qz_qc_all_pass(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path, with_responses=True)
    qc = _read(tmp_path / v1qz.OUT_QC.name)
    assert all(c["passed"] == "true" for c in qc)

def test_v1qz_c3_candidates_supervisor_required(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path, with_responses=True)
    adj = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    for r in adj:
        if r["adjudication_decision"] == v1qy.PROMOTE_C3:
            assert r["supervisor_review_required"] == "true"

def test_v1qz_blocked_rows_have_reason(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path, with_responses=True)
    adj = _read(tmp_path / v1qy.OUT_REGISTRY.name)
    for r in adj:
        if r["adjudication_decision"].startswith("BLOCK"):
            assert r["blocked_reason"]

def test_v1qz_tcc_mandatory_sentence(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path)
    doc = (tmp_path / v1qz.DOC.name).read_text(encoding="utf-8")
    assert "nao cria" in doc and "ground truth operacional" in doc
    assert "DINO" in doc and "nao validam evento" in doc

def test_v1qz_tcc_table_exists(monkeypatch, tmp_path):
    _run_full_pipeline(monkeypatch, tmp_path)
    tcc = _read(tmp_path / v1qz.OUT_TCC.name)
    metrics = {r["metric"] for r in tcc}
    assert "final_status" in metrics and "labels_created" in metrics

def test_v1qz_c4_blocked_no_formal_negative(monkeypatch, tmp_path):
    out = _run_full_pipeline(monkeypatch, tmp_path, with_responses=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1qz.OUT_SUMMARY.name)}
    assert summ["c4_formal_negatives"] == "0"


# ===========================================================================
# Hygiene: no test writes to real datasets/
# ===========================================================================

def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1qu*")}
    _redirect(monkeypatch, v1qu, tmp_path)
    v1qu.run()
    after = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1qu*")}
    assert before == after  # untouched by redirected run

def test_schemas_written_in_tmp(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1qv, tmp_path)
    v1qv.run()
    assert (tmp_path / v1qv.SCHEMA_FRAME.name).exists()
