"""Testes para SUSC-GT01 - Ground Reference Policy & Evidence State Taxonomy."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"


def _load_common():
    path = SCRIPTS / "susc_gt01_ground_reference_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("gt01_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def _taxonomy():
    return {r["state"]: r for r in _read(S.TAXONOMY_CSV)}


# --- 1. positive_strong valido (com todos os campos exigidos) ---------------
def test_positive_strong_valido():
    ps = _taxonomy()["positive_strong"]
    campos = set(ps["required_fields"].split(";"))
    assert {
        "event_date", "geometry_type", "source_authority",
        "patch_link_quality", "uncertainty_m", "qa_status",
    } <= campos
    assert ps["definition"].strip() and ps["allowed_use"].strip()
    assert ps["eligible_for_evaluation"] == "true"


# --- 2. positive_strong invalido sem geometria (validador acusa) ------------
def test_positive_strong_invalido_sem_geometria():
    tax = S.taxonomy_rows()
    for r in tax:
        if r["state"] == "positive_strong":
            r["required_fields"] = "event_date;source_authority;qa_status"  # sem geometry_type
    S.write_csv(S.TAXONOMY_CSV, tax)
    try:
        errors = S.validate_outputs(S.manifest_obj())
        assert any("positive_strong_sem_campo:geometry_type" in e for e in errors)
    finally:
        S.build_all()  # restaura estado valido


# --- 3. positive_provisional valido sem patch_link forte --------------------
def test_positive_provisional_sem_patch_link_forte():
    pp = _taxonomy()["positive_provisional"]
    # nao entra em avaliacao patch-level (sem geometria forte), mas calibra review-only
    assert pp["eligible_for_evaluation"] == "false"
    assert pp["eligible_for_calibration"] == "true"
    assert pp["is_negative"] == "false"


# --- 4. unlabeled como default ----------------------------------------------
def test_unlabeled_default():
    unl = _taxonomy()["unlabeled"]
    assert unl["is_default"] == "true"
    assert S.DEFAULT_STATE == "unlabeled"


# --- 5. unlabeled nao pode ser negative -------------------------------------
def test_unlabeled_nao_e_negative():
    unl = _taxonomy()["unlabeled"]
    assert unl["is_negative"] == "false"
    tax = S.taxonomy_rows()
    for r in tax:
        if r["state"] == "unlabeled":
            r["is_negative"] = "true"  # violacao
    S.write_csv(S.TAXONOMY_CSV, tax)
    try:
        assert "unlabeled_tratado_como_negative" in S.validate_outputs(S.manifest_obj())
    finally:
        S.build_all()


# --- 6. alert_only nao pode virar patch_link forte --------------------------
def test_alert_only_nao_vira_patch_link_forte():
    dec = {r["input_evidence_type"]: r for r in S.decision_rows()}
    assert dec["alert_only"]["max_patch_link_strength"] != S.STRONG_PATCH_LINK
    assert dec["alert_only"]["can_promote_to"] != "positive_strong"
    # transicao proibida declarada como allowed=false
    for r in S.transition_rows():
        if r["from_state"] == "alert_only" and r["to_state"] == "positive_strong":
            assert r["allowed"] == "false"


# --- 7. risk_area_not_event nao pode virar evento observado -----------------
def test_risk_area_nao_vira_evento():
    dec = {r["input_evidence_type"]: r for r in S.decision_rows()}
    assert dec["risk_area"]["can_promote_to"] != "positive_strong"
    assert dec["risk_area"]["max_patch_link_strength"] != S.STRONG_PATCH_LINK
    for r in S.transition_rows():
        if r["from_state"] == "risk_area_not_event" and r["to_state"] == "positive_strong":
            assert r["allowed"] == "false"


# --- 8. hard_negative_audited exige QA --------------------------------------
def test_hard_negative_exige_qa():
    hn = _taxonomy()["hard_negative_audited"]
    campos = set(hn["required_fields"].split(";"))
    assert {"observability", "exclusion_check", "qa_status"} <= campos


# --- 9. no_data exige blocking_reason ---------------------------------------
def test_no_data_exige_blocking_reason():
    nd = _taxonomy()["no_data"]
    assert "blocking_reason" in nd["required_fields"].split(";")
    assert nd["blocking_reason_required"] == "true"


# --- 10. eligible_for_training sempre false ---------------------------------
def test_eligible_for_training_sempre_false():
    for r in _read(S.TAXONOMY_CSV):
        assert r["eligible_for_training"] == "false"
    for r in S.decision_rows():
        assert r["eligible_for_training"] == "false"
    assert S.GLOBAL_GUARDRAILS["eligible_for_training"] == "false"


# --- 11. eligible_for_ground_truth sempre false -----------------------------
def test_eligible_for_ground_truth_sempre_false():
    for r in _read(S.TAXONOMY_CSV):
        assert r["eligible_for_ground_truth"] == "false"
    for r in S.decision_rows():
        assert r["eligible_for_ground_truth"] == "false"
    assert S.GLOBAL_GUARDRAILS["eligible_for_ground_truth"] == "false"


# --- 12. score_v7_candidate sempre false ------------------------------------
def test_score_v7_candidate_sempre_false():
    for r in _read(S.TAXONOMY_CSV):
        assert r["score_v7_candidate"] == "false"
    for r in S.decision_rows():
        assert r["score_v7_candidate"] == "false"
    assert S.GLOBAL_GUARDRAILS["score_v7_candidate"] == "false"


# --- garantias adicionais ----------------------------------------------------
def test_todos_estados_tem_definicao_e_allowed_use():
    tax = _taxonomy()
    for state in S.STATE_NAMES:
        assert state in tax
        assert tax[state]["definition"].strip()
        assert tax[state]["allowed_use"].strip()


def test_petropolis_separa_fenomeno():
    policy = json.loads(S.POLICY_JSON.read_text(encoding="utf-8"))
    sep = policy["petropolis_phenomenon_separation"]
    assert sep["required"] is True
    assert set(sep["phenomena"]) == {"flood", "flash_flood", "landslide", "mixed_flood_landslide"}


def test_nao_ha_claim_de_previsao_operacional():
    for path in (S.REPORT, S.POLICY_JSON, S.NOT_GT_CSV):
        txt = path.read_text(encoding="utf-8")
        assert S.operational_forecast_violations(txt) == []


def test_score_v6_intacto_e_sem_score_v7():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["score_v6_changed"] is False
    assert manifest["score_v7_created"] is False


def test_validador_completo_passa():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []
