"""Testes para SUSC-GT07 - Protocolo de QA Humano para Canarios Observacionais."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
SUSC = ROOT / "outputs_public" / "suscetibilidade"


def _load(name, filename):
    path = SCRIPTS / filename
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load("gt07_common", "susc_gt07_human_qa_canary_protocol_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def _fields(**over):
    base = {k: "" for k in ("phenomenon_type", "region", "city", "patch_id",
                            "geometry_acquisition_status", "geometry_type",
                            "resolved_event_date", "window_status")}
    base.update(over)
    return base


READY = dict(phenomenon_type="hydrological_flood", city="recife", region="Pina",
             patch_id="S18A_PATCH_0301", geometry_acquisition_status="geometria_parcial_resolvivel",
             geometry_type="point", resolved_event_date="2022-05-24", window_status="usable")


# 1. Leitura dos outputs GT06 ------------------------------------------------
def test_le_outputs_gt06():
    can = _read(SUSC / "susc_gt06_canarios_prioritarios.csv")
    fila = _read(S.QUEUE_CSV)
    assert len(fila) == len(can)
    assert fila[0]["qa_queue_id"].startswith("QAQ_")


# 2. apenas selecionados entram na fila --------------------------------------
def test_apenas_selecionados_na_fila():
    pack = {p["sar_target_id"]: p for p in _read(SUSC / "susc_gt06_pacote_canarios_sar.csv")}
    for q in _read(S.QUEUE_CSV):
        assert pack[q["sar_target_id"]]["selected_for_canary_pack"] == "true"


# 3. fila no maximo 5 --------------------------------------------------------
def test_fila_max_5():
    assert len(_read(S.QUEUE_CSV)) <= 5


# 4-5. current_qa_status nunca accepted/rejected -----------------------------
def test_current_qa_nunca_accepted_rejected():
    for q in _read(S.QUEUE_CSV):
        assert q["current_qa_status"] not in ("accepted", "rejected")
        assert q["current_qa_status"] in S.CURRENT_QA_ALLOWED


# 6. QA real nao marcado como executado --------------------------------------
def test_qa_nao_executado():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["qa_executado"] == 0
    assert manifest["accepted_now"] == 0
    assert manifest["rejected_now"] == 0


# 7. qa_can_be_completed_now=false sem footprint/visual ----------------------
def test_qa_now_false_sem_footprint():
    form = {f["qa_queue_id"]: f for f in _read(S.FORM_CSV)}
    for q in _read(S.QUEUE_CSV):
        if q["qa_can_be_completed_now"] == "true":
            f = form[q["qa_queue_id"]]
            assert f["footprint_available"] == "true" or f["visual_artifact_available"] == "true"
        else:
            assert q["qa_can_be_completed_now"] == "false"


# 8. checklist contem grupos obrigatorios ------------------------------------
def test_checklist_grupos():
    grupos = {c["checklist_group"] for c in _read(S.CHECKLIST_CSV)}
    for g in S.CHECKLIST_GROUPS:
        assert g in grupos


# 9. formulario contem campos-chave ------------------------------------------
def test_formulario_campos():
    cols = set(_read(S.FORM_CSV)[0].keys())
    for c in ("phenomenon_confirmed", "temporal_match_confirmed", "spatial_match_confirmed",
              "source_quality_confirmed", "permanent_water_checked", "leakage_check_passed",
              "uncertainty_m_reviewed"):
        assert c in cols


# 10-12. matriz nunca permite treino/GT/score_v7 -----------------------------
def test_matriz_nunca_permite():
    for r in _read(S.MATRIX_CSV):
        assert r["can_enable_training"] == "false"
        assert r["can_enable_ground_truth"] == "false"
        assert r["can_enable_score_v7"] == "false"


# 13-14. accepted/rejected now sempre 0 --------------------------------------
def test_accepted_rejected_zero():
    for r in _read(S.SUMMARY_CSV):
        assert r["accepted_now_count"] == "0"
        assert r["rejected_now_count"] == "0"


# 15-17. guardrails sempre false ---------------------------------------------
def test_guardrails_false():
    for q in _read(S.QUEUE_CSV):
        assert q["eligible_for_training"] == "false"
        assert q["eligible_for_ground_truth"] == "false"
        assert q["score_v7_candidate"] == "false"


# 18. nenhum canario vira positive_strong ------------------------------------
def test_nenhum_positive_strong():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_promovidos"] == 0
    for q in _read(S.QUEUE_CSV):
        assert "positive_strong" not in q["rationale_pt"].lower()


# 19. relatorio publico em portugues -----------------------------------------
def test_relatorio_portugues():
    h1 = S._report_h1(S.REPORT.read_text(encoding="utf-8"))
    assert S.title_is_portuguese(h1)


# 20. relatorio declara que QA nao foi executado -----------------------------
def test_relatorio_declara_qa_nao_executado():
    txt = S.REPORT.read_text(encoding="utf-8").lower()
    assert "não executa o qa real" in txt or "sem executar o qa real" in txt


# 21. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.QUEUE_CSV, S.FORM_CSV, S.CHECKLIST_CSV, S.MATRIX_CSV, S.CRITERIA_CSV,
            S.BLOCKERS_CSV, S.SUMMARY_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 22-23. schemas aceitam validos e rejeitam invalidos ------------------------
def test_schema_aceita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_QUEUE.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    for r in _read(S.QUEUE_CSV):
        assert list(v.iter_errors(r)) == []


def test_schema_rejeita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_QUEUE.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    inv = dict(_read(S.QUEUE_CSV)[0])
    inv["current_qa_status"] = "accepted"  # fora do enum permitido
    assert list(v.iter_errors(inv))
    inv2 = dict(_read(S.QUEUE_CSV)[0])
    inv2["eligible_for_training"] = "true"
    assert list(v.iter_errors(inv2))


# 24. GT07 nao modifica outputs GT01-06 --------------------------------------
def test_nao_modifica_marcos_anteriores():
    import hashlib

    def digest():
        h = {}
        for p in (sorted(SUSC.glob("susc_gt0[123456]_*"))
                  + sorted((ROOT / "schemas" / "suscetibilidade").glob("susc_gt0[123456]_*"))):
            h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return h

    before = digest()
    S.build_all()
    assert before == digest()


# 25. manifesto lista sha256 -------------------------------------------------
def test_manifesto_sha256():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    assert len(arts) == 12  # 8 publicos + 4 schemas
    for a in arts:
        assert len(a["sha256"]) == 64


# 26. recomendacao GT08 dos bloqueios reais ----------------------------------
def test_recomendacao_gt08():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["proximo_marco"].startswith("GT08")
    assert "Sentinel-1" in manifest["proximo_marco"]  # bloqueio por footprint


# 27. leakage temporal e criterio obrigatorio --------------------------------
def test_leakage_obrigatorio():
    leak_check = [c for c in _read(S.CHECKLIST_CSV) if c["checklist_group"] == "leakage"]
    assert leak_check
    assert all(c["required_for_acceptance"] == "true" for c in leak_check)
    crit = [c for c in _read(S.CRITERIA_CSV) if c["criterion_type"] == "leakage"]
    assert crit


# 28. accepted futuro e apenas review-only -----------------------------------
def test_accepted_futuro_review_only():
    aceitar = [r for r in _read(S.MATRIX_CSV)
               if r["decision_option"] == "aceitar_como_referencia_observacional_forte_review_only"]
    assert aceitar
    for r in aceitar:
        assert r["can_enable_evaluation_review_only"] == "true"
        assert r["can_enable_training"] == "false"
        assert r["can_enable_ground_truth"] == "false"
        assert "review-only" in r["allowed_use_after_decision"].lower()


# 29. formulario futuro tem placeholders, nao decisoes reais -----------------
def test_formulario_placeholders():
    for f in _read(S.FORM_CSV):
        assert f["qa_status_after_review"] == "not_reviewed"
        assert f["reviewer_decision"] in ("a_preencher", "")
        assert f["footprint_available"] == "nao_verificado"


# 30. canarios continuam not_training/not_ground_truth -----------------------
def test_canarios_not_training_not_ground_truth():
    for q in _read(S.QUEUE_CSV):
        assert q["trainable"] == "false"
        assert q["eligible_for_training"] == "false"
        assert q["eligible_for_ground_truth"] == "false"


# --- estados sinteticos ------------------------------------------------------
def test_classify_estados():
    assert S.classify_qa_prep(_fields(**READY), selected=True) == "pronto_para_qa_futuro"
    assert S.classify_qa_prep(_fields(**READY), selected=False) == "fora_da_fila_qa"
    petro = {**READY, "city": "Petropolis", "region": "petropolis", "phenomenon_type": "mixed_flood_landslide"}
    assert S.classify_qa_prep(_fields(**petro), selected=True) == "aguardando_separacao_fenomenologica"
    no_geom = {**READY, "patch_id": "", "geometry_acquisition_status": "geometria_ausente", "geometry_type": ""}
    assert S.classify_qa_prep(_fields(**no_geom), selected=True) == "aguardando_geometria_ou_aoi"
    no_win = {**READY, "window_status": "blocked_temporal_resolution_insufficient"}
    assert S.classify_qa_prep(_fields(**no_win), selected=True) == "bloqueado_para_qa"


# extra: validador completo passa --------------------------------------------
def test_validador_completo_passa():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []
