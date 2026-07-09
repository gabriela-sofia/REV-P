"""Testes para SUSC-GT06 - Preparacao de Canarios SAR sem Execucao."""

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


S = _load("gt06_common", "susc_gt06_sar_canary_preparation_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def _fields(**over):
    base = {k: "" for k in ("temporal_precision", "window_status", "source_authority",
                            "geometry_type", "phenomenon_type", "geometry_acquisition_status",
                            "region", "city", "resolved_event_date", "patch_id", "footprint_id",
                            "geometry_id", "geometry_priority_class", "can_advance_to_qa_future")}
    base.update(over)
    return base


def _elig(**over):
    return S.classify_eligibility(S.analyze_sar(_fields(**over)))


DATED = dict(temporal_precision="exact_day", window_status="usable",
             resolved_event_date="2022-05-24", city="recife", region="Pina")


# 1. Leitura dos outputs GT05 ------------------------------------------------
def test_le_outputs_gt05():
    g5 = _read(SUSC / "susc_gt05_pacote_aquisicao_geometria_oficial.csv")
    pack = _read(S.PACK_CSV)
    assert len(pack) == len(g5)
    assert pack[0]["sar_target_id"].startswith("SAR_")


# 2. datado + janela + requer_footprint vira elegivel SAR --------------------
def test_datado_requer_footprint_elegivel():
    el = _elig(**DATED, geometry_acquisition_status="requer_footprint_tecnico_futuro",
               source_authority="defesa_civil", patch_id="P1", phenomenon_type="flood")
    assert el in ("elegivel_canario_sar_prioritario", "elegivel_canario_sar_secundario")


# 3. temporal unknown bloqueia SAR -------------------------------------------
def test_unknown_bloqueia():
    assert _elig(temporal_precision="unknown", window_status="blocked_temporal_resolution_insufficient") == "bloqueado_temporalmente"


# 4. ausencia de janela bloqueia SAR -----------------------------------------
def test_sem_janela_bloqueia():
    assert _elig(temporal_precision="exact_day", window_status="blocked_temporal_resolution_insufficient") == "bloqueado_temporalmente"


# 5. sem patch/AOI nao pode ser selecionado ----------------------------------
def test_sem_patch_aoi_nao_selecionado():
    for p in _read(S.PACK_CSV):
        if p["selected_for_canary_pack"] == "true":
            has_patch = p["patch_id"] != "not_available"
            has_aoi = p["geometry_acquisition_status"] in ("geometria_parcial_resolvivel", "geometria_forte_existente")
            assert has_patch or has_aoi


# 6. Petropolis mixed exige separacao ----------------------------------------
def test_petropolis_mixed_separacao():
    el = _elig(**{**DATED, "city": "Petropolis", "region": "petropolis"},
               phenomenon_type="mixed_flood_landslide", patch_id="P1")
    assert el == "requer_separacao_fenomenologica"


# 7-9. fontes fracas nao viram prioritario -----------------------------------
@pytest.mark.parametrize("weak", ["risk_area_not_event", "alert_only", "news_only", "city_level_record"])
def test_fonte_fraca_nao_prioritario(weak):
    el = _elig(**DATED, source_authority=weak, patch_id="P1", phenomenon_type="flood")
    assert el != "elegivel_canario_sar_prioritario"


# 10. score sempre 0..100 -----------------------------------------------------
def test_score_intervalo():
    for p in _read(S.PACK_CSV):
        assert 0 <= int(p["sar_canary_priority_score"]) <= 100


# 11. classe na lista --------------------------------------------------------
def test_classe_valida():
    for p in _read(S.PACK_CSV):
        assert p["sar_canary_priority_class"] in S.SAR_PRIORITY_CLASSES


# 12. elegibilidade na lista -------------------------------------------------
def test_elegibilidade_valida():
    for p in _read(S.PACK_CSV):
        assert p["sar_canary_eligibility"] in S.SAR_ELIGIBILITY


# 13. selecao no maximo 5 ----------------------------------------------------
def test_selecao_max_5():
    sel = [p for p in _read(S.PACK_CSV) if p["selected_for_canary_pack"] == "true"]
    assert len(sel) <= 5
    assert len(_read(S.CANARY_CSV)) <= 5


# 14. can_execute_sar_now sempre false ---------------------------------------
def test_can_execute_sar_now_false():
    for e in _read(S.ELIG_CSV):
        assert e["can_execute_sar_now"] == "false"
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["can_execute_sar_now_true_count"] == 0


# 15-16. flags de nao-execucao/nao-rede sempre true --------------------------
def test_no_sar_no_network_true():
    for e in _read(S.ELIG_CSV):
        assert e["no_sar_execution_in_this_milestone"] == "true"
        assert e["no_network_in_this_milestone"] == "true"
        assert e["no_raster_download_in_this_milestone"] == "true"


# 17. nenhum vira positive_strong --------------------------------------------
def test_nenhum_positive_strong():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_promovidos"] == 0
    assert manifest["sar_executado"] == 0
    assert manifest["footprint_criado"] == 0
    assert manifest["sentinel_baixado"] == 0
    for p in _read(S.PACK_CSV):
        assert "positive_strong" not in p["rationale_pt"].lower()


# 18-20. guardrails sempre false ---------------------------------------------
def test_guardrails_false():
    for p in _read(S.PACK_CSV):
        assert p["eligible_for_training"] == "false"
        assert p["eligible_for_ground_truth"] == "false"
        assert p["score_v7_candidate"] == "false"


# 21. relatorio publico principal em portugues -------------------------------
def test_relatorio_portugues():
    h1 = S._report_h1(S.REPORT.read_text(encoding="utf-8"))
    assert S.title_is_portuguese(h1)


# 22. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.PACK_CSV, S.ELIG_CSV, S.REQ_CSV, S.PLAN_CSV, S.EXCL_CSV, S.CANARY_CSV,
            S.BLOCKERS_CSV, S.SUMMARY_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 23-24. schemas aceitam validos e rejeitam invalidos ------------------------
def test_schema_aceita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_TARGET.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    for r in _read(S.PACK_CSV)[:60]:
        assert list(v.iter_errors(r)) == []


def test_schema_rejeita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_TARGET.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    inv = dict(_read(S.PACK_CSV)[0])
    inv["eligible_for_training"] = "true"
    assert list(v.iter_errors(inv))
    inv2 = dict(_read(S.PACK_CSV)[0])
    inv2["sar_canary_priority_score"] = "150"
    assert list(v.iter_errors(inv2))


# 25. GT06 nao modifica outputs GT01-05 --------------------------------------
def test_nao_modifica_marcos_anteriores():
    import hashlib

    def digest():
        h = {}
        for p in (sorted(SUSC.glob("susc_gt0[12345]_*"))
                  + sorted((ROOT / "schemas" / "suscetibilidade").glob("susc_gt0[12345]_*"))):
            h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return h

    before = digest()
    S.build_all()
    assert before == digest()


# 26. manifesto lista sha256 -------------------------------------------------
def test_manifesto_sha256():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    assert len(arts) == 13  # 9 publicos + 4 schemas
    for a in arts:
        assert len(a["sha256"]) == 64


# 27. recomendacao GT07 da distribuicao real ---------------------------------
def test_recomendacao_gt07():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["proximo_marco"].startswith("GT07")
    assert S._next_milestone({}, 5, 10).endswith("QA Humano para Canarios")
    assert S._next_milestone({}, 1, 10).endswith("Cenas Sentinel-1")
    assert S._next_milestone({"requer_geometria_antes_do_sar": 500}, 0, 3).endswith("Geometria/AOI")


# 28. footprint pos-evento e referencia futura, nao feature pre-evento -------
def test_footprint_referencia_futura():
    txt = S.REPORT.read_text(encoding="utf-8").lower()
    assert "nunca feature pré-evento" in txt or "nunca feature pre-evento" in txt
    for p in _read(S.PACK_CSV):
        assert "referencia de avaliacao" in p["expected_future_sar_artifact_pt"].lower()


# 29. canarios sao review-only, nao treino -----------------------------------
def test_canarios_review_only():
    for c in _read(S.CANARY_CSV):
        assert c["remains_review_only"] == "true"
        assert "nao e base de treino" in c["not_training_reason_pt"].lower() or "nao e" in c["not_training_reason_pt"].lower()


# 30. selecao vazia valida (logica) ------------------------------------------
def test_selecao_vazia_valida():
    # se nao houver prioritarios, selecao e vazia e valida
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["canarios_selecionados"] <= min(5, manifest["canarios_prioritarios_elegiveis"])


# extra: validador completo passa --------------------------------------------
def test_validador_completo_passa():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []


def test_recife_selecionado():
    can = _read(S.CANARY_CSV)
    assert can  # ha canarios (Recife Defesa Civil)
    assert all(c["city"] == "recife" for c in can)
    assert all(c["patch_id"] != "not_available" for c in can)
