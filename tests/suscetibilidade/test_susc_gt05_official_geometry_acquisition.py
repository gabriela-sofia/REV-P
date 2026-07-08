"""Testes para SUSC-GT05 - Pacote de Aquisicao de Geometria Oficial."""

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


S = _load("gt05_common", "susc_gt05_official_geometry_acquisition_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def _fields(**over):
    base = {k: "" for k in ("geometry_type", "geometry_id", "footprint_id", "patch_id",
                            "patch_link_quality", "uncertainty_m", "qa_status",
                            "source_authority", "phenomenon_type", "city", "region",
                            "temporal_precision")}
    base.update(over)
    return base


def _status(**over):
    return S.classify_status(S.analyze_geometry(_fields(**over)))


# 1. Leitura dos outputs GT04 ------------------------------------------------
def test_le_outputs_gt04():
    dates = _read(SUSC / "susc_gt04_datas_resolvidas_por_alvo.csv")
    tg = _read(S.TARGETS_CSV)
    assert len(tg) == len(dates)
    assert tg[0]["geometry_target_id"].startswith("GEO_")


# 2. geometry_type forte -> geometria_forte_existente ------------------------
def test_forte_existente():
    st = _status(geometry_type="official_observed_event_polygon", patch_id="P1",
                 source_authority="defesa_civil", phenomenon_type="flood")
    assert st == "geometria_forte_existente"


# 3. ponto/parcial -> geometria_parcial_resolvivel ---------------------------
def test_parcial_resolvivel():
    assert _status(geometry_type="point", source_authority="defesa_civil") == "geometria_parcial_resolvivel"


# 4. city_level_record -> contextual_insuficiente ----------------------------
def test_contextual_insuficiente():
    assert _status(geometry_type="city_level_record") == "geometria_contextual_insuficiente"


# 5-7. tipos fracos nunca geometria forte ------------------------------------
@pytest.mark.parametrize("weak", ["risk_area_not_event", "alert_only", "news_only"])
def test_fraco_nunca_forte(weak):
    st = _status(source_authority=weak, patch_id="P1")
    assert st != "geometria_forte_existente"


# 8. sem geometry_type -> ausente ou requer_footprint ------------------------
def test_sem_geometria():
    assert _status() == "geometria_ausente"
    assert _status(temporal_precision="exact_day") == "requer_footprint_tecnico_futuro"


# 9. datado sem geometria -> preparar_footprint_sar_futuro sem SAR -----------
def test_datado_prepara_sar_sem_execucao():
    fl = S.analyze_geometry(_fields(temporal_precision="exact_day", source_authority="defesa_civil"))
    tracks = S.geometry_tracks(fl, S.classify_status(fl))
    assert "preparar_footprint_sar_futuro" in tracks
    for tr in _read(S.TRACKS_CSV):
        assert tr["no_sar_execution_in_this_milestone"] == "true"
        assert tr["no_geometry_created_in_this_milestone"] == "true"


# 10. temporal unknown nao liberado para SAR ---------------------------------
def test_unknown_nao_libera_sar():
    for t in _read(S.TARGETS_CSV):
        if t["temporal_precision"] in ("unknown", "invalid_or_conflicting", ""):
            assert t["can_advance_to_sar_future"] == "false"


# 11. Petropolis mixed exige separar_fenomeno --------------------------------
def test_petropolis_mixed_separar():
    fields = _fields(city="Petropolis", region="petropolis",
                     phenomenon_type="mixed_flood_landslide", geometry_type="point")
    fl = S.analyze_geometry(fields)
    st = S.classify_status(fl)
    tracks = S.geometry_tracks(fl, st)
    assert st == "geometria_rejeitada"
    assert "separar_fenomeno" in tracks
    score = S.geometry_priority_score(fl, "unknown", "prioridade_media", S.missing_geometry(fl))
    assert S.geometry_priority_class(fl, st, score) != "prioridade_geometrica_alta"


# 12. geometry_priority_score sempre 0..100 ----------------------------------
def test_score_intervalo():
    for t in _read(S.TARGETS_CSV):
        assert 0 <= int(t["geometry_priority_score"]) <= 100


# 13. geometry_priority_class na lista ---------------------------------------
def test_classe_valida():
    for t in _read(S.TARGETS_CSV):
        assert t["geometry_priority_class"] in S.GEOMETRY_PRIORITY_CLASSES


# 14. geometry_acquisition_status na lista -----------------------------------
def test_status_valido():
    for t in _read(S.TARGETS_CSV):
        assert t["geometry_acquisition_status"] in S.GEOMETRY_STATUSES


# 15. nenhum alvo vira positive_strong ---------------------------------------
def test_nenhum_positive_strong():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_promovidos"] == 0
    assert manifest["geometria_criada"] == 0
    for t in _read(S.TARGETS_CSV):
        assert "positive_strong" not in t["rationale_pt"].lower()
        if t["can_become_positive_strong_after_geometry_and_qa"] == "true":
            assert t["missing_geometry_requirements"] not in ("", "none")


# 16-18. guardrails sempre false ---------------------------------------------
def test_guardrails_false():
    for t in _read(S.TARGETS_CSV):
        assert t["eligible_for_training"] == "false"
        assert t["eligible_for_ground_truth"] == "false"
        assert t["score_v7_candidate"] == "false"
    for r in _read(S.SUMMARY_CSV):
        assert r["score_v7_candidate_count"] == "0"


# 19. relatorio publico principal em portugues -------------------------------
def test_relatorio_portugues():
    h1 = S._report_h1(S.REPORT.read_text(encoding="utf-8"))
    assert S.title_is_portuguese(h1)


# 20. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.TARGETS_CSV, S.DIAG_CSV, S.TRACKS_CSV, S.PRIORITY_CSV, S.PLAN_CSV,
            S.BLOCKERS_CSV, S.SUMMARY_CSV, S.RELEASED_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 21-22. schemas aceitam validos e rejeitam invalidos ------------------------
def test_schema_aceita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_TARGET.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    for r in _read(S.TARGETS_CSV)[:60]:
        assert list(v.iter_errors(r)) == []


def test_schema_rejeita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_TARGET.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    inv = dict(_read(S.TARGETS_CSV)[0])
    inv["eligible_for_training"] = "true"
    assert list(v.iter_errors(inv))
    inv2 = dict(_read(S.TARGETS_CSV)[0])
    inv2["geometry_priority_score"] = "200"
    assert list(v.iter_errors(inv2))


# 23. GT05 nao modifica outputs GT01/GT02/GT03/GT04 --------------------------
def test_nao_modifica_marcos_anteriores():
    import hashlib

    def digest():
        h = {}
        for p in (sorted(SUSC.glob("susc_gt0[1234]_*"))
                  + sorted((ROOT / "schemas" / "suscetibilidade").glob("susc_gt0[1234]_*"))):
            h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return h

    before = digest()
    S.build_all()
    assert before == digest()


# 24. manifesto lista sha256 -------------------------------------------------
def test_manifesto_sha256():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    assert len(arts) == 13  # 9 publicos + 4 schemas
    for a in arts:
        assert len(a["sha256"]) == 64


# 25. recomendacao GT06 da distribuicao real ---------------------------------
def test_recomendacao_gt06():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["proximo_marco"].startswith("GT06")
    assert S._next_milestone({"geometria_parcial_resolvivel": 400}, 557).endswith("QA Humano")
    assert S._next_milestone({"requer_footprint_tecnico_futuro": 400}, 557).endswith("SAR sem Execucao")
    assert S._next_milestone({"geometria_ausente": 500}, 557).endswith("Documental Manual Orientada")


# 26. tipos contextuais proibidos nunca geometria_forte_existente ------------
def test_contextual_nunca_forte():
    for t in _read(S.TARGETS_CSV):
        gtype = t["geometry_type"].lower()
        src = t["source_authority"].lower()
        if gtype in S.CONTEXTUAL_FORBIDDEN_TYPES or src in S.CONTEXTUAL_FORBIDDEN_TYPES:
            assert t["geometry_acquisition_status"] != "geometria_forte_existente"


# extra: validador completo passa --------------------------------------------
def test_validador_completo_passa():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []


def test_recife_defesa_civil_vira_alta():
    # ocorrencias Recife datadas + point + patch_link + fonte oficial -> alta
    altas = [t for t in _read(S.TARGETS_CSV) if t["geometry_priority_class"] == "prioridade_geometrica_alta"]
    assert altas
    assert all(t["can_advance_to_qa_future"] == "true" for t in altas)
