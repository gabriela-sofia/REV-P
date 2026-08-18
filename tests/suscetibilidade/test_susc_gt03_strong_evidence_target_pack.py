"""Testes para SUSC-GT03 - Pacote de Alvos para Aquisicao de Evidencia Forte."""

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


S = _load("gt03_common", "susc_gt03_strong_evidence_target_pack_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


# campos base de um alvo quase completo (falta so qa_status)
NEAR_STRONG = {
    "event_id": "EVT_1", "event_date": "2022-05-24", "source_authority": "defesa_civil_oficial",
    "geometry_type": "official_observed_event_polygon", "patch_id": "P1", "footprint_id": "F1",
    "geometry_id": "F1", "patch_link_quality": "high_spatial_overlap", "uncertainty_m": "30",
    "qa_status": "", "phenomenon_type": "flood", "city": "Recife", "region": "recife",
}


def _fields(**over):
    base = {k: "" for k in ("event_id", "event_date", "source_authority", "geometry_type",
                            "patch_id", "footprint_id", "geometry_id", "patch_link_quality",
                            "uncertainty_m", "qa_status", "phenomenon_type", "city", "region")}
    base.update(over)
    return base


def _classify(fields, assigned="positive_provisional"):
    flags = S.analyze(fields)
    missing = S.compute_missing(flags)
    tracks = S.acquisition_tracks(flags)
    score = S.priority_score(flags, assigned, missing)
    cls = S.priority_class(flags, score, missing)
    return flags, missing, tracks, score, cls


# 1. Leitura dos outputs GT02 -------------------------------------------------
def test_le_outputs_gt02():
    replay = _read(SUSC / "susc_gt02_registro_replay_evidencias.csv")
    targets = _read(S.TARGETS_CSV)
    assert len(targets) == len(replay)  # um alvo por registro de replay
    assert targets[0]["source_replay_id"].startswith("RP_")


# 2. positive_provisional vira alvo de aquisicao -----------------------------
def test_provisional_vira_alvo():
    provs = [t for t in _read(S.TARGETS_CSV) if t["assigned_state_gt02"] == "positive_provisional"]
    assert provs
    for t in provs:
        assert t["acquisition_tracks"] != "nenhuma"
        assert t["priority_class"] in S.PRIORITY_CLASSES


# 3. no_data com sinal vira bloqueado/baixa ----------------------------------
def test_no_data_sinal_bloqueado_ou_baixo():
    _, _, _, _, cls = _classify(_fields(event_id="EVT_9"), assigned="no_data")
    assert cls in ("bloqueado_sem_acao_imediata", "prioridade_baixa", "descartar_como_contexto")
    assert cls != "prioridade_alta"


# 4. sem event_date -> requisito event_date ----------------------------------
def test_sem_event_date_requisito():
    _, missing, _, _, _ = _classify(_fields(event_id="E", source_authority="oficial",
                                            phenomenon_type="flood", geometry_type="point"))
    assert "event_date" in missing


# 5. sem geometry_type forte -> requisito geometry_type ----------------------
def test_sem_geometria_requisito():
    _, missing, _, _, _ = _classify(_fields(event_id="E", event_date="2022-01-01",
                                            source_authority="oficial", phenomenon_type="flood",
                                            geometry_type="point"))
    assert "geometry_type" in missing


# 6. sem qa aceito -> trilha executar_qa_humano ------------------------------
def test_sem_qa_trilha_qa():
    _, _, tracks, _, _ = _classify(_fields(**{**NEAR_STRONG, "qa_status": "pending"}))
    assert "executar_qa_humano" in tracks


# 7. sem uncertainty_m -> trilha estimar_incerteza_espacial ------------------
def test_sem_incerteza_trilha():
    _, _, tracks, _, _ = _classify(_fields(**{**NEAR_STRONG, "uncertainty_m": ""}))
    assert "estimar_incerteza_espacial" in tracks


# 8. Petropolis mixed exige separar_fenomeno ---------------------------------
def test_petropolis_mixed_separar_fenomeno():
    fields = _fields(**{**NEAR_STRONG, "region": "petropolis", "city": "Petropolis",
                        "phenomenon_type": "mixed_flood_landslide"})
    flags, missing, tracks, score, cls = _classify(fields)
    assert "separar_fenomeno" in tracks
    assert cls != "prioridade_alta"


# 9-10. fontes fracas nunca prioridade_alta sem blocker ----------------------
@pytest.mark.parametrize("kind", ["alert_only", "risk_area_not_event", "news_only", "city_level_record"])
def test_fonte_fraca_nunca_alta(kind):
    fields = _fields(**{**NEAR_STRONG, "source_authority": kind})
    flags, missing, tracks, score, cls = _classify(fields)
    assert cls == "descartar_como_contexto"
    assert cls != "prioridade_alta"
    assert "descartar_ou_rebaixar" in tracks


# 11. priority_score sempre 0..100 -------------------------------------------
def test_priority_score_intervalo():
    for t in _read(S.TARGETS_CSV):
        sc = int(t["priority_score"])
        assert 0 <= sc <= 100


# 12. priority_class na lista permitida --------------------------------------
def test_priority_class_valida():
    for t in _read(S.TARGETS_CSV):
        assert t["priority_class"] in S.PRIORITY_CLASSES


# 13. nenhum alvo vira positive_strong ---------------------------------------
def test_nenhum_alvo_positive_strong():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_promovidos"] == 0
    for t in _read(S.TARGETS_CSV):
        assert t["assigned_state_gt02"] != "positive_strong"
        # can_become e possibilidade futura, exige requisitos faltantes
        if t["can_become_positive_strong_after_acquisition"] == "true":
            assert t["missing_requirements"] not in ("", "none")


# 14-16. guardrails sempre false ---------------------------------------------
def test_eligible_for_training_sempre_false():
    for t in _read(S.TARGETS_CSV):
        assert t["eligible_for_training"] == "false"


def test_eligible_for_ground_truth_sempre_false():
    for t in _read(S.TARGETS_CSV):
        assert t["eligible_for_ground_truth"] == "false"


def test_score_v7_candidate_sempre_false():
    for t in _read(S.TARGETS_CSV):
        assert t["score_v7_candidate"] == "false"
    for r in _read(S.PRIORITY_SUMMARY_CSV):
        assert r["score_v7_candidate_count"] == "0"


# 17. relatorio publico principal em portugues -------------------------------
def test_relatorio_titulo_portugues():
    h1 = S._report_h1(S.REPORT.read_text(encoding="utf-8"))
    assert S.title_is_portuguese(h1)
    assert S.TITULO_H1_TOKEN in h1


# 18. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.TARGETS_CSV, S.MISSING_CSV, S.TRACKS_CSV, S.SOURCES_CSV, S.PLAN_CSV,
            S.PRIORITY_SUMMARY_CSV, S.BLOCKERS_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 19-20. schemas aceitam validos e rejeitam invalidos ------------------------
def test_schema_aceita_alvo_valido():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_TARGET.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    rows = _read(S.TARGETS_CSV)
    assert rows
    for r in rows[:60]:
        assert list(v.iter_errors(r)) == []


def test_schema_rejeita_alvo_invalido():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_TARGET.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    invalido = dict(_read(S.TARGETS_CSV)[0])
    invalido["eligible_for_training"] = "true"  # viola const false
    assert list(v.iter_errors(invalido))
    invalido2 = dict(_read(S.TARGETS_CSV)[0])
    invalido2["priority_score"] = "150"  # fora de 0..100
    assert list(v.iter_errors(invalido2))


# 21. GT03 nao modifica outputs GT01/GT02 ------------------------------------
def test_nao_modifica_gt01_gt02():
    import hashlib

    def digest():
        h = {}
        for p in sorted(SUSC.glob("susc_gt0[12]_*")) + sorted((ROOT / "schemas" / "suscetibilidade").glob("susc_gt0[12]_*")):
            h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return h

    before = digest()
    S.build_all()
    after = digest()
    assert before == after


# 22. manifesto lista sha256 dos artefatos GT03 ------------------------------
def test_manifesto_lista_sha256():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    assert len(arts) == 12  # 8 publicos + 4 schemas
    for a in arts:
        assert len(a["sha256"]) == 64
        assert a["path"].startswith(("outputs_public/", "schemas/"))


# extra: validador completo passa --------------------------------------------
def test_validador_completo_passa():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []


def test_alvo_quase_completo_vira_alta():
    # registro sintetico faltando so qa_status -> prioridade_alta
    flags, missing, tracks, score, cls = _classify(_fields(**NEAR_STRONG))
    assert missing == ["qa_status"]
    assert cls == "prioridade_alta"
    assert S.can_become_strong(flags, missing) is True
