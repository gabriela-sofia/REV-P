"""Testes para SUSC-GT02 - Motor de Decisao de Referencia Observacional e Replay."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"


def _load(name, filename):
    path = SCRIPTS / filename
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load("gt02_common", "susc_gt02_ground_reference_replay_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


# --- registro sintetico completo (positive_strong) --------------------------
STRONG = {
    "event_id": "EVT_1", "event_date": "2022-05-24", "source_authority": "defesa_civil_oficial",
    "geometry_type": "official_observed_event_polygon", "patch_id": "P1",
    "patch_link_quality": "high_spatial_overlap", "uncertainty_m": "30",
    "qa_status": "accepted", "phenomenon_type": "flood",
}


def _rec(**over):
    base = dict.fromkeys(S.CONCEPT_SYNONYMS, "")
    base.update(STRONG)
    base.update(over)
    return base


# 1. Descoberta de input parseavel ------------------------------------------
def test_descoberta_input_parseavel():
    inv = _read(S.INVENTORY_CSV)
    assert inv
    usaveis = [r for r in inv if r["usable_for_replay"] == "true"]
    assert usaveis
    paths = {r["path"].split("/")[-1] for r in usaveis}
    assert "susc_17a_reference_evidence_registry.csv" in paths


# 2. Input sem colunas minimas vira bloqueado --------------------------------
def test_input_sem_colunas_minimas_bloqueado():
    inv = _read(S.INVENTORY_CSV)
    # tabela de feature/score generica nao pode ser descoberta como evidencia
    paths = {r["path"].split("/")[-1] for r in inv}
    assert "susc_18a_feature_lineage_manifest.csv" not in paths
    # entradas descobertas sem linhas de dados aparecem como nao-usaveis
    bloqueados = [r for r in inv if r["usable_for_replay"] == "false"]
    assert any(r["blocking_reason"] for r in bloqueados)


# 3. Registro completo vira positive_strong ----------------------------------
def test_registro_completo_positive_strong():
    d = S.decide_state(_rec())
    assert d["assigned_state"] == "positive_strong"
    assert d["missing_requirements"] == "none"
    assert d["eligible_for_evaluation"] == "true"


# 4. Registro sem geometria nao vira positive_strong -------------------------
def test_sem_geometria_nunca_strong():
    d = S.decide_state(_rec(geometry_type="point", evidence_class=""))
    assert d["assigned_state"] in ("positive_provisional", "no_data")
    assert d["assigned_state"] != "positive_strong"


# 5-8. Fontes fracas nunca viram positive_strong -----------------------------
@pytest.mark.parametrize("kind", [
    "alert_only", "risk_area_not_event", "news_only", "city_level_record",
])
def test_fonte_fraca_nunca_strong(kind):
    d = S.decide_state(_rec(evidence_source_kind=kind))
    assert d["assigned_state"] != "positive_strong"


# 9. unlabeled nunca e negative ----------------------------------------------
def test_unlabeled_nunca_negative():
    d = S.decide_state({c: "" for c in S.CONCEPT_SYNONYMS})
    assert d["assigned_state"] == "unlabeled"
    for r in _read(S.REPLAY_CSV):
        if r["assigned_state"] == "unlabeled":
            blob = " ".join(str(v).lower() for v in r.values())
            assert "negativ" not in blob


# 10. no_data exige blocking_reason ------------------------------------------
def test_no_data_exige_blocking_reason():
    d = S.decide_state(_rec(qa_status="", geometry_type="", evidence_class="",
                            source_authority="", phenomenon_type="", event_date="",
                            patch_link_quality="", uncertainty_m=""))
    assert d["assigned_state"] == "no_data"
    assert any(b["blocker_type"].startswith("no_data:") and b["reason"] for b in d["blockers"])
    # explicito
    d2 = S.decide_state(_rec(blocking_reason="cloud"))
    assert d2["assigned_state"] == "no_data"


# 11. rejected exige motivo --------------------------------------------------
def test_rejected_exige_motivo():
    d = S.decide_state(_rec(rejection_reason="phenomenon_mismatch"))
    assert d["assigned_state"] == "rejected"
    assert any(b["blocker_type"] == "rejected" and b["reason"] for b in d["blockers"])


# 12. hard_negative_audited exige QA -----------------------------------------
def test_hard_negative_exige_qa():
    audit = dict(observability="true", exclusion_check="true", in_same_event="true",
                 outside_footprint="true", permanent_water="false")
    sem_qa = S.decide_state(_rec(qa_status="pending", **audit))
    assert sem_qa["assigned_state"] != "hard_negative_audited"
    com_qa = S.decide_state({c: "" for c in S.CONCEPT_SYNONYMS} | dict(
        qa_status="accepted", **audit))
    assert com_qa["assigned_state"] == "hard_negative_audited"


# 13. Petropolis mixed bloqueia calibracao -----------------------------------
def test_petropolis_mixed_bloqueia_calibracao():
    d = S.decide_state(_rec(region="petropolis", phenomenon_type="mixed_flood_landslide"))
    assert d["assigned_state"] != "positive_strong"
    assert d["eligible_for_calibration"] == "false"
    assert any(b["blocker_type"] == "phenomenon_mismatch_or_mixed_event" for b in d["blockers"])


# 14-16. guardrails sempre false ---------------------------------------------
def test_eligible_for_training_sempre_false():
    for path in (S.REPLAY_CSV, S.DECISION_CSV):
        for r in _read(path):
            assert r["eligible_for_training"] == "false"


def test_eligible_for_ground_truth_sempre_false():
    for path in (S.REPLAY_CSV, S.DECISION_CSV):
        for r in _read(path):
            assert r["eligible_for_ground_truth"] == "false"


def test_score_v7_candidate_sempre_false():
    for r in _read(S.DECISION_CSV):
        assert r["score_v7_candidate"] == "false"
    for r in _read(S.STATE_SUMMARY_CSV):
        assert r["score_v7_candidate_count"] == "0"


# 17. GT01 e chamado / respeitado --------------------------------------------
def test_gt01_respeitado():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["gt01_valida"] is True
    import susc_gt01_ground_reference_common as gt01
    assert gt01.validate_outputs(gt01.build_all()) == []


# 18-19. schemas aceitam validos e rejeitam invalidos ------------------------
def test_schema_aceita_decisao_valida():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_DECISION.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    rows = _read(S.DECISION_CSV)
    assert rows
    for r in rows[:50]:
        assert list(v.iter_errors(r)) == []


def test_schema_rejeita_decisao_invalida():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_DECISION.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    invalida = dict(_read(S.DECISION_CSV)[0])
    invalida["eligible_for_training"] = "true"  # viola const false
    assert list(v.iter_errors(invalida))


# 20. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.INVENTORY_CSV, S.COLMAP_CSV, S.REPLAY_CSV, S.DECISION_CSV,
            S.BLOCKERS_CSV, S.STATE_SUMMARY_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 21. relatorio publico principal em portugues -------------------------------
def test_relatorio_titulo_portugues():
    txt = S.REPORT.read_text(encoding="utf-8")
    h1 = S._report_h1(txt)
    assert S.title_is_portuguese(h1)
    assert S.TITULO_H1_TOKEN in h1


# extra: validador completo passa e sem previsao operacional ------------------
def test_validador_completo_passa():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []


def test_sem_claim_previsao_operacional():
    txt = S.REPORT.read_text(encoding="utf-8")
    assert S.operational_forecast_violations(txt) == []


def test_positive_strong_ausente_no_replay_real():
    # dados reais nao tem QA aceito -> fail-closed: nenhum positive_strong
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_count"] == 0
