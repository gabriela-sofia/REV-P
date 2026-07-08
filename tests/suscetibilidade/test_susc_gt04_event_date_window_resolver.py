"""Testes para SUSC-GT04 - Resolvedor de Datas e Janelas Pre/Pos-Evento."""

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


S = _load("gt04_common", "susc_gt04_event_date_window_resolver_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


# 1. Leitura dos outputs GT03 ------------------------------------------------
def test_le_outputs_gt03():
    targets = _read(SUSC / "susc_gt03_pacote_alvos_evidencia_forte.csv")
    dates = _read(S.DATES_CSV)
    assert len(dates) == len(targets)
    assert dates[0]["target_id"].startswith("TGT_")


# 2. event_date explicito -> exact_day ---------------------------------------
def test_event_date_explicito_exact_day():
    r = S.resolve_date({"event_date": "2022-05-24"})
    assert r["temporal_precision"] == "exact_day"
    assert r["resolved_event_date"] == "2022-05-24"
    assert r["temporal_confidence_score"] == 100


# 3. event_date_candidate -> exact_day ---------------------------------------
def test_event_date_candidate_exact_day():
    r = S.resolve_date({"event_date_candidate": "2019-01-15"})
    assert r["temporal_precision"] == "exact_day"
    assert r["resolved_event_date"] == "2019-01-15"


# 4-5. event_id padrao -> inferred_from_event_id + flag inferida -------------
def test_event_id_inferred():
    r = S.resolve_date({"event_id": "REC_2022_05_24_30"})
    assert r["temporal_precision"] == "inferred_from_event_id"
    assert r["resolved_event_date"] == "2022-05-24"
    assert r["date_is_inferred"] == "true"


# 6. publication_date_only nao libera SAR ------------------------------------
def test_publication_date_nao_libera_sar():
    r = S.resolve_date({"publication_date": "2022-05-24"})
    assert r["temporal_precision"] == "publication_date_only"
    w = S.build_window(r)
    assert w["usable_for_sar_candidate_future"] == "false"
    assert w["window_status"] == "usable_for_documentary_context_only"


# 7. month_only bloqueia janela ----------------------------------------------
def test_month_only_bloqueia_janela():
    r = S.resolve_date({"notes": "evento em 2022-05"})
    assert r["temporal_precision"] == "month_only"
    w = S.build_window(r)
    assert w["window_status"] == "blocked_temporal_resolution_insufficient"
    assert w["window_blocking_reason"]


# 8. year_only bloqueia janela -----------------------------------------------
def test_year_only_bloqueia_janela():
    r = S.resolve_date({"notes": "ocorrido em 2022"})
    assert r["temporal_precision"] == "year_only"
    w = S.build_window(r)
    assert w["window_status"] == "blocked_temporal_resolution_insufficient"


# 9. unknown bloqueia janela -------------------------------------------------
def test_unknown_bloqueia_janela():
    r = S.resolve_date({"event_id": "S16ALOCG_00054"})
    assert r["temporal_precision"] == "unknown"
    w = S.build_window(r)
    assert w["window_status"] == "blocked_temporal_resolution_insufficient"
    assert w["usable_for_sar_candidate_future"] == "false"


# 10. date_range gera janela com incerteza -----------------------------------
def test_date_range_janela_incerteza():
    r = S.resolve_date({"date_range_start": "2022-05-01", "date_range_end": "2022-05-10"})
    assert r["temporal_precision"] == "date_range"
    w = S.build_window(r)
    assert w["window_status"] == "usable_with_range_uncertainty"
    assert w["usable_for_sar_candidate_future"] == "true"


# 11. data invalida -> invalid_or_conflicting --------------------------------
def test_data_invalida():
    r = S.resolve_date({"event_date": "2022-13-40"})
    assert r["temporal_precision"] == "invalid_or_conflicting"


# 12. datas conflitantes -> invalid_or_conflicting ---------------------------
def test_datas_conflitantes():
    r = S.resolve_date({"event_date": "2022-05-24 ou 2019-01-01"})
    assert r["temporal_precision"] == "invalid_or_conflicting"
    assert r["date_conflict_detected"] == "true"


# 13. exact_day gera janela pre=-30..-1 e post=0..+7 -------------------------
def test_janela_exact_day():
    r = S.resolve_date({"event_date": "2022-05-24"})
    w = S.build_window(r)
    assert w["pre_event_window_start"] == "2022-04-24"
    assert w["pre_event_window_end"] == "2022-05-23"
    assert w["post_event_window_start"] == "2022-05-24"
    assert w["post_event_window_end"] == "2022-05-31"


# 14. window start nunca maior que end (todas as saidas) ---------------------
def test_window_start_menor_igual_end():
    for w in _read(S.WINDOWS_CSV):
        for a, b in (("pre_event_window_start", "pre_event_window_end"),
                     ("post_event_window_start", "post_event_window_end")):
            if w[a] and w[b]:
                assert w[a] <= w[b]


# 15. temporal_confidence_score sempre 0..100 --------------------------------
def test_confidence_intervalo():
    for d in _read(S.DATES_CSV):
        sc = int(d["temporal_confidence_score"])
        assert 0 <= sc <= 100


# 16-18. guardrails sempre false ---------------------------------------------
def test_eligible_for_training_sempre_false():
    for d in _read(S.DATES_CSV):
        assert d["eligible_for_training"] == "false"


def test_eligible_for_ground_truth_sempre_false():
    for d in _read(S.DATES_CSV):
        assert d["eligible_for_ground_truth"] == "false"


def test_score_v7_candidate_sempre_false():
    for d in _read(S.DATES_CSV):
        assert d["score_v7_candidate"] == "false"
    for r in _read(S.SUMMARY_CSV):
        assert r["score_v7_candidate_count"] == "0"


# 19. nenhum alvo vira positive_strong ---------------------------------------
def test_nenhum_positive_strong():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_promovidos"] == 0
    for d in _read(S.DATES_CSV):
        assert "positive_strong" not in d["rationale_pt"].lower()


# 20. relatorio publico principal em portugues -------------------------------
def test_relatorio_titulo_portugues():
    h1 = S._report_h1(S.REPORT.read_text(encoding="utf-8"))
    assert S.title_is_portuguese(h1)


# 21. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.DATES_CSV, S.WINDOWS_CSV, S.DIAG_CSV, S.BLOCKERS_CSV, S.SUMMARY_CSV,
            S.RELEASED_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 22-23. schemas aceitam validos e rejeitam invalidos ------------------------
def test_schema_aceita_data_valida():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_DATE.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    rows = _read(S.DATES_CSV)
    for r in rows[:60]:
        assert list(v.iter_errors(r)) == []


def test_schema_rejeita_data_invalida():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_DATE.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    inv = dict(_read(S.DATES_CSV)[0])
    inv["eligible_for_training"] = "true"
    assert list(v.iter_errors(inv))
    inv2 = dict(_read(S.DATES_CSV)[0])
    inv2["temporal_confidence_score"] = "150"
    assert list(v.iter_errors(inv2))


# 24. GT04 nao modifica outputs GT01/GT02/GT03 -------------------------------
def test_nao_modifica_marcos_anteriores():
    import hashlib

    def digest():
        h = {}
        for p in (sorted(SUSC.glob("susc_gt0[123]_*"))
                  + sorted((ROOT / "schemas" / "suscetibilidade").glob("susc_gt0[123]_*"))):
            h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return h

    before = digest()
    S.build_all()
    assert before == digest()


# 25. manifesto lista sha256 dos artefatos GT04 ------------------------------
def test_manifesto_sha256():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    assert len(arts) == 11  # 7 publicos + 4 schemas
    for a in arts:
        assert len(a["sha256"]) == 64


# 26. recomendacao GT05 calculada da distribuicao real -----------------------
def test_recomendacao_gt05():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["proximo_marco"].startswith("GT05")
    # muitas datas resolvidas -> geometria oficial
    assert "Geometria Oficial" in manifest["proximo_marco"]
    # logica pura
    assert S._next_milestone({"exact_day": 300}, 557).endswith("Geometria Oficial")
    assert S._next_milestone({"unknown": 500, "exact_day": 5}, 557).endswith("Enriquecimento Documental Manual")


# extra: validador completo passa --------------------------------------------
def test_validador_completo_passa():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []


def test_janela_exact_day_no_csv_real():
    # ao menos um alvo real exact_day com janela utilizavel
    dates = {d["target_id"]: d for d in _read(S.DATES_CSV)}
    windows = _read(S.WINDOWS_CSV)
    usable = [w for w in windows if w["window_status"] == "usable"]
    assert usable
    w = usable[0]
    assert dates[w["target_id"]]["temporal_precision"] in ("exact_day", "inferred_from_event_id")
