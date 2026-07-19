"""Testes para SUSC-19C avaliacao observacional review-only."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_19c_avaliacao_observacional_review_only"


def _load_common():
    path = SCRIPTS / "susc_19c_avaliacao_observacional_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s19c_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def test_registro_observacional_contem_7_patches():
    obs = _read(S.REG_OBS)
    assert len(obs) == 7


def test_recife_tem_5_observacionais():
    obs = _read(S.REG_OBS)
    assert sum(1 for r in obs if r["regiao"] == "recife") == 5


def test_curitiba_tem_2_observacionais():
    obs = _read(S.REG_OBS)
    cur = [r for r in obs if r["regiao"] == "curitiba"]
    assert len(cur) == 2
    assert {r["patch_id"] for r in cur} == {"curitiba_01050", "curitiba_01101"}


def test_petropolis_nao_entra_como_observado():
    obs = _read(S.REG_OBS)
    assert not any(r["regiao"] == "petropolis" for r in obs)
    exc = _read(S.REG_EXC)
    assert all(r["status"] == "contextual_bloqueado" for r in exc)


def test_universo_e_background_nao_negativo():
    uni = _read(S.UNIVERSO)
    assert uni
    assert all(r["rotulo"] == "unlabeled_background" for r in uni)
    for r in uni:
        blob = " ".join(str(v) for v in r.values()).lower()
        if "negativ" in blob:
            assert "nao e negativo" in blob


def test_hit_rate_e_review_only():
    for r in _read(S.HITRATE):
        lim = r["limitation"].lower()
        assert "review-only" in lim or "exploratori" in lim
        assert not ("benchmark" in lim and "nao e" not in lim)


def test_score_v6_intacto():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


def test_score_v7_bloqueado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["status_score_v7"] in S.STATUS_V7_ALLOWED
    assert summary["status_score_v7"] != "SCORE_V7_AUTORIZADO"
    for r in _read(S.GATE_V7):
        assert r["status_score_v7"] in S.STATUS_V7_ALLOWED


def test_17b_nao_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["marco_17b_criado"] is False
    assert summary["benchmark_17b_criado"] is False
    assert summary["status_17b"] in S.STATUS_17B_ALLOWED


def test_ground_truth_trainable_proibidos():
    for r in _read(S.MAT_AVAL):
        assert r["ground_truth"] == "false"
        assert r["eligible_for_training"] == "false"
        assert r["score_v7_allowed"] == "false"


def test_patch_stats_nao_e_feature_pre_evento():
    for path in (S.CONTRASTE, S.MAT_AVAL):
        for r in _read(path):
            blob = " ".join(str(v).lower() for v in r.values())
            assert "patch_stats" not in blob


def test_output_19c_versionavel():
    ignored = S.git_ignored([OUT / "summary.json", S.MAT_AVAL])
    assert S.rel(OUT / "summary.json") not in ignored


def test_vocabulario_publico_proibido_falha():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo}")


def test_saidas_publicas_sem_vocabulario_proibido():
    assert S.validate_public_text() == []


def test_validator_completo_passa():
    summary = S.build_all()
    assert S.validate_outputs(summary) == []
