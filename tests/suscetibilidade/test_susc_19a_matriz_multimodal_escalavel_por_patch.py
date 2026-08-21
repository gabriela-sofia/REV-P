"""Testes para SUSC-19A matriz multimodal escalavel por patch."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_19a_matriz_multimodal_escalavel_por_patch"


def _load_common():
    path = SCRIPTS / "susc_19a_matriz_multimodal_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s19a_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def test_matriz_final_tem_patches():
    final = _read(S.MAT_FINAL)
    assert len(final) > 0


def test_patch_id_unico():
    final = _read(S.MAT_FINAL)
    ids = [r["patch_id"] for r in final]
    assert len(ids) == len(set(ids))


def test_todas_saidas_obrigatorias_existem():
    for path in S.REQUIRED_OUTPUTS:
        assert path.exists(), f"faltando {path}"


def test_recife_aparece():
    final = _read(S.MAT_FINAL)
    assert any(r["regiao"] == "recife" for r in final)


def test_curitiba_aparece():
    final = _read(S.MAT_FINAL)
    assert any(r["regiao"] == "curitiba" for r in final)


def test_petropolis_aparece_se_houver_registry():
    final = _read(S.MAT_FINAL)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if "petropolis" in summary["patches_por_regiao"]:
        assert any(r["regiao"] == "petropolis" for r in final)


def test_ground_truth_trainable_score_v7_proibidos():
    for r in _read(S.MAT_FINAL):
        assert r["ground_truth"] == "false"
        assert r["eligible_for_training"] == "false"
        assert r["score_v7_allowed"] == "false"


def test_score_v6_intacto():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


def test_patch_stats_sar_nao_vira_feature_pre_evento():
    doc = _read(S.MAT_DOC)
    for r in doc:
        assert r["technical_sar_patch_stats"] in ("not_available", "disponivel_pos_evento_nao_feature")
    for r in _read(S.MAT_ESPECTRAL):
        assert r["pre_event_flag"] != "true"


def test_coverage_score_nao_e_score_de_suscetibilidade():
    for r in _read(S.INDICE):
        assert r["score_oficial"] == "false"
        assert r["substituir_score_v6"] == "false"
        assert r["usar_em_treino"] == "false"
        assert "nao" in r["uso_permitido"].lower() and "suscetibilidade" in r["uso_permitido"].lower()
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage_is_susceptibility_score"] is False


def test_missingness_calculada():
    miss = _read(S.MISSINGNESS)
    assert len(miss) > 0
    # territorial e a maior lacuna: todos os patches devem ter missing_territorial true
    assert all(r["missing_territorial"] == "true" for r in miss)
    assert any(r["features_faltantes"] != "nenhuma" for r in miss)


def test_curitiba_sar_nao_e_geometria_oficial():
    doc = _read(S.MAT_DOC)
    sar = [r for r in doc if r["curitiba_sar_evidence"] == "true"]
    assert sar
    for r in sar:
        assert "nao" in r["justificativa_tecnica"].lower()


def test_outputs_19a_versionaveis():
    ignored = S.git_ignored([OUT / "summary.json", S.MAT_FINAL])
    assert S.rel(OUT / "summary.json") not in ignored
    assert S.rel(S.MAT_FINAL) not in ignored


def test_vocabulario_publico_proibido_falha():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo}")


def test_saidas_publicas_sem_vocabulario_proibido():
    assert S.validate_public_text() == []


def test_validator_completo_passa():
    summary = S.build_all()
    assert S.validate_outputs(summary) == []
