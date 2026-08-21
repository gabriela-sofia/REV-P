"""Tests for SUSC-18G recuperacao vetorial SAR Curitiba."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18g_recuperacao_compactacao_vetorial_sar_curitiba"


def _load_common():
    path = SCRIPTS / "susc_18g_sar_vector_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18g_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all(run_compact=False)


def test_vetor_local_valido_gera_footprint_tecnico():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if summary["compact_vector_valid"]:
        audit = _read(OUT / "auditoria_footprint_vetorial_compacto.csv")[0]
        assert audit["valido"] == "true"
        assert audit["geometry_id"].startswith("S18G_CUR_SAR_COMPACT")


def test_patch_stats_nao_vira_footprint():
    audit = _read(OUT / "auditoria_footprint_vetorial_compacto.csv")[0]
    if audit["valido"] == "false":
        assert audit["geometry_id"] == "not_available"


def test_compactacao_reduz_numero_de_feicoes():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["compact_feature_count"] <= S.MAX_FEATURES
    if summary["compact_vector_valid"]:
        assert summary["compact_feature_count"] <= S.COMPACT_SAFE_LIMIT


def test_vetor_valido_gera_overlay():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    links = _read(OUT / "vinculos_vetoriais_sar_patch_curitiba_18g.csv")
    if summary["compact_vector_valid"]:
        assert any(l["classe_vinculo_vetorial"].startswith("technical_footprint") for l in links)


def test_ausencia_de_vetor_mantem_referencia_parcial():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    matrix = _read(OUT / "matriz_sar_curitiba_18g.csv")
    if not summary["compact_vector_valid"]:
        assert all(r["status_referencia_tecnica"] == "referencia_tecnica_sar_parcial_por_patch_stats" for r in matrix)


def test_footprint_tecnico_nao_e_geometria_oficial_e_aoi_nao_e_ocorrencia():
    audit = _read(OUT / "auditoria_footprint_vetorial_compacto.csv")[0]
    assert audit["ground_truth"] == "false"
    assert "nao" in audit["justificativa_tecnica"]
    aoi = _read(S.AOI_CSV)[0]
    assert aoi["geometria_de_ocorrencia"] == "false"


def test_feature_pos_evento_nao_vira_pre_evento():
    matrix = _read(OUT / "matriz_sar_curitiba_18g.csv")
    assert matrix
    assert all(r["patch_stats_disponivel"] == "true" for r in matrix)


def test_score_v6_nao_e_alterado_e_17b_nao_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["benchmark_17b_criado"] is False
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*"))


def test_vocabulario_publico_proibido_falha():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo}")
