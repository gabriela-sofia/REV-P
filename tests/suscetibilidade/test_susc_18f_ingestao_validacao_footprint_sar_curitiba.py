"""Tests for SUSC-18F ingestao SAR Curitiba."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18f_ingestao_validacao_footprint_sar_curitiba"


def _load_common():
    path = SCRIPTS / "susc_18f_sar_ingestao_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18f_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all(attempt_recovery=False)


def test_patch_stats_local_gera_referencia_tecnica():
    rows = _read(OUT / "estatisticas_sar_por_patch_curitiba_18f.csv")
    assert len(rows) == 43
    matrix = _read(OUT / "matriz_referencia_tecnica_sar_curitiba.csv")
    assert matrix
    assert all(r["patch_stats_disponivel"] == "true" for r in matrix)


def test_geojson_valido_gera_footprint_tecnico_quando_existente():
    fp = _read(OUT / "footprint_sar_curitiba_ingestado.csv")[0]
    if S.LOCAL_VECTOR.exists():
        assert fp["footprint_vetorial_disponivel"] == "true"
        assert fp["geometry_id"].startswith("S18F_CUR_SAR_FOOTPRINT")
    else:
        assert fp["footprint_vetorial_disponivel"] == "false"


def test_footprint_valido_gera_vinculo_tecnico():
    links = _read(OUT / "vinculos_sar_patch_curitiba_18f.csv")
    fp = _read(OUT / "footprint_sar_curitiba_ingestado.csv")[0]
    if fp["footprint_vetorial_disponivel"] == "true":
        assert any(l["classe_vinculo_tecnico"].startswith("technical_footprint") for l in links)
    else:
        assert all(l["classe_vinculo_tecnico"] == "sar_patch_stat_available" for l in links)


def test_ausencia_de_geojson_impede_vinculo_geometrico():
    fp = _read(OUT / "footprint_sar_curitiba_ingestado.csv")[0]
    links = _read(OUT / "vinculos_sar_patch_curitiba_18f.csv")
    if fp["footprint_vetorial_disponivel"] == "false":
        assert all(not l["classe_vinculo_tecnico"].startswith("technical_footprint") for l in links)


def test_patch_stats_nao_vira_footprint():
    fp = _read(OUT / "footprint_sar_curitiba_ingestado.csv")[0]
    if fp["footprint_vetorial_disponivel"] == "false":
        assert fp["geometry_id"] == "not_available"
        assert fp["geometry_type"] == "none"


def test_footprint_tecnico_nao_e_geometria_oficial():
    fp = _read(OUT / "footprint_sar_curitiba_ingestado.csv")[0]
    assert fp["geometria_oficial_de_ocorrencia"] == "false"
    assert "nao e geometria oficial" in fp["justificativa_tecnica"]


def test_feature_pos_evento_proibida_como_pre_evento():
    features = _read(OUT / "features_por_vinculo_sar_curitiba_18f.csv")
    assert features
    assert all(f["periodo_evento"] == "pos_evento" for f in features)
    assert all(f["usavel_pre_evento"] == "false" for f in features)


def test_score_v6_nao_alterado_e_score_v7_nao_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


def test_17b_nao_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_17b_criado"] is False
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*"))


def test_vocabulario_publico_proibido_falha():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo}")
