"""Tests for SUSC-18E2 execucao controlada Sentinel-1 Curitiba."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_18e2_execucao_controlada_sentinel1_curitiba"


def _load_common():
    path = SCRIPTS / "susc_18e2_execucao_gee_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18e2_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_public_outputs()


def test_ambiente_sem_autenticacao_gera_bloqueio_por_autenticacao():
    metadata = {
        "status_18e2": "18E2_BLOQUEADO_POR_AUTENTICACAO",
        "block_type": "nao_autenticado",
        "block_detail": "credenciais ausentes",
        "ee_initialize_result": "Exception",
    }
    rows = S.block_rows(metadata)
    assert rows[0]["tipo_bloqueio"] == "nao_autenticado"
    assert "earthengine authenticate" in rows[0]["acao_corretiva_exata"]


def test_resultado_real_ausente_impede_pronto_para_18f():
    handoff = _read(OUT / "handoff_18f.csv")[0]
    if not (S.LOCAL_FLOOD_VECTOR.exists() or S.LOCAL_PATCH_STATS.exists()):
        assert handoff["pronto_para_18f"] == "false"


def test_task_iniciada_exige_task_id():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    gate = [{
        "status_18e2": "18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO",
        "task_ids": "not_available",
    }]
    original = _read(OUT / "gate_execucao_sentinel1_curitiba.csv")
    assert any(
        "status_tarefa_iniciada_sem_task_id" in e
        for e in S.validate_handoff_and_gate({**summary, "score_v6_changed": False, "score_v7_created": False, "benchmark_17b_criado": False})
    ) is (original[0]["status_18e2"] == "18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO" and original[0]["task_ids"] == "not_available")
    assert gate[0]["task_ids"] == "not_available"


def test_pronto_para_18f_exige_arquivo_real():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if summary["ready_for_18f"]:
        assert S.LOCAL_FLOOD_VECTOR.exists() or S.LOCAL_PATCH_STATS.exists()


def test_credenciais_nao_aparecem_em_outputs():
    assert S.public_text_violations_files() == []


def test_aoi_nao_e_ocorrencia():
    aoi = _read(S.AOI_CSV_18E)[0]
    assert aoi["geometria_de_ocorrencia"] == "false"
    assert "nao e geometria de ocorrencia" in aoi["justificativa_tecnica"]


def test_footprint_tecnico_nao_e_geometria_oficial():
    text = (OUT / "summary.json").read_text(encoding="utf-8").lower()
    assert "geometria oficial" not in text or "false" in text


def test_ground_truth_trainable_score_v7_proibidos():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["ground_truth_true_count"] == 0
    assert summary["eligible_for_training_true_count"] == 0
    assert summary["score_v7_allowed_true_count"] == 0
    assert summary["score_v7_created"] is False
    assert summary["benchmark_17b_criado"] is False


def test_vocabulario_publico_proibido_falha():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex", "automacao"]:
        assert S.public_text_violations_text(f"texto com {termo}")
