"""Testes para SUSC-18H consolidacao mestre da cadeia observacional."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_18h_consolidacao_mestre_cadeia_observacional"


def _load_common():
    path = SCRIPTS / "susc_18h_consolidacao_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18h_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def test_build_gera_todas_as_saidas_obrigatorias():
    for path in S.REQUIRED_OUTPUTS:
        assert path.exists(), f"faltando {path}"


def test_inventario_contem_marcos_17c_ate_18g():
    inv = _read(S.INVENTARIO)
    marcos = {r["marco"] for r in inv}
    for esperado in ["SUSC-17C", "SUSC-17C5", "SUSC-17D", "SUSC-17E", "SUSC-17F",
                     "SUSC-17G", "SUSC-17H", "SUSC-17I", "SUSC-18A", "SUSC-18B",
                     "SUSC-18C", "SUSC-18D", "SUSC-18E", "SUSC-18E2", "SUSC-18F", "SUSC-18G"]:
        assert esperado in marcos, f"marco ausente: {esperado}"


def test_recife_como_referencia_forte_tecnica_review_only():
    lin = _read(S.LINHAGEM)
    rec = [r for r in lin if r["city"] == "recife"]
    assert rec
    assert all(r["review_only"] == "true" for r in rec)
    assert any(r["evidence_tier"] == "C_footprint_tecnico_sar" for r in rec)
    assert any("forte" in r["status"] for r in rec)


def test_curitiba_oficial_bloqueada_por_geometria():
    lin = _read(S.LINHAGEM)
    oficiais = [r for r in lin if r["city"] == "curitiba" and r["entidade"] == "event_record"]
    assert oficiais
    assert all(r["geometry_id"] == "not_available" for r in oficiais)
    assert any("sem_geometria" in r["status"] for r in oficiais)


def test_curitiba_sar_segunda_regiao_tecnica_nao_oficial():
    lin = _read(S.LINHAGEM)
    sar = [r for r in lin if r["geometry_id"].startswith("S18G_CUR_SAR")]
    assert sar
    for r in sar:
        assert "oficial" not in r["geometry_type"].lower()
        assert r["evidence_tier"] == "C_footprint_tecnico_sar"
        assert "nao" in r["justificativa_tecnica"].lower() and "oficial" in r["justificativa_tecnica"].lower()


def test_petropolis_misto_bloqueado_e_nao_promovido():
    lin = _read(S.LINHAGEM)
    pet = [r for r in lin if r["city"] == "petropolis" and "misto" in r["phenomenon_type"]]
    assert pet
    for r in pet:
        assert r["evidence_tier"] in {"E_contexto_rua_bairro", "F_area_risco_ou_suscetibilidade_nao_evento", "G_rejeitado_ou_insuficiente"}
        assert "separac" in r["justificativa_tecnica"].lower()


def test_17b_nao_e_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["marco_17b_criado"] is False
    assert summary["benchmark_17b_criado"] is False
    assert summary["status_17b_mestre"] in S.STATUS_17B_ALLOWED
    diag = _read(S.DIAG_17B)
    assert diag
    for r in diag:
        assert r["status_17b"] in S.STATUS_17B_ALLOWED
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*"))


def test_ground_truth_trainable_score_v7_proibidos_em_todas_matrizes():
    for path in (S.LINHAGEM, S.QUALIDADE):
        for r in _read(path):
            assert r["ground_truth"] == "false"
            assert r["eligible_for_training"] == "false"
            assert r["score_v7_allowed"] == "false"
    for r in _read(S.INVENTARIO):
        assert r["ground_truth_count"] == "0"
        assert r["trainable_count"] == "0"
        assert r["score_v7_count"] == "0"


def test_score_v6_intacto():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    for r in _read(S.DIAG_17B):
        assert r["score_v6_intacto"] == "true"


def test_vocabulario_publico_proibido_falha():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo}")


def test_saidas_publicas_sem_vocabulario_proibido():
    assert S.validate_public_text() == []


def test_plano_commit_exclui_local_runs():
    txt = (OUT / "plano_commit_seletivo_pos_18g.md").read_text(encoding="utf-8")
    assert "local_runs" in txt
    assert "nunca entra" in txt
    blocos = txt.split("```")
    assert len(blocos) > 1
    assert "local_runs" not in blocos[1]


def test_validator_completo_passa():
    summary = S.build_all()
    assert S.validate_outputs(summary) == []
