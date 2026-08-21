"""Testes do fechamento de downloads + integracao das evidencias externas (MV1).

Sem rede: o script roda em modo offline e usa os arquivos ja presentes na
quarentena local. Verifica estrutura, guardrails e a regra dura de que
`pode_virar_label_agora` e sempre `false`.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPT = ROOT / "scripts" / "curadoria_externa" / "revp_fechamento_downloads_e_integracao_evidencias_externas_mv1.py"

spec = importlib.util.spec_from_file_location("revp_fechamento_downloads_mv1", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PUBLICOS = ROOT / "outputs_public"


def setup_module(module):
    del module
    mod.run(do_network=False)


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_csvs_revisados_existem():
    for p in (mod.P_MANIFEST, mod.P_AUDIT, mod.P_EVENTS, mod.P_GEOMS, mod.P_LOG, mod.P_INT_TABLE):
        assert p.is_file(), p


def test_jsons_validos():
    for p in (mod.P_METRICS, mod.P_INT_METRICS):
        json.loads(p.read_text(encoding="utf-8"))


def test_log_downloads_colunas():
    rows = _read_csv(mod.P_LOG)
    assert rows
    for c in mod.LOG_COLS:
        assert c in rows[0], c
    # status dentro do conjunto permitido
    permitidos = {
        "baixado", "ja_existia_localmente", "bloqueado_sem_download_direto",
        "requer_download_manual", "falha_download", "bloqueado_sem_url",
        "bloqueado_licenca_ou_uso", "bloqueado_risco_metodologico",
    }
    for r in rows:
        assert r["status_download"] in permitidos, r["status_download"]


def test_integracao_colunas():
    rows = _read_csv(mod.P_INT_TABLE)
    assert rows
    for c in mod.INTEGRACAO_COLS:
        assert c in rows[0], c


def test_pode_virar_label_sempre_false():
    for r in _read_csv(mod.P_INT_TABLE):
        assert r["pode_virar_label_agora"] == "false", r["item"]


def test_nenhum_evento_promovido_a_label():
    rows = _read_csv(mod.P_EVENTS)
    assert rows
    assert all(r["pode_sustentar_label_patch_level"] == "false" for r in rows)


def test_sha256_real_para_arquivos_presentes():
    for r in _read_csv(mod.P_LOG):
        if r["status_download"] in {"baixado", "ja_existia_localmente"}:
            assert r["sha256_arquivo"] and len(r["sha256_arquivo"]) == 64, r["id_fonte"]


def test_nenhum_arquivo_pesado_gerado_por_esta_tarefa():
    """Os artefatos gerados por esta tarefa sao leves (.csv/.md/.json) e pequenos.

    Escopo restrito aos arquivos que este script cria (outputs_public ja contem
    artefatos de outros trabalhos, fora do escopo deste teste).
    """
    permitido = {".csv", ".md", ".json"}
    meus = (mod.P_MANIFEST, mod.P_AUDIT, mod.P_EVENTS, mod.P_GEOMS, mod.P_LOG,
            mod.P_REPORT, mod.P_METRICS, mod.P_INT_TABLE, mod.P_INT_METRICS, mod.P_INT_REPORT)
    for p in meus:
        assert p.is_file(), p
        assert p.suffix.lower() in permitido, f"extensao inesperada: {p}"
        assert p.stat().st_size < 2_000_000, f"artefato grande demais: {p}"


def test_bruto_so_em_quarentena_local():
    """O PDF baixado vive apenas em local_only/, nunca em outputs_public/."""
    nome = "nhess_23_1157_2023_petropolis_fev2022.pdf"
    assert (mod.QUARENTENA / "fontes_internacionais" / nome).is_file()
    assert not list(PUBLICOS.rglob(nome))


def test_colunas_e_chaves_em_portugues():
    # Amostra de colunas que devem estar em portugues nos artefatos publicos.
    for c in ("status_download", "motivo_falha_ou_bloqueio", "caminho_local_quarentena"):
        assert c in mod.LOG_COLS
    for c in ("pode_virar_label_agora", "bloqueador", "acao_recomendada", "observacao_metodologica"):
        assert c in mod.INTEGRACAO_COLS
    data = json.loads(mod.P_INT_METRICS.read_text(encoding="utf-8"))
    for k in ("status_integracao", "guardrails_preservados", "proximo_passo_recomendado"):
        assert k in data, k


def test_guardrails_no_relatorio():
    for p in (mod.P_REPORT, mod.P_INT_REPORT):
        txt = p.read_text(encoding="utf-8")
        assert "Guardrails preservados" in txt
        assert "review-only" in txt


def test_linguagem_proibida_ausente_nos_csvs():
    proibido = (
        "label confirmado", "negativo confirmado", "classe 0", "classe_0",
        "positivo gold", "ground truth fechado", "treino liberado",
        "modelo detecta", "modelo prediz", "validacao operacional",
    )
    for p in (mod.P_MANIFEST, mod.P_AUDIT, mod.P_EVENTS, mod.P_GEOMS, mod.P_LOG, mod.P_INT_TABLE):
        txt = p.read_text(encoding="utf-8").lower()
        for termo in proibido:
            assert termo not in txt, (p.name, termo)
