"""Testes da navegacao real + download de evidencias externas (MV1)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPT = ROOT / "scripts" / "curadoria_externa" / "revp_navegacao_downloads_evidencias_externas_mv1.py"

spec = importlib.util.spec_from_file_location("revp_navegacao_downloads_mv1", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PUBLICOS = ROOT / "outputs_public"


def setup_module(module):
    del module
    mod.run()


def _read(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_artefatos_navegacao_existem():
    for p in (mod.P_MANIFEST, mod.P_AUDIT, mod.P_LOG, mod.P_ARQ, mod.P_EVENTS,
              mod.P_GEOMS, mod.P_REPORT, mod.P_METRICS, mod.P_INT_TABLE,
              mod.P_INT_METRICS, mod.P_INT_REPORT):
        assert p.is_file(), p


def test_colunas_obrigatorias():
    for path, cols in [(mod.P_LOG, mod.LOG_COLS), (mod.P_ARQ, mod.ARQ_COLS),
                       (mod.P_INT_TABLE, mod.INTEGRACAO_COLS)]:
        rows = _read(path)
        assert rows, path
        for c in cols:
            assert c in rows[0], (path.name, c)


def test_jsons_validos():
    for p in (mod.P_METRICS, mod.P_INT_METRICS):
        json.loads(p.read_text(encoding="utf-8"))


def test_log_tem_tentativa_por_fonte_pendente():
    rows = _read(mod.P_LOG)
    fontes_no_log = {r["id_fonte"] for r in rows}
    for fid in mod.NAV:
        assert fid in fontes_no_log, f"sem tentativa registrada para {fid}"


def test_nenhuma_fonte_fica_requer_download_manual():
    """Nenhuma fonte pendente pode terminar como requer_download_manual."""
    proibido = "requer_download_manual"
    for r in _read(mod.P_MANIFEST):
        assert r["status_download"] != proibido, r["id_fonte"]
    for r in _read(mod.P_LOG):
        assert r["status_download"] != proibido, r["id_fonte"]
    # cada fonte navegada tem status detalhado permitido
    permitidos = {
        "baixado", "baixado_parcialmente", "fonte_alternativa_oficial_baixada",
        "bloqueado_apos_navegacao_real", "requer_solicitacao_formal", "falha_download",
    }
    for fid, nav in mod.NAV.items():
        assert nav["status"] in permitidos, (fid, nav["status"])


def test_todo_arquivo_baixado_tem_sha256_real():
    for r in _read(mod.P_ARQ):
        assert r["sha256_arquivo"] and len(r["sha256_arquivo"]) == 64, r["id_arquivo"]
        assert int(r["tamanho_bytes"]) > 0, r["id_arquivo"]


def test_todo_arquivo_baixado_em_local_only():
    for r in _read(mod.P_ARQ):
        caminho = r["caminho_local_quarentena"]
        assert caminho.startswith("local_only/"), caminho
        assert (ROOT / caminho).is_file(), caminho


def test_nenhum_bruto_pesado_em_outputs_public_desta_tarefa():
    permitido = {".csv", ".md", ".json"}
    meus = (mod.P_MANIFEST, mod.P_AUDIT, mod.P_LOG, mod.P_ARQ, mod.P_EVENTS, mod.P_GEOMS,
            mod.P_REPORT, mod.P_METRICS, mod.P_INT_TABLE, mod.P_INT_METRICS, mod.P_INT_REPORT)
    for p in meus:
        assert p.suffix.lower() in permitido, p
        assert p.stat().st_size < 2_000_000, p


def test_pode_virar_label_sempre_false():
    for r in _read(mod.P_INT_TABLE):
        assert r["pode_virar_label_agora"] == "false", r["item"]
    for r in _read(mod.P_EVENTS):
        assert r["pode_sustentar_label_patch_level"] == "false", r["id_fonte"]


def test_geometrias_nao_viram_overlay():
    rows = _read(mod.P_GEOMS)
    assert rows
    assert all(r["adequada_para_overlay_patch"] == "false" for r in rows)


def test_guardrails_no_relatorio():
    for p in (mod.P_REPORT, mod.P_INT_REPORT):
        txt = p.read_text(encoding="utf-8")
        assert "Guardrails preservados" in txt
        assert "review-only" in txt


def test_linguagem_proibida_ausente_csvs():
    proibido = ("label confirmado", "negativo confirmado", "classe 0", "classe_0",
                "positivo gold", "ground truth fechado", "treino liberado",
                "modelo detecta", "modelo prediz", "validacao operacional")
    for p in (mod.P_MANIFEST, mod.P_AUDIT, mod.P_LOG, mod.P_ARQ, mod.P_EVENTS, mod.P_GEOMS, mod.P_INT_TABLE):
        txt = p.read_text(encoding="utf-8").lower()
        for termo in proibido:
            assert termo not in txt, (p.name, termo)
