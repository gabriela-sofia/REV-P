"""Testes simples da curadoria de evidencias externas (MV1).

Garante: CSVs existem com colunas obrigatorias, JSON valido com chaves
obrigatorias, relatorio existe com guardrails, e que os guardrails
metodologicos (review-only, sem label, Curitiba nunca negativo) sao
preservados nas saidas.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "curadoria_externa" / "revp_curadoria_evidencias_externas_mv1.py"

spec = importlib.util.spec_from_file_location("revp_curadoria_externa_mv1", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def setup_module(module):
    del module
    mod.run()


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_check_passa():
    assert mod.check() == 0


def test_manifesto_colunas_obrigatorias():
    rows = _read_csv(mod.P_MANIFEST)
    assert rows
    for c in mod.MANIFEST_COLS:
        assert c in rows[0]


def test_quatro_csvs_existem():
    for p in (mod.P_MANIFEST, mod.P_AUDIT, mod.P_EVENTS, mod.P_GEOMS):
        assert p.is_file(), p


def test_json_valido_com_chaves_obrigatorias():
    data = json.loads(mod.P_METRICS.read_text(encoding="utf-8"))
    for k in [
        "total_fontes_pesquisadas", "total_fontes_baixadas",
        "total_fontes_aprovadas_para_contexto", "total_fontes_bloqueadas",
        "total_eventos_candidatos", "total_geometrias_candidatas",
        "contagem_por_cidade", "contagem_por_familia_fonte",
        "contagem_por_categoria_evento", "fontes_com_geometria",
        "fontes_sem_geometria", "fontes_com_crs_declarado",
        "fontes_sem_crs_declarado", "eventos_potencialmente_revisaveis",
        "eventos_que_nao_podem_virar_label", "guardrails_preservados",
        "proximo_passo_recomendado",
    ]:
        assert k in data, k


def test_relatorio_tem_guardrails():
    txt = mod.P_REPORT.read_text(encoding="utf-8")
    assert "Guardrails preservados" in txt
    assert "review-only" in txt


def test_sha256_real_para_arquivos_presentes():
    """Todo arquivo marcado como presente deve ter SHA256 nao vazio e coerente."""
    for r in _read_csv(mod.P_MANIFEST):
        if r["status_download"] == "ja_presente_no_repositorio":
            assert r["sha256_arquivo"], r["id_fonte"]
            assert len(r["sha256_arquivo"]) == 64, r["id_fonte"]


def test_nenhum_evento_vira_label_sem_revisao():
    """pode_sustentar_label_patch_level deve ser false por padrao (fail-closed)."""
    rows = _read_csv(mod.P_EVENTS)
    assert rows
    # Nesta passada nenhum evento pode virar label.
    assert all(r["pode_sustentar_label_patch_level"] == "false" for r in rows)


def test_linguagem_proibida_ausente():
    """Nenhuma saida pode conter linguagem de claim proibida (label/negativo confirmado, classe 0, treino liberado)."""
    proibido = (
        "label confirmado", "negativo confirmado", "classe 0", "classe_0",
        "positivo gold", "ground truth fechado", "treino liberado",
        "modelo detecta", "modelo prediz", "validacao operacional",
    )
    # Apenas CSVs estruturados: o relatorio enuncia os guardrails ("nunca vira classe 0"),
    # entao a presenca desses termos la e protetora, nao um claim.
    for p in (mod.P_MANIFEST, mod.P_AUDIT, mod.P_EVENTS, mod.P_GEOMS):
        txt = p.read_text(encoding="utf-8").lower()
        for termo in proibido:
            assert termo not in txt, (p.name, termo)


def test_curitiba_nao_classificada_como_negativo():
    """Nenhuma linha de Curitiba pode ter status/evento de negativo formal ou classe 0."""
    for p in (mod.P_MANIFEST, mod.P_EVENTS, mod.P_GEOMS):
        for r in _read_csv(p):
            if r.get("cidade") == "Curitiba":
                blob = " ".join(r.values()).lower()
                assert "classe_0" not in blob and "negativo_confirmado" not in blob, r


def test_geometria_suscetibilidade_nao_e_overlay():
    """Geometria candidata nunca e adequada para overlay nesta tarefa."""
    rows = _read_csv(mod.P_GEOMS)
    assert all(r["adequada_para_overlay_patch"] == "false" for r in rows)
