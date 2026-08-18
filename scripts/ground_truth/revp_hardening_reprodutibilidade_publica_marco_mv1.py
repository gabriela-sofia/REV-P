from __future__ import annotations

import csv
import hashlib
import json
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CAMINHO_RELATORIO = Path("outputs_public/execution_reports/revp_hardening_reprodutibilidade_publica_marco_mv1.md")
CAMINHO_MANIFESTO = Path("outputs_public/tables/revp_manifesto_mestre_marco_mv1.csv")
CAMINHO_INDICE = Path("outputs_public/tables/revp_indice_reprodutibilidade_marco_mv1.csv")
CAMINHO_LOCAL_ONLY = Path("outputs_public/tables/revp_dependencias_local_only_marco_mv1.csv")
CAMINHO_PLANO_PACOTE = Path("outputs_public/tables/revp_plano_pacote_externo_reprodutibilidade_mv1.csv")
CAMINHO_CHECKLIST = Path("outputs_public/tables/revp_checklist_reproducao_publica_marco_mv1.csv")
CAMINHO_METRICAS = Path("outputs_public/metrics/revp_hardening_reprodutibilidade_publica_marco_mv1.json")

SCRIPT_ATUAL = Path("scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py")
TESTE_ATUAL = Path("tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py")

INDICE_ARQUIVOS_EXTERNOS = Path("outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv")
MANIFESTO_FONTES_EXTERNAS = Path("outputs_public/tables/revp_manifesto_evidencias_externas_navegacao_mv1.csv")
MANIFESTO_DINO_LOCAL = Path("local_runs/dino_embeddings/v1ge/dino_expanded_embedding_manifest_v1ge.csv")
MANIFESTO_DINO_INPUTS = Path("manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv")
INVENTARIO_DINO_PUBLICO = Path("outputs_public/tables/table_dino_embedding_inventory.csv")

COLUNAS_MANIFESTO = [
    "artefato",
    "caminho",
    "tipo",
    "fase",
    "existe",
    "eh_publico_git",
    "eh_local_only",
    "eh_bruto_pesado",
    "tem_hash",
    "sha256",
    "tamanho_bytes",
    "gerado_por_script",
    "script_gerador",
    "teste_associado",
    "papel_no_marco",
    "status_reprodutibilidade",
    "observacao",
]

COLUNAS_INDICE = [
    "item",
    "categoria",
    "como_reproduzir",
    "comando_ou_procedimento",
    "entrada_necessaria",
    "saida_esperada",
    "hash_ou_contagem_esperada",
    "bloqueador",
    "status",
    "observacao",
]

COLUNAS_LOCAL_ONLY = [
    "id_dependencia",
    "caminho_local_relativo",
    "tipo_arquivo",
    "tamanho_bytes",
    "sha256",
    "fonte_original",
    "url_origem",
    "pode_ser_publicado_no_git",
    "motivo_nao_publicar_no_git",
    "necessario_para_reproduzir_qual_etapa",
    "alternativa_publica",
    "observacao",
]

COLUNAS_PLANO_PACOTE = [
    "item_pacote",
    "origem_local",
    "tipo",
    "tamanho_bytes",
    "sha256",
    "prioridade",
    "motivo_inclusao",
    "licenca_ou_uso",
    "risco_publicacao",
    "acao_antes_publicar",
    "observacao",
]

COLUNAS_CHECKLIST = [
    "item",
    "verificacao",
    "status",
    "bloqueia_reproducao_publica",
    "acao_necessaria",
    "observacao",
]

LINGUAGEM_PROIBIDA = [
    "modelo detecta",
    "modelo prediz",
    "Curitiba é negativo",
    "classe 0",
    "positivo confirmado",
    "negativo confirmado",
    "ground truth fechado",
    "treino liberado",
    "validação operacional",
]

STATUS_VALIDOS = {
    "reprodutivel_no_git",
    "verificavel_por_hash",
    "depende_local_only",
    "depende_download_externo",
    "depende_solicitacao_formal",
    "documental_nao_reexecutavel",
    "bloqueado_para_reproducao_publica",
}

FASES_PRINCIPAIS = [
    "restauracao_v2dz_v2ef",
    "auditoria_temporal",
    "validacao_label_free",
    "protocolo_fail_closed",
    "evidencias_externas",
    "integracao_final",
    "auditoria_critica_banca",
    "hardening_pre_stage",
    "hardening_reprodutibilidade_publica",
]

ARTEFATOS_PUBLICOS_BASE = [
    ("relatorio_restauracao_v2dz_v2ef", "outputs_public/execution_reports/revp_restauracao_manual_v2dz_v2ef.md", "relatorio", "restauracao_v2dz_v2ef", "documenta restauracao forense publica"),
    ("manifesto_restauracao_v2dz_v2ef", "outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_manifesto.csv", "csv", "restauracao_v2dz_v2ef", "inventario da restauracao forense"),
    ("validacao_restauracao_v2dz_v2ef", "outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_validacao.csv", "csv", "restauracao_v2dz_v2ef", "validacao da restauracao forense"),
    ("script_restauracao_v2dz_v2ef", "scripts/ground_truth/revp_v2dz_to_v2ef_orchestrator.py", "script", "restauracao_v2dz_v2ef", "gerador da restauracao"),
    ("teste_restauracao_v2dz_v2ef", "tests/test_revp_v2dz_to_v2ef_orchestrator.py", "teste", "restauracao_v2dz_v2ef", "teste da restauracao"),
    ("relatorio_auditoria_temporal", "outputs_public/execution_reports/revp_auditoria_prontidao_temporal_assets_mv1.md", "relatorio", "auditoria_temporal", "documenta prontidao temporal"),
    ("tabela_auditoria_temporal", "outputs_public/tables/revp_auditoria_prontidao_temporal_assets_mv1.csv", "csv", "auditoria_temporal", "tabela de auditoria temporal"),
    ("metricas_auditoria_temporal", "outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json", "json", "auditoria_temporal", "metricas de auditoria temporal"),
    ("script_auditoria_temporal", "scripts/ground_truth/revp_auditoria_prontidao_temporal_assets_mv1.py", "script", "auditoria_temporal", "gerador da auditoria temporal"),
    ("teste_auditoria_temporal", "tests/test_revp_auditoria_prontidao_temporal_assets_mv1.py", "teste", "auditoria_temporal", "teste da auditoria temporal"),
    ("relatorio_validacao_label_free", "outputs_public/execution_reports/revp_validacao_label_free_evidencia_estrutural_mv1.md", "relatorio", "validacao_label_free", "relatorio de validacao estrutural"),
    ("tabela_validacao_label_free", "outputs_public/tables/revp_validacao_label_free_evidencia_estrutural_mv1.csv", "csv", "validacao_label_free", "amostras label-free"),
    ("topologia_label_free", "outputs_public/tables/revp_matriz_topologica_cidades_mv1.csv", "csv", "validacao_label_free", "topologia entre cidades"),
    ("vizinhos_label_free", "outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv", "csv", "validacao_label_free", "vizinhos de embeddings"),
    ("guardrails_label_free", "outputs_public/tables/revp_guardrails_validacao_label_free_mv1.csv", "csv", "validacao_label_free", "guardrails label-free"),
    ("metricas_validacao_label_free", "outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json", "json", "validacao_label_free", "metricas label-free"),
    ("inventario_publico_dino", "outputs_public/tables/table_dino_embedding_inventory.csv", "csv", "validacao_label_free", "inventario publico dos 12 embeddings"),
    ("matriz_similaridade_dino", "outputs_public/tables/table_dino_similarity_matrix.csv", "csv", "validacao_label_free", "matriz publica de similaridade"),
    ("vizinhos_publicos_dino", "outputs_public/tables/table_dino_nearest_neighbors.csv", "csv", "validacao_label_free", "vizinhos publicos DINO"),
    ("script_validacao_label_free", "scripts/ground_truth/revp_validacao_label_free_evidencia_estrutural_mv1.py", "script", "validacao_label_free", "gerador label-free"),
    ("teste_validacao_label_free", "tests/test_revp_validacao_label_free_evidencia_estrutural_mv1.py", "teste", "validacao_label_free", "teste label-free"),
    ("relatorio_protocolo_fail_closed", "outputs_public/execution_reports/revp_protocolo_ground_truth_fail_closed_mv1.md", "relatorio", "protocolo_fail_closed", "documenta gates fail-closed"),
    ("ontologia_estados_label", "outputs_public/tables/revp_ontologia_estados_label_mv1.csv", "csv", "protocolo_fail_closed", "ontologia de estados futuros"),
    ("gates_readiness_treino", "outputs_public/tables/revp_gates_readiness_treino_mv1.csv", "csv", "protocolo_fail_closed", "gates de readiness"),
    ("dashboard_bloqueio_treino", "outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv", "csv", "protocolo_fail_closed", "dashboard de bloqueio"),
    ("metricas_protocolo_fail_closed", "outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json", "json", "protocolo_fail_closed", "metricas do protocolo"),
    ("script_protocolo_fail_closed", "scripts/ground_truth/revp_protocolo_ground_truth_fail_closed_mv1.py", "script", "protocolo_fail_closed", "gerador do protocolo"),
    ("teste_protocolo_fail_closed", "tests/test_revp_protocolo_ground_truth_fail_closed_mv1.py", "teste", "protocolo_fail_closed", "teste do protocolo"),
    ("manifesto_fontes_externas", "outputs_public/tables/revp_manifesto_evidencias_externas_navegacao_mv1.csv", "csv", "evidencias_externas", "fontes externas navegadas"),
    ("indice_arquivos_externos", "outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv", "csv", "evidencias_externas", "indice de arquivos baixados"),
    ("log_navegacao_externa", "outputs_public/tables/revp_log_navegacao_downloads_evidencias_externas_mv1.csv", "csv", "evidencias_externas", "log de navegacao"),
    ("relatorio_navegacao_externa", "outputs_public/execution_reports/revp_navegacao_downloads_evidencias_externas_mv1.md", "relatorio", "evidencias_externas", "relatorio de navegacao"),
    ("metricas_navegacao_externa", "outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json", "json", "evidencias_externas", "metricas de navegacao"),
    ("teste_navegacao_externa", "tests/test_revp_navegacao_downloads_evidencias_externas_mv1.py", "teste", "evidencias_externas", "teste de navegacao externa"),
    ("relatorio_integracao_final", "outputs_public/execution_reports/revp_integracao_final_marco_mv1_maturidade_revisao_humana.md", "relatorio", "integracao_final", "relatorio de integracao final"),
    ("matriz_maturidade", "outputs_public/tables/revp_matriz_maturidade_metodologica_mv1.csv", "csv", "integracao_final", "matriz de maturidade"),
    ("fila_revisao_humana", "outputs_public/tables/revp_fila_revisao_humana_candidatos_mv1.csv", "csv", "integracao_final", "fila de revisao humana"),
    ("bloqueadores_finais", "outputs_public/tables/revp_bloqueadores_finais_ground_truth_treino_mv1.csv", "csv", "integracao_final", "bloqueadores finais"),
    ("metricas_integracao_final", "outputs_public/metrics/revp_integracao_final_marco_mv1_maturidade_revisao_humana.json", "json", "integracao_final", "metricas de integracao final"),
    ("script_integracao_final", "scripts/ground_truth/revp_integracao_final_marco_mv1_maturidade_revisao_humana.py", "script", "integracao_final", "gerador da integracao final"),
    ("teste_integracao_final", "tests/test_revp_integracao_final_marco_mv1_maturidade_revisao_humana.py", "teste", "integracao_final", "teste da integracao final"),
    ("relatorio_auditoria_critica_banca", "outputs_public/execution_reports/revp_auditoria_critica_banca_marco_mv1.md", "relatorio", "auditoria_critica_banca", "auditoria critica para banca"),
    ("tabela_auditoria_critica_banca", "outputs_public/tables/revp_auditoria_critica_banca_marco_mv1.csv", "csv", "auditoria_critica_banca", "tabela da auditoria critica"),
    ("metricas_auditoria_critica_banca", "outputs_public/metrics/revp_auditoria_critica_banca_marco_mv1.json", "json", "auditoria_critica_banca", "metricas da auditoria critica"),
    ("checklist_pre_stage", "outputs_public/tables/revp_checklist_pre_stage_marco_mv1.csv", "csv", "hardening_pre_stage", "checklist pre-stage seletivo"),
    ("relatorio_hardening_reprodutibilidade", CAMINHO_RELATORIO.as_posix(), "relatorio", "hardening_reprodutibilidade_publica", "relatorio de reprodutibilidade publica"),
    ("manifesto_mestre_hardening", CAMINHO_MANIFESTO.as_posix(), "csv", "hardening_reprodutibilidade_publica", "manifesto mestre do marco"),
    ("indice_reprodutibilidade", CAMINHO_INDICE.as_posix(), "csv", "hardening_reprodutibilidade_publica", "indice de reproducao publica"),
    ("dependencias_local_only", CAMINHO_LOCAL_ONLY.as_posix(), "csv", "hardening_reprodutibilidade_publica", "indice de dependencias locais"),
    ("plano_pacote_externo", CAMINHO_PLANO_PACOTE.as_posix(), "csv", "hardening_reprodutibilidade_publica", "plano de pacote externo"),
    ("checklist_reproducao_publica", CAMINHO_CHECKLIST.as_posix(), "csv", "hardening_reprodutibilidade_publica", "checklist de reproducao publica"),
    ("metricas_hardening_reprodutibilidade", CAMINHO_METRICAS.as_posix(), "json", "hardening_reprodutibilidade_publica", "metricas de reproducao publica"),
    ("script_hardening_reprodutibilidade", SCRIPT_ATUAL.as_posix(), "script", "hardening_reprodutibilidade_publica", "gerador do pacote"),
    ("teste_hardening_reprodutibilidade", TESTE_ATUAL.as_posix(), "teste", "hardening_reprodutibilidade_publica", "teste do pacote"),
]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Hardening de reprodutibilidade publica do marco MV1.")
    parser.add_argument("--repo-root", default=".", help="Raiz do repositorio REV-P.")
    return parser


def ler_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ler_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def escrever_csv(path: Path, linhas: list[dict[str, str]], colunas: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=colunas, lineterminator="\n")
        writer.writeheader()
        writer.writerows(linhas)


def escrever_texto(path: Path, texto: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto, encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tamanho(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return str(path.stat().st_size)


def bool_txt(valor: bool) -> str:
    return "true" if valor else "false"


def caminho_posix(path: str | Path) -> str:
    return Path(str(path)).as_posix()


def eh_local(path: str) -> bool:
    normal = caminho_posix(path)
    return normal.startswith("local_only/") or normal.startswith("local_runs/")


def eh_publico_git(path: str) -> bool:
    normal = caminho_posix(path)
    if eh_local(normal):
        return False
    return normal.startswith(("outputs_public/", "docs/", "scripts/", "tests/", "manifests/"))


def eh_bruto_pesado(path: str, size: int) -> bool:
    normal = caminho_posix(path).lower()
    extensoes_brutas = (".tif", ".tiff", ".geotiff", ".local_geotiff", ".zip", ".7z", ".rar", ".npz")
    return size > 10 * 1024 * 1024 or normal.endswith(extensoes_brutas)


def status_para_artefato(path: str, existe: bool, local: bool, tipo: str) -> str:
    if local:
        return "depende_local_only"
    if not existe:
        return "documental_nao_reexecutavel" if tipo in {"relatorio", "csv", "json"} else "bloqueado_para_reproducao_publica"
    if tipo in {"script", "teste", "csv", "json", "relatorio"}:
        return "reprodutivel_no_git"
    return "verificavel_por_hash"


def obter_branch(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "desconhecido"
    texto = head.read_text(encoding="utf-8", errors="replace").strip()
    return texto.removeprefix("ref: refs/heads/") if texto.startswith("ref: refs/heads/") else texto


def artefato_row(
    repo_root: Path,
    artefato: str,
    caminho: str,
    tipo: str,
    fase: str,
    papel: str,
    script_gerador: str = "",
    teste_associado: str = "",
    observacao: str = "",
) -> dict[str, str]:
    rel = caminho_posix(caminho)
    path = repo_root / rel
    existe = path.exists()
    size = int(tamanho(path) or "0") if existe and path.is_file() else 0
    local = eh_local(rel)
    status = status_para_artefato(rel, existe, local, tipo)
    sha = sha256_file(path) if existe and path.is_file() else ""
    return {
        "artefato": artefato,
        "caminho": rel,
        "tipo": tipo,
        "fase": fase,
        "existe": bool_txt(existe),
        "eh_publico_git": bool_txt(eh_publico_git(rel)),
        "eh_local_only": bool_txt(local),
        "eh_bruto_pesado": bool_txt(eh_bruto_pesado(rel, size)),
        "tem_hash": bool_txt(bool(sha)),
        "sha256": sha,
        "tamanho_bytes": str(size) if existe and path.is_file() else "",
        "gerado_por_script": bool_txt(bool(script_gerador)),
        "script_gerador": script_gerador,
        "teste_associado": teste_associado,
        "papel_no_marco": papel,
        "status_reprodutibilidade": status,
        "observacao": observacao or "Artefato publico ou dependencia registrada sem promover label, treino ou decisao operacional.",
    }


def montar_manifesto(repo_root: Path, deps_local: list[dict[str, str]]) -> list[dict[str, str]]:
    linhas = [
        artefato_row(repo_root, artefato, caminho, tipo, fase, papel, observacao="Parte do pacote MV1 review-only.")
        for artefato, caminho, tipo, fase, papel in ARTEFATOS_PUBLICOS_BASE
    ]
    for dep in deps_local:
        rel = dep["caminho_local_relativo"]
        size = int(dep["tamanho_bytes"] or "0")
        linhas.append(
            {
                "artefato": dep["id_dependencia"],
                "caminho": rel,
                "tipo": dep["tipo_arquivo"],
                "fase": "dependencias_local_only",
                "existe": bool_txt((repo_root / rel).exists()),
                "eh_publico_git": "false",
                "eh_local_only": "true",
                "eh_bruto_pesado": bool_txt(eh_bruto_pesado(rel, size)),
                "tem_hash": bool_txt(bool(dep["sha256"])),
                "sha256": dep["sha256"],
                "tamanho_bytes": dep["tamanho_bytes"],
                "gerado_por_script": "false",
                "script_gerador": "",
                "teste_associado": "",
                "papel_no_marco": dep["necessario_para_reproduzir_qual_etapa"],
                "status_reprodutibilidade": "depende_local_only" if dep["sha256"] else "bloqueado_para_reproducao_publica",
                "observacao": "Dependencia local indexada para reproducao controlada; nao e conteudo publicavel no Git.",
            }
        )
    return linhas


def montar_dependencias_local_only(repo_root: Path) -> list[dict[str, str]]:
    linhas: list[dict[str, str]] = []
    for row in ler_csv(repo_root / INDICE_ARQUIVOS_EXTERNOS):
        caminho = caminho_posix(row.get("caminho_local_quarentena", ""))
        if not caminho:
            continue
        linhas.append(
            {
                "id_dependencia": row.get("id_arquivo", "") or f"ARQ_LOCAL_{len(linhas) + 1:03d}",
                "caminho_local_relativo": caminho,
                "tipo_arquivo": row.get("formato", "") or row.get("tipo_arquivo", "") or "arquivo_externo",
                "tamanho_bytes": row.get("tamanho_bytes", ""),
                "sha256": row.get("sha256_arquivo", ""),
                "fonte_original": row.get("url_pagina_origem", "") or row.get("id_fonte", ""),
                "url_origem": row.get("url_final_download", "") or row.get("url_pagina_origem", ""),
                "pode_ser_publicado_no_git": "false",
                "motivo_nao_publicar_no_git": "Arquivo bruto ou externo mantido em quarentena local; publicar apenas apos decisao de licenca, tamanho e escopo.",
                "necessario_para_reproduzir_qual_etapa": "verificacao de hashes e auditoria de evidencias externas",
                "alternativa_publica": "Reexecutar download oficial quando permitido ou publicar pacote externo controlado.",
                "observacao": "Arquivo local nao cria label e nao substitui revisao humana.",
            }
        )

    base = MANIFESTO_DINO_LOCAL.parent
    for row in ler_csv(repo_root / MANIFESTO_DINO_LOCAL):
        if row.get("success", "").strip().upper() != "SUCCESS":
            continue
        caminho = caminho_posix(base / row.get("embedding_path", ""))
        path = repo_root / caminho
        linhas.append(
            {
                "id_dependencia": row.get("dino_input_id", "") or f"DINO_LOCAL_{len(linhas) + 1:03d}",
                "caminho_local_relativo": caminho,
                "tipo_arquivo": "embedding_dinov2_npz",
                "tamanho_bytes": tamanho(path),
                "sha256": row.get("hash", "") or sha256_file(path),
                "fonte_original": "manifesto DINO local v1ge sem expor caminho absoluto de origem",
                "url_origem": "",
                "pode_ser_publicado_no_git": "false",
                "motivo_nao_publicar_no_git": "Embedding bruto permanece local; publicacao requer pacote externo e decisao explicita.",
                "necessario_para_reproduzir_qual_etapa": "reexecucao completa da analise DINOv2 label-free",
                "alternativa_publica": "Usar inventario publico, matriz de similaridade e hashes; publicar embeddings apenas em pacote externo revisado.",
                "observacao": "Embedding usado como insumo estrutural review-only; nao representa label.",
            }
        )

    if (repo_root / MANIFESTO_DINO_LOCAL).exists():
        linhas.append(
            {
                "id_dependencia": "MANIFESTO_DINO_LOCAL_V1GE",
                "caminho_local_relativo": MANIFESTO_DINO_LOCAL.as_posix(),
                "tipo_arquivo": "csv_indice_local",
                "tamanho_bytes": tamanho(repo_root / MANIFESTO_DINO_LOCAL),
                "sha256": sha256_file(repo_root / MANIFESTO_DINO_LOCAL),
                "fonte_original": "execucao local DINO v1ge",
                "url_origem": "",
                "pode_ser_publicado_no_git": "false",
                "motivo_nao_publicar_no_git": "Manifesto local contem detalhes de execucao local; usar derivado publico sanitizado.",
                "necessario_para_reproduzir_qual_etapa": "rastrear embeddings brutos locais",
                "alternativa_publica": "Inventario publico de embeddings e manifesto mestre deste pacote.",
                "observacao": "Nao expor caminhos absolutos de origem no pacote publico.",
            }
        )
    return linhas


def risco_licenca(licenca: str, tipo: str) -> str:
    texto = (licenca or "").lower()
    if "ibge" in texto or "public" in texto or "publico" in texto:
        return "baixo"
    if "desconhe" in texto or "restr" in texto or not texto:
        return "alto"
    if tipo.lower() in {"pdf", "xlsx"}:
        return "medio"
    return "medio"


def montar_plano_pacote(repo_root: Path, deps_local: list[dict[str, str]]) -> list[dict[str, str]]:
    linhas: list[dict[str, str]] = []
    externos_por_caminho = {
        caminho_posix(row.get("caminho_local_quarentena", "")): row
        for row in ler_csv(repo_root / INDICE_ARQUIVOS_EXTERNOS)
        if row.get("caminho_local_quarentena")
    }
    for dep in deps_local:
        rel = dep["caminho_local_relativo"]
        externo = externos_por_caminho.get(rel, {})
        licenca = externo.get("licenca_ou_uso", "") if externo else "execucao local; requer decisao explicita"
        tipo = dep["tipo_arquivo"]
        is_dino = tipo == "embedding_dinov2_npz"
        linhas.append(
            {
                "item_pacote": dep["id_dependencia"],
                "origem_local": rel,
                "tipo": tipo,
                "tamanho_bytes": dep["tamanho_bytes"],
                "sha256": dep["sha256"],
                "prioridade": "alta" if is_dino or externo.get("licenca_ou_uso", "").lower().find("ibge") >= 0 else "media",
                "motivo_inclusao": "Permite reexecutar a etapa DINOv2 label-free." if is_dino else "Permite verificar evidencia externa por hash e origem.",
                "licenca_ou_uso": licenca,
                "risco_publicacao": risco_licenca(licenca, tipo),
                "acao_antes_publicar": "Remover caminhos absolutos, revisar licenca e publicar fora do Git." if is_dino else "Revisar licenca, tamanho, termos de uso e necessidade de redistribuicao.",
                "observacao": "Nao incluir no Git por padrao; usar pacote externo controlado quando aprovado.",
            }
        )

    if (repo_root / MANIFESTO_DINO_INPUTS).exists():
        ids = {row.get("source_asset_id", "") for row in ler_csv(repo_root / MANIFESTO_DINO_INPUTS) if row.get("source_asset_id")}
        linhas.append(
            {
                "item_pacote": "sentinel_scene_asset_ids_mv1",
                "origem_local": MANIFESTO_DINO_INPUTS.as_posix(),
                "tipo": "indice_sentinel_sem_geotiff",
                "tamanho_bytes": tamanho(repo_root / MANIFESTO_DINO_INPUTS),
                "sha256": sha256_file(repo_root / MANIFESTO_DINO_INPUTS),
                "prioridade": "alta",
                "motivo_inclusao": "Preserva IDs de assets Sentinel usados como referencia sem publicar GeoTIFF pesado.",
                "licenca_ou_uso": "indice interno de reproducao; rasters exigem fonte externa propria",
                "risco_publicacao": "medio",
                "acao_antes_publicar": "Publicar apenas IDs e instrucoes; nao incluir GeoTIFF sem decisao explicita.",
                "observacao": f"Total de assets Sentinel referenciados: {len(ids)}.",
            }
        )

    linhas.extend(
        [
            {
                "item_pacote": "raw_checksums_e_indices_publicos",
                "origem_local": INDICE_ARQUIVOS_EXTERNOS.as_posix(),
                "tipo": "indice_csv_publico",
                "tamanho_bytes": tamanho(repo_root / INDICE_ARQUIVOS_EXTERNOS),
                "sha256": sha256_file(repo_root / INDICE_ARQUIVOS_EXTERNOS),
                "prioridade": "alta",
                "motivo_inclusao": "Permite verificar arquivos externos sem redistribuir brutos.",
                "licenca_ou_uso": "metadado publico do projeto",
                "risco_publicacao": "baixo",
                "acao_antes_publicar": "Manter somente metadados, hashes e origem oficial.",
                "observacao": "Nao contem payload bruto dos arquivos.",
            },
            {
                "item_pacote": "scripts_geradores_reproducao_mv1",
                "origem_local": "scripts/ground_truth",
                "tipo": "codigo_fonte",
                "tamanho_bytes": "",
                "sha256": "",
                "prioridade": "alta",
                "motivo_inclusao": "Permite reexecutar as etapas publicas do marco.",
                "licenca_ou_uso": "codigo do repositorio",
                "risco_publicacao": "baixo",
                "acao_antes_publicar": "Fazer stage seletivo dos scripts do marco.",
                "observacao": "Nao usar git add -A.",
            },
            {
                "item_pacote": "readme_reproducao_publica_mv1",
                "origem_local": CAMINHO_RELATORIO.as_posix(),
                "tipo": "documentacao",
                "tamanho_bytes": tamanho(repo_root / CAMINHO_RELATORIO),
                "sha256": sha256_file(repo_root / CAMINHO_RELATORIO),
                "prioridade": "alta",
                "motivo_inclusao": "Explica como reproduzir, verificar hashes e separar dependencias locais.",
                "licenca_ou_uso": "documentacao publica do projeto",
                "risco_publicacao": "baixo",
                "acao_antes_publicar": "Revisar linguagem publica em PT-BR tecnico.",
                "observacao": "Nao promove resultado operacional.",
            },
            {
                "item_pacote": "geotiffs_pesados_sentinel",
                "origem_local": "data/sentinel ou local_runs com rasters",
                "tipo": "geotiff_pesado",
                "tamanho_bytes": "",
                "sha256": "",
                "prioridade": "baixa",
                "motivo_inclusao": "Somente necessario para reproducao integral pixel-level futura.",
                "licenca_ou_uso": "depende da fonte Sentinel/Copernicus e da politica de redistribuicao",
                "risco_publicacao": "alto",
                "acao_antes_publicar": "Nao incluir sem decisao explicita; preferir instrucoes de download oficial.",
                "observacao": "Regra do marco: nao incluir GeoTIFF pesado no Git.",
            },
        ]
    )
    return linhas


def montar_indice_reprodutibilidade(manifesto: list[dict[str, str]], deps_local: list[dict[str, str]]) -> list[dict[str, str]]:
    contagens = Counter(row["status_reprodutibilidade"] for row in manifesto)
    return [
        {
            "item": "executar_hardening_reprodutibilidade",
            "categoria": "script_principal",
            "como_reproduzir": "Rodar o gerador publico do pacote de hardening.",
            "comando_ou_procedimento": "python scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py --repo-root .",
            "entrada_necessaria": "Repo REV-P com artefatos MV1 ja gerados.",
            "saida_esperada": "Relatorio, manifesto, indices, plano, checklist e JSON de metricas.",
            "hash_ou_contagem_esperada": f"{len(manifesto)} artefatos manifestados apos a execucao completa.",
            "bloqueador": "Nenhum para a etapa publica de metadados.",
            "status": "reprodutivel_no_git",
            "observacao": "Nao exige publicar dados brutos locais.",
        },
        {
            "item": "validar_csvs_jsons",
            "categoria": "qa_publica",
            "como_reproduzir": "Executar teste dedicado do pacote.",
            "comando_ou_procedimento": "python -m pytest tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py -q",
            "entrada_necessaria": "Script e outputs publicos.",
            "saida_esperada": "Teste dedicado verde.",
            "hash_ou_contagem_esperada": "7 artefatos publicos do hardening existem.",
            "bloqueador": "Falha de schema ou linguagem bloqueia reproducao publica.",
            "status": "reprodutivel_no_git",
            "observacao": "Teste nao altera dados brutos.",
        },
        {
            "item": "verificar_hashes_publicos",
            "categoria": "hash",
            "como_reproduzir": "Comparar sha256 do manifesto mestre com os arquivos existentes.",
            "comando_ou_procedimento": "python -m pytest tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py -q",
            "entrada_necessaria": "Arquivos publicos manifestados.",
            "saida_esperada": "Hashes presentes para arquivos existentes.",
            "hash_ou_contagem_esperada": str(contagens.get("verificavel_por_hash", 0) + contagens.get("reprodutivel_no_git", 0)),
            "bloqueador": "Arquivo ausente ou hash vazio.",
            "status": "verificavel_por_hash",
            "observacao": "Hash verifica integridade, nao cria evidencia operacional.",
        },
        {
            "item": "reproduzir_validacao_label_free",
            "categoria": "script_principal_mv1",
            "como_reproduzir": "Rodar a validacao estrutural DINOv2 label-free.",
            "comando_ou_procedimento": "python scripts/ground_truth/revp_validacao_label_free_evidencia_estrutural_mv1.py --repo-root .",
            "entrada_necessaria": "Inventario publico e matriz de similaridade; embeddings brutos apenas para reproducao integral.",
            "saida_esperada": "CSV, relatorio e JSON label-free.",
            "hash_ou_contagem_esperada": "12 embeddings publicos no piloto atual.",
            "bloqueador": "n=12 limita escopo; embeddings brutos ficam fora do Git.",
            "status": "depende_local_only",
            "observacao": "Valida estrutura e vizinhanca, sem label ou treino.",
        },
        {
            "item": "reproduzir_downloads_externos",
            "categoria": "evidencia_externa",
            "como_reproduzir": "Reexecutar downloads oficiais ou verificar arquivos por hash.",
            "comando_ou_procedimento": "Consultar outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv.",
            "entrada_necessaria": "URLs oficiais, acesso de rede e permissao de uso.",
            "saida_esperada": "Arquivos locais com os sha256 registrados.",
            "hash_ou_contagem_esperada": f"{len([d for d in deps_local if not d['tipo_arquivo'].startswith('embedding')])} dependencias externas/local-only registradas.",
            "bloqueador": "Algumas fontes exigem acesso externo, canal alternativo ou solicitacao formal.",
            "status": "depende_download_externo",
            "observacao": "Arquivos externos sao contexto e auditoria, nao decisao supervisionada.",
        },
        {
            "item": "solicitacoes_formais",
            "categoria": "fonte_externa_restrita",
            "como_reproduzir": "Abrir solicitacao formal quando a fonte nao oferece download publico direto.",
            "comando_ou_procedimento": "Consultar manifesto de fontes externas e registrar protocolo de solicitacao.",
            "entrada_necessaria": "Contato institucional e escopo de pedido.",
            "saida_esperada": "Arquivo ou resposta formal com hash futuro.",
            "hash_ou_contagem_esperada": "Contagem derivada do manifesto de fontes externas.",
            "bloqueador": "Dependencia de resposta externa.",
            "status": "depende_solicitacao_formal",
            "observacao": "Ausencia de resposta formal mantem o item bloqueado.",
        },
        {
            "item": "stage_seletivo",
            "categoria": "git",
            "como_reproduzir": "Selecionar manualmente apenas arquivos do pacote e dependencias publicas revisadas.",
            "comando_ou_procedimento": "git add <lista explicita>; nunca usar git add -A",
            "entrada_necessaria": "Revisao manual dos caminhos nao rastreados.",
            "saida_esperada": "Somente arquivos aprovados na area staged.",
            "hash_ou_contagem_esperada": "Area staged inicialmente vazia antes da selecao.",
            "bloqueador": "git add -A pode incluir bruto local, cache ou artefato fora de escopo.",
            "status": "documental_nao_reexecutavel",
            "observacao": "Etapa deliberadamente manual para preservar escopo cientifico.",
        },
    ]


def montar_checklist(repo_root: Path, manifesto: list[dict[str, str]], deps_local: list[dict[str, str]]) -> list[dict[str, str]]:
    jsons_validos = all(ler_json(repo_root / row["caminho"]) is not None for row in manifesto if row["tipo"] == "json" and (repo_root / row["caminho"]).exists())
    heavy_public = [row for row in manifesto if row["eh_publico_git"] == "true" and row["eh_bruto_pesado"] == "true"]
    scripts_existem = all((repo_root / p).exists() for p in [SCRIPT_ATUAL, Path("scripts/ground_truth/revp_validacao_label_free_evidencia_estrutural_mv1.py")])
    return [
        ("artefatos_publicos_ptbr", "Relatorio e tabelas do pacote estao em PT-BR tecnico.", "ok", "false", "Revisar texto antes de stage.", "Sem prometer decisao operacional."),
        ("csvs_parseaveis", "CSVs do hardening foram escritos com cabecalho e lineterminator estavel.", "ok", "false", "Executar teste dedicado.", "Schemas obrigatorios preservados."),
        ("jsons_validos", "JSONs existentes no manifesto sao parseaveis.", "ok" if jsons_validos else "bloqueado", bool_txt(not jsons_validos), "Corrigir JSON invalido.", "Metricas do pacote usam valores fixos conservadores."),
        ("scripts_principais_executaveis", "Scripts publicos principais existem.", "ok" if scripts_existem else "atencao", bool_txt(not scripts_existem), "Restaurar script ausente se necessario.", "Execucao deve ser feita por comando explicito."),
        ("testes_principais_verdes", "Testes principais devem ser rodados antes de stage.", "pendente_execucao", "true", "Executar bateria pytest indicada.", "Este checklist registra o requisito; o resultado final vem da validacao."),
        ("hashes_reais_downloads", "Arquivos baixados tem sha256 no indice publico.", "ok" if all(d["sha256"] for d in deps_local if not d["tipo_arquivo"].startswith("embedding")) else "atencao", "false", "Completar hash ausente se houver.", "Hash verifica integridade, nao redistribui bruto."),
        ("sem_bruto_pesado_outputs_public", "Nenhum bruto pesado foi identificado em outputs_public.", "ok" if not heavy_public else "bloqueado", bool_txt(bool(heavy_public)), "Remover do escopo publico antes de stage.", f"Brutos pesados publicos detectados: {len(heavy_public)}."),
        ("local_only_apenas_dependencia", "local_only/local_runs aparecem como dependencia, nao como publico Git.", "ok", "false", "Manter fora de git add seletivo.", "Dependencias locais foram marcadas com pode_ser_publicado_no_git=false."),
        ("stage_seletivo_necessario", "Stage deve ser seletivo e manual.", "ok", "false", "Usar lista explicita de arquivos.", "Evita incluir cache, bruto ou artefato fora de escopo."),
        ("git_add_A_proibido", "git add -A nao e aceitavel para este marco.", "ok", "true", "Nao executar git add -A.", "A arvore contem muitos nao rastreados de escopo amplo."),
        ("sem_reivindicacao_operacional", "O pacote nao afirma decisao operacional.", "ok", "false", "Manter linguagem review-only.", "A reproducao publica e de metadados, hashes e scripts."),
        ("pode_virar_label_agora_0", "Nenhum item pode virar label agora.", "ok", "true", "Manter fila como revisao futura.", "Valor fixo do JSON: 0."),
        ("treino_bloqueado", "Treino supervisionado permanece bloqueado.", "ok", "true", "Manter G8 bloqueado.", "Nao ha amostra liberada para treino."),
        ("ground_truth_ausente", "Ground truth operacional permanece ausente.", "ok", "true", "Exigir revisao humana, overlay e anti-leakage antes de qualquer promocao.", "Estado fixo do JSON: ausente."),
    ]


def checklist_dict(rows: list[tuple[str, str, str, str, str, str]]) -> list[dict[str, str]]:
    return [dict(zip(COLUNAS_CHECKLIST, row)) for row in rows]


def contar_brutos_outputs_public(repo_root: Path) -> int:
    raiz = repo_root / "outputs_public"
    if not raiz.exists():
        return 0
    total = 0
    for path in raiz.rglob("*"):
        if path.is_file() and eh_bruto_pesado(path.relative_to(repo_root).as_posix(), path.stat().st_size):
            total += 1
    return total


def calcular_metricas(repo_root: Path, manifesto: list[dict[str, str]]) -> dict[str, object]:
    contagens = Counter(row["status_reprodutibilidade"] for row in manifesto)
    return {
        "status_reprodutibilidade": "PUBLICO_REPRODUTIVEL_PARCIAL_COM_DEPENDENCIAS_LOCAL_ONLY_E_EXTERNAS",
        "branch": obter_branch(repo_root),
        "total_artefatos_manifestados": len(manifesto),
        "artefatos_reprodutiveis_no_git": contagens.get("reprodutivel_no_git", 0),
        "artefatos_verificaveis_por_hash": contagens.get("verificavel_por_hash", 0),
        "artefatos_dependem_local_only": contagens.get("depende_local_only", 0),
        "artefatos_dependem_download_externo": contagens.get("depende_download_externo", 0),
        "artefatos_dependem_solicitacao_formal": contagens.get("depende_solicitacao_formal", 0),
        "brutos_pesados_em_outputs_public": contar_brutos_outputs_public(repo_root),
        "pode_reproduzir_marco_integral_so_com_git": False,
        "pode_verificar_marco_por_hashes_e_indices": True,
        "pode_fazer_stage_seletivo": "true_com_revisao_manual",
        "pode_usar_git_add_A": False,
        "ground_truth_operacional_status": "ausente",
        "treino_supervisionado_status": "bloqueado",
        "itens_podem_virar_label_agora": 0,
        "guardrails_preservados": [
            "review-only",
            "label-free",
            "sem positivos formais",
            "sem negativos formais",
            "sem treino supervisionado",
            "local_only fora do Git",
            "stage seletivo obrigatorio",
        ],
        "proximo_passo_recomendado": "Revisar manualmente o manifesto mestre e executar stage seletivo apenas dos arquivos publicos aprovados.",
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
    }


def gerar_relatorio(metricas: dict[str, object], manifesto: list[dict[str, str]], deps_local: list[dict[str, str]], plano: list[dict[str, str]]) -> str:
    por_status = Counter(row["status_reprodutibilidade"] for row in manifesto)
    publicos_git = [row for row in manifesto if row["eh_publico_git"] == "true" and row["existe"] == "true"]
    hashes = [row for row in manifesto if row["tem_hash"] == "true"]
    comandos = [
        "python scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py --repo-root .",
        "python -m pytest tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py -q",
        "python -m pytest tests/test_revp_integracao_final_marco_mv1_maturidade_revisao_humana.py -q",
        "python -m pytest tests/test_revp_protocolo_ground_truth_fail_closed_mv1.py -q",
        "python -m pytest tests/test_revp_navegacao_downloads_evidencias_externas_mv1.py -q",
        "python -m pytest tests/test_revp_validacao_label_free_evidencia_estrutural_mv1.py -q",
        "python -m pytest tests/test_revp_auditoria_prontidao_temporal_assets_mv1.py -q",
    ]
    return "\n".join(
        [
            "# Hardening de reprodutibilidade publica do marco MV1",
            "",
            "## 1. Escopo",
            "Este pacote documenta o que pode ser reproduzido publicamente no Git, o que pode ser verificado por hash e o que depende de arquivos locais, downloads externos ou solicitacao formal. O escopo permanece review-only e label-free.",
            "",
            "## 2. O que significa reprodutibilidade publica no REV-P",
            "Reprodutibilidade publica significa conseguir reexecutar scripts, validar schemas, comparar hashes e auditar proveniencia sem publicar dados brutos pesados ou dependencias de licenca incerta.",
            "",
            "## 3. O que e reproduzivel so com Git",
            f"Artefatos reproduziveis no Git: {por_status.get('reprodutivel_no_git', 0)}. Incluem scripts, testes, CSVs, JSONs e relatorios publicos quando os insumos ja estao no repositorio.",
            "",
            "## 4. O que e verificavel por hash",
            f"Artefatos com hash registrado: {len(hashes)}. O hash serve para integridade e rastreabilidade, nao para promover conclusao operacional.",
            "",
            "## 5. O que depende de `local_only`",
            f"Dependencias locais registradas: {len(deps_local)}. Elas incluem arquivos externos em quarentena e embeddings DINOv2 brutos mantidos fora do Git.",
            "",
            "## 6. O que depende de download externo",
            "Arquivos listados no indice de evidencias externas podem ser baixados novamente quando a fonte oficial permite. A reproducao integral depende de rede, licenca e disponibilidade da fonte.",
            "",
            "## 7. O que depende de solicitacao formal",
            "Fontes sem download publico direto devem permanecer bloqueadas ate haver resposta formal e hash do material recebido.",
            "",
            "## 8. O que precisa de pacote externo futuro",
            f"O plano de pacote externo tem {len(plano)} itens. Ele prioriza embeddings DINOv2, indices, hashes, scripts e instrucoes; GeoTIFF pesado nao entra sem decisao explicita.",
            "",
            "## 9. O que nao deve ser publicado no Git",
            "Nao publicar GeoTIFF pesado, rasters brutos, arquivos em `local_only`, embeddings brutos `.npz`, caches locais ou arquivos com licenca incerta sem revisao.",
            "",
            "## 10. Como reproduzir scripts principais",
            *[f"- `{cmd}`" for cmd in comandos],
            "",
            "## 11. Como validar CSVs/JSONs",
            "Use o teste dedicado do hardening para verificar colunas obrigatorias, JSON parseavel, valores fixos conservadores e ausencia de linguagem proibida nos campos estruturados.",
            "",
            "## 12. Como validar hashes",
            "Compare os campos `sha256` do manifesto mestre e dos indices locais com os arquivos existentes. Para arquivos externos, use o indice publico de downloads como fonte de hash esperada.",
            "",
            "## 13. Limite dos 12 embeddings",
            "Os 12 embeddings DINOv2 sustentam um piloto exploratorio label-free. Eles nao sustentam promocao automatica, treino supervisionado ou fechamento de ground truth operacional.",
            "",
            "## 14. Limite dos brutos externos",
            "Arquivos externos baixados ajudam rastreabilidade e revisao humana futura, mas continuam como contexto ou dependencia. Redistribuicao exige licenca, tamanho e decisao de escopo.",
            "",
            "## 15. Stage seletivo recomendado",
            "Recomendacao: usar apenas `git add <lista explicita>` para os arquivos publicos revisados. `git add -A` permanece proibido neste marco.",
            "",
            "## 16. Guardrails preservados",
            "- Sem treino supervisionado.",
            "- Sem labels novos.",
            "- Sem positivos formais.",
            "- Sem negativos formais.",
            "- `local_only` e `local_runs` ficam como dependencias, nao como publico Git.",
            "- `itens_podem_virar_label_agora` permanece 0.",
            "",
            "## 17. Conclusao",
            f"Status: `{metricas['status_reprodutibilidade']}`. O marco pode ser auditado publicamente por scripts, tabelas, JSONs, hashes e indices, mas a reproducao integral depende de dependencias locais e externas controladas.",
            "",
            "## Artefatos publicos existentes no Git",
            *[f"- `{row['caminho']}`" for row in publicos_git[:80]],
            "",
        ]
    )


def validar_linguagem_estruturada(grupos: list[list[dict[str, str]]], metricas: dict[str, object]) -> None:
    textos = []
    for grupo in grupos:
        for row in grupo:
            textos.extend(str(valor).lower() for valor in row.values())
    textos.append(json.dumps(metricas, ensure_ascii=False).lower())
    combinado = "\n".join(textos)
    encontrados = [termo for termo in LINGUAGEM_PROIBIDA if termo.lower() in combinado]
    if encontrados:
        raise ValueError(f"Linguagem proibida em campos estruturados: {', '.join(encontrados)}")


def executar(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    deps_local = montar_dependencias_local_only(repo_root)

    # Escreve primeiro os artefatos derivados para que o manifesto registre seus hashes reais.
    escrever_csv(repo_root / CAMINHO_LOCAL_ONLY, deps_local, COLUNAS_LOCAL_ONLY)
    plano = montar_plano_pacote(repo_root, deps_local)
    escrever_csv(repo_root / CAMINHO_PLANO_PACOTE, plano, COLUNAS_PLANO_PACOTE)

    manifesto = montar_manifesto(repo_root, deps_local)
    indice = montar_indice_reprodutibilidade(manifesto, deps_local)
    checklist = checklist_dict(montar_checklist(repo_root, manifesto, deps_local))
    metricas = calcular_metricas(repo_root, manifesto)

    validar_linguagem_estruturada([manifesto, indice, deps_local, plano, checklist], metricas)

    escrever_csv(repo_root / CAMINHO_MANIFESTO, manifesto, COLUNAS_MANIFESTO)
    escrever_csv(repo_root / CAMINHO_INDICE, indice, COLUNAS_INDICE)
    escrever_csv(repo_root / CAMINHO_CHECKLIST, checklist, COLUNAS_CHECKLIST)
    (repo_root / CAMINHO_METRICAS).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / CAMINHO_METRICAS).write_text(json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    escrever_texto(repo_root / CAMINHO_RELATORIO, gerar_relatorio(metricas, manifesto, deps_local, plano))
    return metricas


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    executar(Path(args.repo_root))


if __name__ == "__main__":
    main()
