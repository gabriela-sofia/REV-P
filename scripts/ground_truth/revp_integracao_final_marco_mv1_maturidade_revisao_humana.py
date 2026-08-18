from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path


CAMINHO_RELATORIO = Path("outputs_public/execution_reports/revp_integracao_final_marco_mv1_maturidade_revisao_humana.md")
CAMINHO_MATURIDADE = Path("outputs_public/tables/revp_matriz_maturidade_metodologica_mv1.csv")
CAMINHO_FILA = Path("outputs_public/tables/revp_fila_revisao_humana_candidatos_mv1.csv")
CAMINHO_EVIDENCIAS_GATES = Path("outputs_public/tables/revp_matriz_evidencias_externas_gates_mv1.csv")
CAMINHO_BLOQUEADORES = Path("outputs_public/tables/revp_bloqueadores_finais_ground_truth_treino_mv1.csv")
CAMINHO_METRICAS = Path("outputs_public/metrics/revp_integracao_final_marco_mv1_maturidade_revisao_humana.json")

ENTRADAS_OBRIGATORIAS = [
    Path("outputs_public/execution_reports/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md"),
    Path("outputs_public/tables/revp_manifesto_marco_validacao_label_free_evidencia_estrutural_mv1.csv"),
    Path("outputs_public/metrics/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json"),
    Path("outputs_public/tables/revp_proximos_passos_pos_marco_label_free_mv1.csv"),
    Path("outputs_public/tables/revp_validacao_label_free_evidencia_estrutural_mv1.csv"),
    Path("outputs_public/tables/revp_matriz_topologica_cidades_mv1.csv"),
    Path("outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv"),
    Path("outputs_public/tables/revp_guardrails_validacao_label_free_mv1.csv"),
    Path("outputs_public/execution_reports/revp_validacao_label_free_evidencia_estrutural_mv1.md"),
    Path("outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json"),
    Path("outputs_public/tables/revp_auditoria_prontidao_temporal_assets_mv1.csv"),
    Path("outputs_public/execution_reports/revp_auditoria_prontidao_temporal_assets_mv1.md"),
    Path("outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json"),
    Path("docs/metodologia_cientifica/revp_ontologia_labels_ground_truth_mv1.md"),
    Path("docs/metodologia_cientifica/revp_politica_evidencia_negativa_mv1.md"),
    Path("docs/metodologia_cientifica/revp_politica_anti_leakage_mv1.md"),
    Path("outputs_public/tables/revp_ontologia_estados_label_mv1.csv"),
    Path("outputs_public/tables/revp_gates_readiness_treino_mv1.csv"),
    Path("outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv"),
    Path("outputs_public/execution_reports/revp_protocolo_ground_truth_fail_closed_mv1.md"),
    Path("outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json"),
    Path("outputs_public/tables/revp_manifesto_evidencias_externas_navegacao_mv1.csv"),
    Path("outputs_public/tables/revp_auditoria_fontes_externas_navegacao_mv1.csv"),
    Path("outputs_public/tables/revp_log_navegacao_downloads_evidencias_externas_mv1.csv"),
    Path("outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv"),
    Path("outputs_public/tables/revp_indice_eventos_externos_candidatos_navegacao_mv1.csv"),
    Path("outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv"),
    Path("outputs_public/execution_reports/revp_navegacao_downloads_evidencias_externas_mv1.md"),
    Path("outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json"),
    Path("outputs_public/execution_reports/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.md"),
    Path("outputs_public/tables/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.csv"),
    Path("outputs_public/metrics/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.json"),
]

COLUNAS_MATURIDADE = [
    "camada",
    "status_atual",
    "nivel_maturidade",
    "evidencia_atual",
    "artefatos_suporte",
    "bloqueador_principal",
    "acao_para_evoluir",
    "pode_sustentar_artigo_agora",
    "pode_sustentar_ground_truth_agora",
    "pode_sustentar_treino_agora",
    "observacao_metodologica",
]

COLUNAS_FILA = [
    "id_item_revisao",
    "tipo_item",
    "cidade",
    "regiao",
    "fonte_ou_artefato",
    "categoria",
    "evidencia_disponivel",
    "geometria_disponivel",
    "crs_disponivel",
    "janela_temporal_disponivel",
    "risco_circularidade",
    "risco_landslide_vs_flood",
    "pode_virar_label_agora",
    "pode_apoiar_ground_truth_futuro",
    "gates_potencialmente_afetados",
    "gates_bloqueados",
    "acao_humana_necessaria",
    "prioridade_revisao",
    "observacao_metodologica",
]

COLUNAS_EVIDENCIAS_GATES = [
    "id_fonte_ou_arquivo",
    "cidade",
    "tipo_evidencia",
    "gate",
    "pode_ajudar_gate",
    "como_pode_ajudar",
    "bloqueio_atual",
    "evidencia_insuficiente_para",
    "observacao",
]

COLUNAS_BLOQUEADORES = [
    "bloqueador",
    "categoria",
    "status",
    "impacto",
    "artefato_que_comprova",
    "acao_necessaria",
    "prioridade",
    "observacao",
]

GATES = [
    "G0_patch_id_valido",
    "G1_asset_id_valido",
    "G2_janela_temporal_fechada",
    "G3_geometria_espacial_fechada",
    "G4_fonte_label_independente",
    "G5_revisao_humana_completa",
    "G6_politica_negativo_satisfeita",
    "G7_anti_leakage_aprovado",
    "G8_liberado_para_treino_true",
]

GUARDRAILS = [
    "nenhum modelo treinado",
    "nenhum label criado",
    "nenhum positivo formal criado",
    "nenhum negativo formal criado",
    "ground truth operacional ausente",
    "DINOv2 usado apenas em leitura label-free",
    "evidência contextual não vira label",
    "geometria contextual não vira evento observado",
    "landslide e flood permanecem separados",
    "Curitiba não vira negativo formal por default",
    "unknown não vira negativo",
    "fila de revisão humana não libera treino",
]

LINGUAGEM_PROIBIDA = [
    "label confirmado",
    "negativo confirmado",
    "classe 0",
    "positivo gold confirmado",
    "ground truth fechado",
    "treino liberado",
    "modelo detecta",
    "modelo prediz",
    "validação operacional",
]

DECISAO_FINAL = "MARCO_MV1_CONSOLIDADO_REVIEW_ONLY_COM_FILA_DE_REVISAO_HUMANA_SEM_LIBERAR_TREINO"
PROXIMO_PASSO = (
    "Criar uma rodada de revisão humana/adjudicação sobre a fila de candidatos, começando pelos itens com "
    "geometria contextual mais útil e menor risco de circularidade, sem liberar label automaticamente."
)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Integração final MV1 de maturidade e revisão humana.")
    parser.add_argument("--repo-root", default=".", help="Raiz do repositório REV-P.")
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


def obter_branch(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "desconhecido"
    texto = head.read_text(encoding="utf-8", errors="replace").strip()
    if texto.startswith("ref: refs/heads/"):
        return texto.removeprefix("ref: refs/heads/")
    return texto or "desconhecido"


def bool_texto(valor: str) -> bool:
    return str(valor or "").strip().lower() in {"true", "sim", "yes", "1"}


def texto(row: dict[str, str], *campos: str) -> str:
    for campo in campos:
        valor = str(row.get(campo, "")).strip()
        if valor:
            return valor
    return ""


def artefatos_encontrados(repo_root: Path) -> list[str]:
    return [path.as_posix() for path in ENTRADAS_OBRIGATORIAS if (repo_root / path).exists()]


def artefatos_ausentes(repo_root: Path) -> list[str]:
    return [path.as_posix() for path in ENTRADAS_OBRIGATORIAS if not (repo_root / path).exists()]


def montar_maturidade(repo_root: Path) -> list[dict[str, str]]:
    protocolo = ler_json(repo_root / "outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json")
    fechamento = ler_json(repo_root / "outputs_public/metrics/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json")
    navegacao = ler_json(repo_root / "outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json")
    integracao_ext = ler_json(repo_root / "outputs_public/metrics/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.json")

    return [
        {
            "camada": "restauracao_v2dz_v2ef",
            "status_atual": "restaurada_como_trilha_forense",
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": str(fechamento.get("restauracao_v2dz_v2ef_status", "restauracao forense documentada")),
            "artefatos_suporte": "outputs_public/execution_reports/revp_restauracao_manual_v2dz_v2ef.md",
            "bloqueador_principal": "nao_fecha_ground_truth_operacional",
            "acao_para_evoluir": "submeter registros restaurados a revisão humana e overlay patch-evento",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Resolve recuperabilidade e trilha forense, não cria alvo supervisionado.",
        },
        {
            "camada": "auditoria_temporal",
            "status_atual": str(fechamento.get("auditoria_temporal_status", "metadados temporais insuficientes")),
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": "164 patches e 508 assets auditados; Trilha A segue bloqueada.",
            "artefatos_suporte": "outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json",
            "bloqueador_principal": "ausencia_multidata_temporal_e_cloud_cover",
            "acao_para_evoluir": "registrar múltiplas datas Sentinel e cobertura de nuvem por patch",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Auditoria resolve a decisão A/B, mas bloqueia deslocamento temporal amplo.",
        },
        {
            "camada": "validacao_label_free_dinov2",
            "status_atual": str(fechamento.get("validacao_label_free_status", "piloto label-free executado")),
            "nivel_maturidade": "piloto",
            "evidencia_atual": "12 embeddings DINOv2 válidos; decisão PILOTO_LABEL_FREE_COM_LIMITACAO_AMOSTRAL.",
            "artefatos_suporte": "outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json",
            "bloqueador_principal": "n_embeddings_12",
            "acao_para_evoluir": "expandir embeddings para pelo menos 30 patches e idealmente 59",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "DINOv2 sustenta topologia label-free, não prova inundação.",
        },
        {
            "camada": "curadoria_externa",
            "status_atual": "integrada_review_only",
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": f"{integracao_ext.get('itens_potenciais_para_revisao_humana', 0)} itens potenciais para revisão humana.",
            "artefatos_suporte": "outputs_public/tables/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.csv",
            "bloqueador_principal": "fontes_contextuais_nao_sao_labels",
            "acao_para_evoluir": "classificar fontes por uso em revisão humana e anti-leakage",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Curadoria externa adiciona contexto, não decisão supervisionada.",
        },
        {
            "camada": "downloads_externos",
            "status_atual": "baixados_e_auditados_com_ressalvas",
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": f"{navegacao.get('total_arquivos_baixados', 0)} arquivos baixados com SHA256.",
            "artefatos_suporte": "outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv",
            "bloqueador_principal": "downloads_nao_equivalem_a_label",
            "acao_para_evoluir": "submeter arquivos baixados a revisão humana e separação label-feature",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Download real reforça rastreabilidade, não autoriza alvo supervisionado.",
        },
        {
            "camada": "geometrias_contextuais",
            "status_atual": "candidatas_para_contexto",
            "nivel_maturidade": "pronto_para_revisao_humana",
            "evidencia_atual": f"{navegacao.get('total_geometrias_candidatas', 0)} geometrias candidatas.",
            "artefatos_suporte": "outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv",
            "bloqueador_principal": "geometria_contextual_nao_e_evento_observado",
            "acao_para_evoluir": "revisar utilidade para overlay e registrar se representa evento ou contexto",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Pode apoiar leitura territorial futura, sem virar label automaticamente.",
        },
        {
            "camada": "eventos_candidatos",
            "status_atual": "candidatos_para_revisao_futura",
            "nivel_maturidade": "pronto_para_revisao_humana",
            "evidencia_atual": f"{navegacao.get('total_eventos_candidatos', 0)} eventos candidatos.",
            "artefatos_suporte": "outputs_public/tables/revp_indice_eventos_externos_candidatos_navegacao_mv1.csv",
            "bloqueador_principal": "sem_revisao_humana_e_sem_overlay_patch_evento",
            "acao_para_evoluir": "adjudicar eventos e separar landslide vs flood",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Evento candidato exige revisão humana antes de qualquer promoção.",
        },
        {
            "camada": "protocolo_fail_closed",
            "status_atual": str(protocolo.get("status_protocolo", "formalizado")),
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": "Gates G0-G8 formalizados com bloqueio de treino.",
            "artefatos_suporte": "outputs_public/execution_reports/revp_protocolo_ground_truth_fail_closed_mv1.md",
            "bloqueador_principal": "nenhuma_amostra_aprovada_em_G8",
            "acao_para_evoluir": "aplicar gates por item da fila de revisão humana",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Protocolo é maturidade metodológica, não liberação de amostras.",
        },
        {
            "camada": "ontologia_labels",
            "status_atual": "formalizada_para_estados_futuros",
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": "7 estados de label futuros documentados.",
            "artefatos_suporte": "outputs_public/tables/revp_ontologia_estados_label_mv1.csv",
            "bloqueador_principal": "estados_futuros_sem_instancias_aprovadas",
            "acao_para_evoluir": "aplicar ontologia após revisão humana/adjudicação",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Ontologia não preenche labels reais.",
        },
        {
            "camada": "politica_negativos",
            "status_atual": "formalizada_sem_negativos_formais",
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": "Política exige evidência explícita de não inundação.",
            "artefatos_suporte": "docs/metodologia_cientifica/revp_politica_evidencia_negativa_mv1.md",
            "bloqueador_principal": "sem_negativos_formais",
            "acao_para_evoluir": "obter evidência explícita e revisão humana para negativos futuros",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Ausência de evidência não vira negativo.",
        },
        {
            "camada": "anti_leakage",
            "status_atual": "politica_formalizada_sem_aprovacao_por_amostra",
            "nivel_maturidade": "consolidado_para_contexto",
            "evidencia_atual": "Fonte de label e fonte de feature devem ser separadas.",
            "artefatos_suporte": "docs/metodologia_cientifica/revp_politica_anti_leakage_mv1.md",
            "bloqueador_principal": "nenhuma_amostra_auditada_contra_politica",
            "acao_para_evoluir": "criar matriz label-feature por candidato",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Anti-leakage é obrigatório antes de qualquer baseline supervisionado.",
        },
        {
            "camada": "revisao_humana",
            "status_atual": "pendente",
            "nivel_maturidade": "bloqueado_para_treino",
            "evidencia_atual": "Fila futura criada neste pacote; revisão ainda não executada.",
            "artefatos_suporte": CAMINHO_FILA.as_posix(),
            "bloqueador_principal": "sem_revisao_humana",
            "acao_para_evoluir": "executar revisão humana/adjudicação item a item",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Fila é preparação, não decisão operacional.",
        },
        {
            "camada": "ground_truth_patch_level",
            "status_atual": "ausente",
            "nivel_maturidade": "bloqueado_para_treino",
            "evidencia_atual": "Dashboard fail-closed registra ground truth operacional ausente.",
            "artefatos_suporte": "outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv",
            "bloqueador_principal": "sem_ground_truth_patch_level",
            "acao_para_evoluir": "fechar evento, geometria, temporalidade, revisão humana e anti-leakage",
            "pode_sustentar_artigo_agora": "true",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Marco review-only mantém ausência explícita.",
        },
        {
            "camada": "feature_table_multimodal",
            "status_atual": "nao_liberada",
            "nivel_maturidade": "bloqueado_para_treino",
            "evidencia_atual": "Gates G4-G8 seguem bloqueados.",
            "artefatos_suporte": "outputs_public/tables/revp_gates_readiness_treino_mv1.csv",
            "bloqueador_principal": "feature_table_multimodal_nao_liberada",
            "acao_para_evoluir": "avaliar somente depois de ground truth real e anti-leakage por amostra",
            "pode_sustentar_artigo_agora": "false",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Tabela multimodal antes dos gates criaria risco de circularidade.",
        },
        {
            "camada": "treino_supervisionado",
            "status_atual": str(protocolo.get("treino_supervisionado_status", "bloqueado")),
            "nivel_maturidade": "bloqueado_para_treino",
            "evidencia_atual": "G8 permanece bloqueado.",
            "artefatos_suporte": "outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json",
            "bloqueador_principal": "treino_supervisionado_bloqueado",
            "acao_para_evoluir": "não avaliar treino até satisfazer G0-G8",
            "pode_sustentar_artigo_agora": "false",
            "pode_sustentar_ground_truth_agora": "false",
            "pode_sustentar_treino_agora": "false",
            "observacao_metodologica": "Estado final preserva bloqueio de treino.",
        },
    ]


def risco_landslide(row: dict[str, str]) -> str:
    combinado = " ".join(str(valor) for valor in row.values()).lower()
    if "landslide" in combinado or "deslizamento" in combinado or "suscetibilidade" in combinado or "composto" in combinado:
        return "alto"
    return "baixo"


def gates_para_item(item: dict[str, str]) -> tuple[str, str]:
    potenciais = ["G4_fonte_label_independente", "G5_revisao_humana_completa"]
    if item["janela_temporal_disponivel"] == "true":
        potenciais.append("G2_janela_temporal_fechada")
    if item["geometria_disponivel"] == "true" and item["crs_disponivel"] == "true":
        potenciais.append("G3_geometria_espacial_fechada")
    if item["fonte_ou_artefato"]:
        potenciais.extend(["G0_patch_id_valido", "G1_asset_id_valido"])
    potenciais.append("G7_anti_leakage_aprovado")
    bloqueados = [gate for gate in GATES if gate not in {"G0_patch_id_valido", "G1_asset_id_valido"}]
    return ";".join(dict.fromkeys(potenciais)), ";".join(bloqueados)


def montar_fila(repo_root: Path) -> list[dict[str, str]]:
    eventos = ler_csv(repo_root / "outputs_public/tables/revp_indice_eventos_externos_candidatos_navegacao_mv1.csv")
    geometrias = ler_csv(repo_root / "outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv")
    arquivos = ler_csv(repo_root / "outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv")

    fila: list[dict[str, str]] = []
    for idx, row in enumerate(eventos, start=1):
        geom = bool_texto(row.get("geometria_disponivel", ""))
        crs = bool(texto(row, "crs"))
        janela = row.get("janela_temporal_status", "") == "fechada"
        item = {
            "id_item_revisao": f"REV_EVENTO_{idx:03d}",
            "tipo_item": "evento_candidato",
            "cidade": texto(row, "cidade") or "desconhecida",
            "regiao": texto(row, "regiao") or "desconhecida",
            "fonte_ou_artefato": texto(row, "id_evento_externo", "id_fonte"),
            "categoria": texto(row, "categoria_evento") or "evento_candidato",
            "evidencia_disponivel": "true" if row.get("fonte_observacional_independente", "") == "sim" else "parcial",
            "geometria_disponivel": "true" if geom else "false",
            "crs_disponivel": "true" if crs else "false",
            "janela_temporal_disponivel": "true" if janela else "false",
            "risco_circularidade": "medio",
            "risco_landslide_vs_flood": risco_landslide(row),
            "pode_virar_label_agora": "false",
            "pode_apoiar_ground_truth_futuro": "true",
            "gates_potencialmente_afetados": "",
            "gates_bloqueados": "",
            "acao_humana_necessaria": "revisar fonte, separar landslide vs flood, confirmar geometria, janela e independência",
            "prioridade_revisao": "alta" if geom and crs and janela else "media",
            "observacao_metodologica": texto(row, "observacao", "motivo_nao_label") or "Evento candidato não cria label automaticamente.",
        }
        item["gates_potencialmente_afetados"], item["gates_bloqueados"] = gates_para_item(item)
        fila.append(item)

    for idx, row in enumerate(geometrias, start=1):
        geom = bool_texto(row.get("representa_contexto", "")) or bool_texto(row.get("representa_suscetibilidade", "")) or bool_texto(row.get("representa_vulnerabilidade", ""))
        item = {
            "id_item_revisao": f"REV_GEOMETRIA_{idx:03d}",
            "tipo_item": "geometria_contextual",
            "cidade": texto(row, "cidade") or "desconhecida",
            "regiao": texto(row, "regiao") or "desconhecida",
            "fonte_ou_artefato": texto(row, "id_geometria", "id_fonte"),
            "categoria": texto(row, "tipo_geometria") or "geometria_contextual",
            "evidencia_disponivel": "true" if geom else "parcial",
            "geometria_disponivel": "true",
            "crs_disponivel": "true" if texto(row, "crs") else "false",
            "janela_temporal_disponivel": "false",
            "risco_circularidade": "medio",
            "risco_landslide_vs_flood": risco_landslide(row),
            "pode_virar_label_agora": "false",
            "pode_apoiar_ground_truth_futuro": "true",
            "gates_potencialmente_afetados": "",
            "gates_bloqueados": "",
            "acao_humana_necessaria": "revisar se a geometria é contexto, suscetibilidade ou evento observado antes de qualquer uso patch-level",
            "prioridade_revisao": "alta" if texto(row, "crs") else "media",
            "observacao_metodologica": texto(row, "observacao", "motivo_bloqueio_overlay") or "Geometria contextual não equivale a evento observado.",
        }
        item["gates_potencialmente_afetados"], item["gates_bloqueados"] = gates_para_item(item)
        fila.append(item)

    for idx, row in enumerate(arquivos, start=1):
        tem_geo = bool_texto(row.get("tem_geometria", ""))
        item = {
            "id_item_revisao": f"REV_ARQUIVO_{idx:03d}",
            "tipo_item": "arquivo_baixado_contextual",
            "cidade": texto(row, "cidade") or "desconhecida",
            "regiao": texto(row, "regiao") or "desconhecida",
            "fonte_ou_artefato": texto(row, "id_arquivo", "id_fonte"),
            "categoria": texto(row, "categoria_evento", "tipo_arquivo") or "contexto_externo",
            "evidencia_disponivel": "true" if texto(row, "sha256_arquivo") else "parcial",
            "geometria_disponivel": "true" if tem_geo else "false",
            "crs_disponivel": "true" if texto(row, "crs") else "false",
            "janela_temporal_disponivel": "true" if texto(row, "periodo_referencia_inicio", "periodo_referencia_fim") else "false",
            "risco_circularidade": "medio",
            "risco_landslide_vs_flood": risco_landslide(row),
            "pode_virar_label_agora": "false",
            "pode_apoiar_ground_truth_futuro": "true" if tem_geo or texto(row, "sha256_arquivo") else "false",
            "gates_potencialmente_afetados": "",
            "gates_bloqueados": "",
            "acao_humana_necessaria": "revisar papel do arquivo como contexto, geometria ou fonte observacional candidata",
            "prioridade_revisao": "media" if tem_geo else "baixa",
            "observacao_metodologica": texto(row, "observacao") or "Arquivo baixado é insumo de revisão, não label.",
        }
        item["gates_potencialmente_afetados"], item["gates_bloqueados"] = gates_para_item(item)
        fila.append(item)

    return fila


def avaliar_gate(item: dict[str, str], gate: str) -> tuple[str, str, str, str, str]:
    if gate == "G0_patch_id_valido":
        return "parcial", "Pode orientar associação futura com patch.", "sem vínculo patch-level revisado", "identificação final de patch", "Não cria amostra por si só."
    if gate == "G1_asset_id_valido":
        return "parcial", "Pode documentar arquivo, fonte ou asset de apoio.", "asset ainda não ligado a amostra supervisionada", "asset de treino", "SHA256 e origem ajudam rastreabilidade."
    if gate == "G2_janela_temporal_fechada":
        ajuda = item["janela_temporal_disponivel"]
        return ajuda, "Pode apoiar janela temporal se datas forem revisadas." if ajuda == "true" else "Não fecha temporalidade.", "janela por patch pendente", "janela temporal de treino", "Datas precisam ser auditadas por patch."
    if gate == "G3_geometria_espacial_fechada":
        ajuda = "true" if item["geometria_disponivel"] == "true" and item["crs_disponivel"] == "true" else "false"
        return ajuda, "Pode apoiar revisão de overlay espacial." if ajuda == "true" else "Não traz geometria auditável suficiente.", "overlay patch-evento pendente", "geometria operacional patch-evento", "Geometria contextual ainda não é evento observado."
    if gate == "G4_fonte_label_independente":
        return "parcial", "Pode ser fonte candidata independente após revisão.", "independência ainda não adjudicada", "label operacional", "Fonte candidata deve ficar separada das features."
    if gate == "G5_revisao_humana_completa":
        return "true", "Item entra na fila de revisão humana.", "revisão humana ainda não executada", "decisão final", "A fila prepara revisão futura."
    if gate == "G6_politica_negativo_satisfeita":
        return "false", "Não há evidência explícita de não inundação.", "negativo formal ausente", "negativo formal", "Ausência de evidência não vira negativo."
    if gate == "G7_anti_leakage_aprovado":
        return "parcial", "Pode alimentar matriz label-feature futura.", "anti-leakage não aplicado por item", "aprovação anti-leakage", "Fonte de label e feature devem ser separadas."
    return "false", "G8 depende de todos os gates anteriores.", "gates anteriores pendentes", "liberação de treino", "Treino supervisionado permanece bloqueado."


def montar_evidencias_gates(fila: list[dict[str, str]]) -> list[dict[str, str]]:
    linhas = []
    for item in fila:
        for gate in GATES:
            ajuda, como, bloqueio, insuficiente, observacao = avaliar_gate(item, gate)
            linhas.append(
                {
                    "id_fonte_ou_arquivo": item["fonte_ou_artefato"],
                    "cidade": item["cidade"],
                    "tipo_evidencia": item["tipo_item"],
                    "gate": gate,
                    "pode_ajudar_gate": ajuda,
                    "como_pode_ajudar": como,
                    "bloqueio_atual": bloqueio,
                    "evidencia_insuficiente_para": insuficiente,
                    "observacao": observacao,
                }
            )
    return linhas


def montar_bloqueadores() -> list[dict[str, str]]:
    dados = [
        ("n_embeddings_12", "amostra_label_free", "ativo", "limita inferência estatística final", "outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json", "expandir embeddings para pelo menos 30 patches", "alta", "n=12 é piloto exploratório."),
        ("ausencia_multidata_temporal", "temporal", "ativo", "bloqueia Trilha A", "outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json", "registrar múltiplas datas Sentinel por patch", "alta", "Zero patches elegíveis para deslocamento temporal."),
        ("cloud_cover_insuficiente", "temporal", "ativo", "impede filtro temporal confiável", "outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json", "registrar cobertura de nuvem por asset", "alta", "Metadados de nuvem permanecem incompletos."),
        ("sem_ground_truth_patch_level", "ground_truth", "ativo", "bloqueia qualquer amostra supervisionada", "outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv", "fechar evento, geometria, temporalidade e revisão", "critica", "Ground truth operacional ausente."),
        ("sem_positivos_formais", "label", "ativo", "sem classe positiva formal", "outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json", "obter fonte independente e adjudicação", "critica", "Não há positivo ouro aprovado."),
        ("sem_negativos_formais", "negativo", "ativo", "sem contraste negativo formal", "docs/metodologia_cientifica/revp_politica_evidencia_negativa_mv1.md", "obter evidência explícita de não inundação", "critica", "Unknown e ausência não viram negativo."),
        ("sem_revisao_humana", "revisao", "ativo", "impede promoção de candidatos", CAMINHO_FILA.as_posix(), "executar revisão humana/adjudicação", "critica", "Fila futura não substitui decisão humana."),
        ("sem_overlay_patch_evento", "espacial", "ativo", "bloqueia ligação patch-evento", "outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv", "validar overlay com CRS e geometria de evento", "alta", "Geometria contextual ainda não fecha evento."),
        ("geometrias_contextuais_nao_evento", "espacial", "ativo", "impede uso direto como evento observado", "outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv", "separar contexto de evento observado", "alta", "Limites, bacias e suscetibilidade são contexto."),
        ("cemaden_requer_solicitacao_formal", "fonte_externa", "ativo", "dados não disponíveis por download direto", "outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json", "abrir solicitação formal", "media", "Acesso futuro depende de pedido externo."),
        ("apac_sem_download_estatico", "fonte_externa", "ativo", "série não resolvida em arquivo estático", "outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json", "buscar canal oficial alternativo", "media", "Navegação real bloqueou download direto."),
        ("copernicus_sem_produto_vetorial_publico_petropolis", "fonte_externa", "ativo", "não fecha vetor de evento para Petrópolis", "outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json", "solicitar ou localizar produto vetorial oficial", "media", "Produto público vetorial não foi obtido."),
        ("ana_inventario_sem_serie_estacao_filtrada", "hidrologia", "ativo", "não fecha série por estação para evento", "outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv", "filtrar HidroWeb por estação e período", "media", "Inventário PDF é contexto inicial."),
        ("sgb_pdf_contextual_sem_sig_vetorial_baixado", "geologia", "ativo", "não fecha geometria vetorial de evento", "outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv", "obter camada SIG vetorial quando existir", "media", "PDF de suscetibilidade é contexto."),
        ("risco_landslide_vs_flood", "ontologia_evento", "ativo", "pode misturar fenômenos distintos", "outputs_public/tables/revp_indice_eventos_externos_candidatos_navegacao_mv1.csv", "adjudicar tipologia de evento", "alta", "Landslide e flood ficam separados."),
        ("risco_circularidade", "anti_leakage", "ativo", "pode contaminar fonte de label e feature", "docs/metodologia_cientifica/revp_politica_anti_leakage_mv1.md", "criar matriz label-feature por item", "alta", "Fonte independente é gate obrigatório."),
        ("feature_table_multimodal_nao_liberada", "feature_table", "ativo", "bloqueia engenharia supervisionada", "outputs_public/tables/revp_gates_readiness_treino_mv1.csv", "satisfazer G0-G8 antes de avaliar tabela", "critica", "Tabela antes dos gates criaria risco metodológico."),
        ("treino_supervisionado_bloqueado", "treino", "ativo", "bloqueia baselines supervisionados", "outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json", "manter bloqueio até ground truth real", "critica", "G8 segue bloqueado."),
    ]
    return [dict(zip(COLUNAS_BLOQUEADORES, row)) for row in dados]


def calcular_metricas(branch: str, maturidade: list[dict[str, str]], fila: list[dict[str, str]], bloqueadores: list[dict[str, str]]) -> dict[str, object]:
    return {
        "status_marco_final": DECISAO_FINAL,
        "branch": branch,
        "total_camadas_maturidade": len(maturidade),
        "camadas_consolidadas_para_contexto": sum(1 for row in maturidade if row["nivel_maturidade"] == "consolidado_para_contexto"),
        "camadas_prontas_para_revisao_humana": sum(1 for row in maturidade if row["nivel_maturidade"] == "pronto_para_revisao_humana"),
        "camadas_bloqueadas_para_treino": sum(1 for row in maturidade if row["pode_sustentar_treino_agora"] == "false"),
        "total_itens_fila_revisao_humana": len(fila),
        "itens_com_geometria_contextual": sum(1 for row in fila if row["tipo_item"] == "geometria_contextual" and row["geometria_disponivel"] == "true"),
        "itens_com_evento_candidato": sum(1 for row in fila if row["tipo_item"] == "evento_candidato"),
        "itens_podem_virar_label_agora": sum(1 for row in fila if row["pode_virar_label_agora"] == "true"),
        "itens_podem_apoiar_ground_truth_futuro": sum(1 for row in fila if row["pode_apoiar_ground_truth_futuro"] == "true"),
        "total_bloqueadores_finais": len(bloqueadores),
        "ground_truth_operacional_status": "ausente",
        "treino_supervisionado_status": "bloqueado",
        "decisao_final_mv1": DECISAO_FINAL,
        "proximo_passo_prioritario": PROXIMO_PASSO,
        "guardrails_preservados": GUARDRAILS,
    }


def validar_linguagem_estruturada(linhas: list[list[dict[str, str]]], metricas: dict[str, object]) -> None:
    textos = []
    for grupo in linhas:
        for row in grupo:
            textos.extend(str(valor).lower() for valor in row.values())
    textos.append(json.dumps(metricas, ensure_ascii=False).lower())
    combinado = "\n".join(textos)
    encontrados = [termo for termo in LINGUAGEM_PROIBIDA if termo.lower() in combinado]
    if encontrados:
        raise ValueError(f"Linguagem proibida em campos estruturados: {', '.join(encontrados)}")


def gerar_relatorio(branch: str, encontrados: list[str], ausentes: list[str], metricas: dict[str, object], bloqueadores: list[dict[str, str]]) -> str:
    principais = [row["bloqueador"] for row in bloqueadores if row["prioridade"] in {"critica", "alta"}]
    return "\n".join(
        [
            "# Integração final do marco MV1: maturidade metodológica e revisão humana",
            "",
            "## 1. Escopo do marco final MV1",
            "Este relatório consolida a branch como marco review-only, com maturidade metodológica, fila de revisão humana e bloqueios explícitos para ground truth operacional e treino supervisionado.",
            "",
            "## 2. Linha do tempo da branch",
            "A branch consolidou restauração forense `v2dz-v2ef`, auditoria temporal, validação label-free DINOv2, fechamento do marco, protocolo fail-closed e curadoria externa com navegação/downloads reais.",
            "",
            "## 3. O que a restauração `v2dz-v2ef` resolveu",
            "Resolveu recuperabilidade e trilha forense pública. Não criou labels reais, positivos formais, negativos formais ou ground truth operacional.",
            "",
            "## 4. O que a auditoria temporal resolveu",
            "A auditoria mostrou que a Trilha A depende de metadados temporais e de nuvem que ainda não estão completos no universo auditado.",
            "",
            "## 5. Por que a Trilha A segue bloqueada",
            "A Trilha A segue bloqueada por ausência de múltiplas datas úteis por patch e cobertura de nuvem suficiente.",
            "",
            "## 6. O que a Trilha B conseguiu demonstrar",
            "A Trilha B demonstrou uma leitura estrutural label-free com DINOv2, topologia de embeddings, vizinhos e evidência contextual como probe externo.",
            "",
            "## 7. Limite do `n=12`",
            "`n=12` sustenta piloto exploratório, mas não sustenta evidência estatística final, ground truth operacional ou treino supervisionado.",
            "",
            "## 8. O que a curadoria externa adicionou",
            "A curadoria externa adicionou fontes, eventos candidatos, geometrias contextuais e bloqueios documentados para revisão futura.",
            "",
            "## 9. O que os downloads reais adicionaram",
            "Os downloads reais adicionaram arquivos em quarentena local com SHA256, incluindo GeoJSON, PDF e XLSX. Esses arquivos reforçam rastreabilidade, não liberam label.",
            "",
            "## 10. O que o protocolo fail-closed formalizou",
            "Formalizou ontologia de estados de label futuros, política de evidência negativa, anti-leakage e gates G0-G8.",
            "",
            "## 11. Matriz de maturidade metodológica",
            f"A matriz possui {metricas['total_camadas_maturidade']} camadas. Nenhuma camada sustenta treino agora.",
            "",
            "## 12. Fila de revisão humana futura",
            f"A fila possui {metricas['total_itens_fila_revisao_humana']} itens. Itens que podem virar label agora: {metricas['itens_podem_virar_label_agora']}.",
            "",
            "## 13. Relação entre evidências externas e gates G0-G8",
            "A matriz evidência externa x gates mostra que fontes podem ajudar parcialmente G0-G5 e G7, mas G6 e G8 permanecem bloqueados.",
            "",
            "## 14. Bloqueadores finais para ground truth operacional",
            "Ground truth operacional permanece ausente por falta de revisão humana, overlay patch-evento, positivos formais, negativos formais e anti-leakage por item.",
            "",
            "## 15. Bloqueadores finais para treino supervisionado",
            "Treino supervisionado permanece bloqueado porque G8 depende de todos os gates anteriores, e os principais gates ainda estão bloqueados.",
            "",
            "## 16. Riscos metodológicos remanescentes",
            "- risco de circularidade entre fonte de label e feature",
            "- risco landslide vs flood",
            "- risco de promover geometria contextual a evento observado",
            "- risco de tratar ausência como negativo",
            "- risco de usar `n=12` além do escopo piloto",
            "",
            "## 17. Próximo passo prioritário",
            str(metricas["proximo_passo_prioritario"]),
            "",
            "## 18. Guardrails preservados",
            *[f"- {item}" for item in GUARDRAILS],
            "",
            "## 19. Conclusão executiva",
            f"Decisão final MV1: `{DECISAO_FINAL}`. A branch está consolidada como marco review-only com fila de revisão humana, sem liberar label automaticamente e com treino supervisionado bloqueado.",
            "",
            "## Artefatos de entrada encontrados",
            *[f"- `{item}`" for item in encontrados],
            "",
            "## Artefatos de entrada ausentes",
            *[f"- `{item}`" for item in ausentes],
            "",
            "## Principais bloqueadores finais",
            *[f"- `{item}`" for item in principais],
            "",
            "## Branch",
            f"`{branch}`",
            "",
        ]
    )


def executar(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    branch = obter_branch(repo_root)
    encontrados = artefatos_encontrados(repo_root)
    ausentes = artefatos_ausentes(repo_root)
    maturidade = montar_maturidade(repo_root)
    fila = montar_fila(repo_root)
    evidencias_gates = montar_evidencias_gates(fila)
    bloqueadores = montar_bloqueadores()
    metricas = calcular_metricas(branch, maturidade, fila, bloqueadores)
    validar_linguagem_estruturada([maturidade, fila, evidencias_gates, bloqueadores], metricas)

    escrever_csv(repo_root / CAMINHO_MATURIDADE, maturidade, COLUNAS_MATURIDADE)
    escrever_csv(repo_root / CAMINHO_FILA, fila, COLUNAS_FILA)
    escrever_csv(repo_root / CAMINHO_EVIDENCIAS_GATES, evidencias_gates, COLUNAS_EVIDENCIAS_GATES)
    escrever_csv(repo_root / CAMINHO_BLOQUEADORES, bloqueadores, COLUNAS_BLOQUEADORES)
    escrever_texto(repo_root / CAMINHO_RELATORIO, gerar_relatorio(branch, encontrados, ausentes, metricas, bloqueadores))

    (repo_root / CAMINHO_METRICAS).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / CAMINHO_METRICAS).write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metricas


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    executar(Path(args.repo_root))


if __name__ == "__main__":
    main()
