from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path


CAMINHO_DOC_ONTOLOGIA = Path("docs/metodologia_cientifica/revp_ontologia_labels_ground_truth_mv1.md")
CAMINHO_DOC_NEGATIVOS = Path("docs/metodologia_cientifica/revp_politica_evidencia_negativa_mv1.md")
CAMINHO_DOC_ANTI_LEAKAGE = Path("docs/metodologia_cientifica/revp_politica_anti_leakage_mv1.md")
CAMINHO_ESTADOS = Path("outputs_public/tables/revp_ontologia_estados_label_mv1.csv")
CAMINHO_GATES = Path("outputs_public/tables/revp_gates_readiness_treino_mv1.csv")
CAMINHO_DASHBOARD = Path("outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv")
CAMINHO_RELATORIO = Path("outputs_public/execution_reports/revp_protocolo_ground_truth_fail_closed_mv1.md")
CAMINHO_METRICAS = Path("outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json")

COLUNAS_ESTADOS = [
    "estado_label",
    "descricao",
    "evidencia_minima_exigida",
    "exige_evento_observado",
    "exige_geometria",
    "exige_janela_temporal_fechada",
    "exige_revisao_humana",
    "pode_treinar",
    "risco_metodologico",
    "observacao",
]

COLUNAS_GATES = [
    "gate",
    "descricao",
    "criterio_aprovacao",
    "criterio_reprovacao",
    "evidencia_exigida",
    "status_atual",
    "motivo_status_atual",
    "bloqueia_treino",
    "observacao",
]

COLUNAS_DASHBOARD = [
    "componente",
    "status_atual",
    "pode_liberar_treino",
    "bloqueador_principal",
    "evidencia_atual",
    "acao_para_desbloquear",
    "observacao_metodologica",
]

ESTADOS_LABEL = [
    {
        "estado_label": "positivo_ouro",
        "descricao": "Estado futuro para patch com evidência independente forte de inundação, geometria compatível, janela temporal fechada e revisão humana completa.",
        "evidencia_minima_exigida": "Evento observado independente, fonte de label independente, geometria compatível entre evento e patch, janela temporal fechada e revisão humana/adjudicação.",
        "exige_evento_observado": "true",
        "exige_geometria": "true",
        "exige_janela_temporal_fechada": "true",
        "exige_revisao_humana": "true",
        "pode_treinar": "true",
        "risco_metodologico": "Circularidade se a fonte de label usar o mesmo sinal que a feature; promoção indevida se faltar revisão humana.",
        "observacao": "Só pode entrar em treino futuro quando todos os gates estiverem satisfeitos; hoje não há amostras nesse estado.",
    },
    {
        "estado_label": "positivo_prata",
        "descricao": "Estado futuro para candidato positivo com evidência observacional parcial ou fonte independente incompleta.",
        "evidencia_minima_exigida": "Evento observado ou documental forte, mas com alguma lacuna em geometria, janela temporal ou adjudicação.",
        "exige_evento_observado": "true",
        "exige_geometria": "true",
        "exige_janela_temporal_fechada": "true",
        "exige_revisao_humana": "true",
        "pode_treinar": "false",
        "risco_metodologico": "Uso indevido no treino principal antes de fechar evidência formal.",
        "observacao": "Não pode entrar no treino principal; só pode aparecer em ablação futura separada e explicitamente marcada.",
    },
    {
        "estado_label": "negativo_pareado",
        "descricao": "Estado futuro para patch pareado com um positivo e com evidência explícita de não inundação na janela controlada.",
        "evidencia_minima_exigida": "Evidência explícita de não inundação, fonte independente, mesma política temporal do positivo e revisão humana.",
        "exige_evento_observado": "false",
        "exige_geometria": "true",
        "exige_janela_temporal_fechada": "true",
        "exige_revisao_humana": "true",
        "pode_treinar": "true",
        "risco_metodologico": "Ausência de registro pode ser confundida com evidência negativa formal.",
        "observacao": "Só pode existir com evidência explícita de não inundação; ausência de evidência não satisfaz o estado.",
    },
    {
        "estado_label": "negativo_dificil",
        "descricao": "Estado futuro para patch visual ou territorialmente similar ao positivo, mas com evidência explícita de não inundação e controle espacial/temporal.",
        "evidencia_minima_exigida": "Evidência explícita de não inundação, pareamento espacial/temporal controlado, revisão humana e aprovação anti-leakage.",
        "exige_evento_observado": "false",
        "exige_geometria": "true",
        "exige_janela_temporal_fechada": "true",
        "exige_revisao_humana": "true",
        "pode_treinar": "true",
        "risco_metodologico": "Construção de contraste artificial ou vazamento de informação contextual.",
        "observacao": "Curitiba nunca vira negativo formal por default; o estado exige evidência negativa explícita.",
    },
    {
        "estado_label": "desconhecido",
        "descricao": "Estado para patch sem evidência suficiente para positivo, negativo ou exclusão metodológica.",
        "evidencia_minima_exigida": "Registro da lacuna e preservação do estado sem inferência.",
        "exige_evento_observado": "false",
        "exige_geometria": "false",
        "exige_janela_temporal_fechada": "false",
        "exige_revisao_humana": "false",
        "pode_treinar": "false",
        "risco_metodologico": "Converter incerteza em negativo por conveniência experimental.",
        "observacao": "Desconhecido nunca pode treinar e nunca vira negativo.",
    },
    {
        "estado_label": "excluido",
        "descricao": "Estado para patch removido do conjunto por problema de qualidade, escopo, duplicidade, licença ou inconsistência metodológica.",
        "evidencia_minima_exigida": "Motivo de exclusão documentado e rastreável.",
        "exige_evento_observado": "false",
        "exige_geometria": "false",
        "exige_janela_temporal_fechada": "false",
        "exige_revisao_humana": "false",
        "pode_treinar": "false",
        "risco_metodologico": "Reintrodução acidental em feature table ou treino.",
        "observacao": "Excluído nunca pode treinar.",
    },
    {
        "estado_label": "bloqueado",
        "descricao": "Estado para patch com evidência insuficiente, conflito de proveniência, risco de circularidade ou gate obrigatório não satisfeito.",
        "evidencia_minima_exigida": "Bloqueador explícito e ação de desbloqueio documentada.",
        "exige_evento_observado": "false",
        "exige_geometria": "false",
        "exige_janela_temporal_fechada": "false",
        "exige_revisao_humana": "false",
        "pode_treinar": "false",
        "risco_metodologico": "Forçar uso experimental apesar de gate não satisfeito.",
        "observacao": "Bloqueado nunca pode treinar.",
    },
]

GATES = [
    {
        "gate": "G0_patch_id_valido",
        "descricao": "Identificador canônico de patch existe, é único e tem linhagem rastreável.",
        "criterio_aprovacao": "Patch possui ID canônico único, sem duplicidade e com origem documentada.",
        "criterio_reprovacao": "ID ausente, duplicado, conflitante ou sem origem verificável.",
        "evidencia_exigida": "Registro público de patch e manifesto de linhagem.",
        "status_atual": "parcial",
        "motivo_status_atual": "Há corpus territorial/contextual e embeddings rastreáveis, mas este protocolo não reclassifica todos os patches como amostras de treino.",
        "bloqueia_treino": "true",
        "observacao": "Gate é por amostra; nenhum conjunto supervisionado está liberado.",
    },
    {
        "gate": "G1_asset_id_valido",
        "descricao": "Asset associado ao patch possui ID, origem e hash ou referência equivalente.",
        "criterio_aprovacao": "Asset rastreável por ID, origem, versão e integridade documentada.",
        "criterio_reprovacao": "Asset sem ID, sem origem, sem integridade ou com conflito de versão.",
        "evidencia_exigida": "Manifesto de asset, hash SHA256 quando aplicável e vínculo com patch.",
        "status_atual": "parcial",
        "motivo_status_atual": "Há assets auditados, mas metadados temporais e de nuvem permanecem insuficientes para readiness de treino.",
        "bloqueia_treino": "true",
        "observacao": "Asset rastreável não equivale a label operacional.",
    },
    {
        "gate": "G2_janela_temporal_fechada",
        "descricao": "Janela temporal entre evento observado e asset Sentinel está fechada.",
        "criterio_aprovacao": "Evento e asset possuem datas independentes suficientes, com janela temporal registrada e sem ambiguidade crítica.",
        "criterio_reprovacao": "Data ausente, data única insuficiente, conflito temporal ou cobertura temporal incompleta.",
        "evidencia_exigida": "Datas de evento, datas Sentinel, critério de janela e auditoria temporal.",
        "status_atual": "bloqueado",
        "motivo_status_atual": "A auditoria temporal MV1 registrou metadados temporais insuficientes e zero patches elegíveis para deslocamento temporal.",
        "bloqueia_treino": "true",
        "observacao": "Sem janela temporal fechada não há amostra supervisionada válida.",
    },
    {
        "gate": "G3_geometria_espacial_fechada",
        "descricao": "Geometria do evento e geometria do patch são compatíveis e auditáveis.",
        "criterio_aprovacao": "Evento observado com geometria vetorial, CRS conhecido, patch boundary e interseção auditável.",
        "criterio_reprovacao": "Geometria ausente, CRS ausente, patch boundary ausente ou interseção não auditável.",
        "evidencia_exigida": "GeoJSON, CSV WKT com CRS explícito ou outro vetor auditável, além de patch boundary.",
        "status_atual": "bloqueado",
        "motivo_status_atual": "A base restaurada permanece como trilha forense e não fecha geometria operacional patch-evento.",
        "bloqueia_treino": "true",
        "observacao": "Imagem, PDF, texto ou coordenada inferida não fecham geometria por si só.",
    },
    {
        "gate": "G4_fonte_label_independente",
        "descricao": "Fonte de label é independente das features usadas no modelo.",
        "criterio_aprovacao": "Fonte observacional ou humana independente, sem derivar do embedding, asset ou feature usada no treino.",
        "criterio_reprovacao": "Label derivado de feature, score contextual, embedding, proxy circular ou ausência de evidência.",
        "evidencia_exigida": "Documento de fonte independente, trilha de decisão e justificativa anti-leakage.",
        "status_atual": "bloqueado",
        "motivo_status_atual": "Não há label operacional nem fonte independente fechada para treino.",
        "bloqueia_treino": "true",
        "observacao": "Evidência contextual pode priorizar revisão, mas não vira label.",
    },
    {
        "gate": "G5_revisao_humana_completa",
        "descricao": "Revisão humana e adjudicação estão completas e travadas.",
        "criterio_aprovacao": "Decisão primária, decisão secundária quando necessária e adjudicação final registradas.",
        "criterio_reprovacao": "Revisão ausente, divergente, incompleta ou não travada.",
        "evidencia_exigida": "Fila de revisão, decisões humanas, adjudicação e registro de bloqueios.",
        "status_atual": "bloqueado",
        "motivo_status_atual": "A restauração v2dz-v2ef preservou decisões humanas vazias.",
        "bloqueia_treino": "true",
        "observacao": "Sem revisão humana não há promoção para treino.",
    },
    {
        "gate": "G6_politica_negativo_satisfeita",
        "descricao": "Política de evidência negativa está satisfeita para qualquer negativo formal futuro.",
        "criterio_aprovacao": "Evidência explícita de não inundação, janela e geometria controladas, revisão humana e fonte independente.",
        "criterio_reprovacao": "Ausência de evento, unknown, contraste regional ou Curitiba por default.",
        "evidencia_exigida": "Fonte formal de não inundação ou adjudicação documentada para a janela e área do patch.",
        "status_atual": "bloqueado",
        "motivo_status_atual": "Não há negativos formais rastreáveis e aprovados.",
        "bloqueia_treino": "true",
        "observacao": "Ausência de evidência nunca vira negativo.",
    },
    {
        "gate": "G7_anti_leakage_aprovado",
        "descricao": "Separação entre fonte de label, fonte de feature e fonte contextual foi aprovada.",
        "criterio_aprovacao": "Auditoria anti-leakage confirma independência, sem circularidade e sem uso de proxies proibidos.",
        "criterio_reprovacao": "Feature deriva do label, label deriva da feature, fonte contextual é usada como alvo ou há sobreposição indevida.",
        "evidencia_exigida": "Matriz label-feature, declaração de fontes independentes e revisão metodológica.",
        "status_atual": "bloqueado",
        "motivo_status_atual": "A política foi formalizada neste pacote, mas nenhuma amostra de treino foi aprovada contra ela.",
        "bloqueia_treino": "true",
        "observacao": "Anti-leakage é gate obrigatório antes de qualquer baseline supervisionado.",
    },
    {
        "gate": "G8_liberado_para_treino_true",
        "descricao": "Gate final que só pode ser verdadeiro quando todos os gates anteriores estiverem satisfeitos.",
        "criterio_aprovacao": "G0 a G7 satisfeitos para a amostra e conjunto supervisionado revisado.",
        "criterio_reprovacao": "Qualquer gate anterior parcial, bloqueado ou ausente.",
        "evidencia_exigida": "Dashboard de gates sem bloqueios e registro final de liberação metodológica.",
        "status_atual": "bloqueado",
        "motivo_status_atual": "Há gates parciais e bloqueados; treino supervisionado permanece bloqueado.",
        "bloqueia_treino": "true",
        "observacao": "No estado atual este gate deve permanecer false.",
    },
]

DASHBOARD = [
    {
        "componente": "ground_truth_patch_level",
        "status_atual": "ausente",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "ground truth operacional ausente",
        "evidencia_atual": "Restauração v2dz-v2ef é trilha forense e não fecha estado operacional.",
        "acao_para_desbloquear": "Fechar evento, geometria, janela temporal, revisão humana e anti-leakage por patch.",
        "observacao_metodologica": "Estado atual é candidato para revisão, não amostra de treino.",
    },
    {
        "componente": "positivos_formais",
        "status_atual": "ausente",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "sem evento observado independente fechado por patch",
        "evidencia_atual": "Não há positivo ouro aprovado.",
        "acao_para_desbloquear": "Registrar fonte independente, geometria compatível, janela temporal e adjudicação.",
        "observacao_metodologica": "Positivo prata não entra no treino principal.",
    },
    {
        "componente": "negativos_formais",
        "status_atual": "ausente",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "sem evidência explícita de não inundação",
        "evidencia_atual": "Não há negativo pareado ou negativo difícil aprovado.",
        "acao_para_desbloquear": "Formalizar evidência negativa explícita e revisão humana.",
        "observacao_metodologica": "Curitiba e unknown não viram negativo por default.",
    },
    {
        "componente": "geometria_evento_patch",
        "status_atual": "bloqueado",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "geometria operacional não fechada",
        "evidencia_atual": "Trilha forense mantém lacunas de geometria e CRS.",
        "acao_para_desbloquear": "Validar vetor de evento, CRS, patch boundary e interseção.",
        "observacao_metodologica": "Artefatos visuais ou textuais não bastam como geometria.",
    },
    {
        "componente": "janela_temporal_evento_patch",
        "status_atual": "bloqueado",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "metadados temporais insuficientes",
        "evidencia_atual": "Auditoria temporal MV1 bloqueou Trilha A.",
        "acao_para_desbloquear": "Registrar múltiplas datas Sentinel e cobertura de nuvem por patch.",
        "observacao_metodologica": "Sem janela temporal fechada não há readiness de treino.",
    },
    {
        "componente": "revisao_humana",
        "status_atual": "bloqueado",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "decisões humanas vazias",
        "evidencia_atual": "A restauração preservou revisão como pendente.",
        "acao_para_desbloquear": "Executar revisão primária, secundária quando necessária e adjudicação.",
        "observacao_metodologica": "Revisão humana é gate obrigatório.",
    },
    {
        "componente": "anti_leakage",
        "status_atual": "em_preparacao",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "nenhuma amostra aprovada contra a política",
        "evidencia_atual": "Política formalizada neste protocolo.",
        "acao_para_desbloquear": "Aplicar matriz label-feature em cada amostra candidata.",
        "observacao_metodologica": "Fonte de label e fonte de feature devem ser independentes.",
    },
    {
        "componente": "feature_table_multimodal",
        "status_atual": "bloqueado",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "gates de label e anti-leakage não satisfeitos",
        "evidencia_atual": "Marco label-free existe, mas não gera alvo supervisionado.",
        "acao_para_desbloquear": "Fechar G0 a G8 antes de construir tabela de treino.",
        "observacao_metodologica": "Tabela multimodal antes dos gates criaria risco de circularidade.",
    },
    {
        "componente": "baselines_supervisionados",
        "status_atual": "bloqueado",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "G8 bloqueado",
        "evidencia_atual": "Nenhum treino supervisionado liberado no estado atual.",
        "acao_para_desbloquear": "Satisfazer todos os gates e revisar política de negativos.",
        "observacao_metodologica": "Baseline supervisionado só pode ser avaliado depois de ground truth real.",
    },
    {
        "componente": "DINOv2_label_free",
        "status_atual": "liberado_com_restricoes",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "uso restrito a análise label-free",
        "evidencia_atual": "12 embeddings válidos no piloto MV1.",
        "acao_para_desbloquear": "Não aplicável para treino; expandir apenas análise estrutural.",
        "observacao_metodologica": "DINOv2 não prova inundação e não cria label.",
    },
    {
        "componente": "evidencias_externas_contextuais",
        "status_atual": "parcial",
        "pode_liberar_treino": "false",
        "bloqueador_principal": "contexto não é fonte de label",
        "evidencia_atual": "Pode apoiar priorização de revisão humana.",
        "acao_para_desbloquear": "Separar contexto de fonte independente de label e documentar anti-leakage.",
        "observacao_metodologica": "Evidência contextual é probe externo, não alvo supervisionado.",
    },
]

GUARDRAILS = [
    "unknown nunca vira negativo",
    "ausência de evento nunca vira negativo formal",
    "Curitiba nunca vira negativo formal por default",
    "evidência contextual nunca vira label",
    "DINOv2 não prova inundação",
    "restauração v2dz-v2ef não vira ground truth operacional",
    "auditoria temporal não libera Trilha A",
    "piloto label-free com n=12 não vira evidência estatística final",
    "nenhum treino supervisionado é liberado",
    "fonte de label e fonte de feature devem ser independentes",
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


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Gera protocolo fail-closed de ground truth futuro MV1.")
    parser.add_argument("--repo-root", default=".", help="Raiz do repositório REV-P.")
    return parser


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


def markdown_ontologia() -> str:
    linhas = [
        "# Ontologia de estados de label e ground truth MV1",
        "",
        "## Escopo",
        "Este documento define estados futuros de label para o REV-P em modo fail-closed. Ele não cria labels reais, positivos formais reais, negativos formais reais ou ground truth operacional.",
        "",
        "## Estados",
    ]
    for estado in ESTADOS_LABEL:
        linhas.extend(
            [
                "",
                f"### `{estado['estado_label']}`",
                estado["descricao"],
                "",
                f"- Evidência mínima exigida: {estado['evidencia_minima_exigida']}",
                f"- Exige evento observado: `{estado['exige_evento_observado']}`",
                f"- Exige geometria: `{estado['exige_geometria']}`",
                f"- Exige janela temporal fechada: `{estado['exige_janela_temporal_fechada']}`",
                f"- Exige revisão humana: `{estado['exige_revisao_humana']}`",
                f"- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `{estado['pode_treinar']}`",
                f"- Observação: {estado['observacao']}",
            ]
        )
    linhas.extend(
        [
            "",
            "## Regra fail-closed",
            "`desconhecido`, `bloqueado` e `excluido` nunca podem treinar. `positivo_prata` não entra no treino principal. `positivo_ouro`, `negativo_pareado` e `negativo_dificil` só são estados futuros treináveis quando todos os gates forem satisfeitos por amostra.",
            "",
        ]
    )
    return "\n".join(linhas)


def markdown_negativos() -> str:
    return "\n".join(
        [
            "# Política de evidência negativa MV1",
            "",
            "## Princípio",
            "Negativo formal só pode existir com evidência explícita de não inundação. Ausência de registro, ausência de evento, cidade de contraste, unknown ou lacuna documental não são evidência negativa.",
            "",
            "## Requisitos para `negativo_pareado`",
            "- Fonte independente que registre não inundação na janela controlada.",
            "- Patch e asset com identificadores rastreáveis.",
            "- Relação temporal compatível com o positivo pareado.",
            "- Revisão humana completa.",
            "- Auditoria anti-leakage aprovada.",
            "",
            "## Requisitos para `negativo_dificil`",
            "- Evidência explícita de não inundação.",
            "- Similaridade espacial, temporal ou visual controlada sem uso circular da feature como label.",
            "- Geometria e janela temporal auditáveis.",
            "- Revisão humana e adjudicação quando houver dúvida.",
            "",
            "## Regras proibitivas",
            "- `unknown` nunca vira negativo.",
            "- Ausência de evento nunca vira negativo formal.",
            "- Curitiba nunca vira negativo formal por default.",
            "- Evidência contextual nunca vira label.",
            "- O contraste entre cidades serve para análise label-free e priorização de revisão humana, não para criar negativo formal.",
            "",
        ]
    )


def markdown_anti_leakage() -> str:
    return "\n".join(
        [
            "# Política anti-leakage MV1",
            "",
            "## Separação de fontes",
            "A fonte de label deve ser independente da fonte de feature. Um embedding DINOv2, um asset Sentinel, uma métrica de similaridade, uma evidência contextual ou uma variável administrativa não pode ser usada simultaneamente como feature e como fonte de decisão supervisionada.",
            "",
            "## Fonte de label",
            "Fonte de label futura deve vir de evento observado independente, revisão humana/adjudicação e documentação rastreável. A fonte precisa ser auditable e não derivada do mesmo sinal usado no modelo.",
            "",
            "## Fonte de feature",
            "Fonte de feature pode incluir asset Sentinel, embedding DINOv2 congelado e metadados controlados, desde que não carregue o alvo ou proxy direto do alvo.",
            "",
            "## Fonte contextual",
            "Evidência contextual pode ser usada como probe externo ou priorização de revisão humana. Ela não deve preencher label, negativo formal ou decisão de treino.",
            "",
            "## Checagens obrigatórias",
            "- Verificar se a fonte de label aparece como feature.",
            "- Verificar se a feature foi usada para selecionar o estado de label.",
            "- Verificar se evidência contextual foi promovida a alvo supervisionado.",
            "- Verificar se cidade, ausência ou unknown foram usados como negativo formal.",
            "",
        ]
    )


def markdown_relatorio(branch: str, metricas: dict[str, object]) -> str:
    gates_bloqueados = ", ".join(metricas["gates_bloqueados_atualmente"])
    return "\n".join(
        [
            "# Protocolo ground truth fail-closed MV1",
            "",
            "## 1. Escopo",
            "Este pacote formaliza uma camada pública, auditável e fail-closed para definir quando um patch poderá, no futuro, virar amostra de treino. Ele não treina modelo, não cria labels reais, não cria positivos formais reais, não cria negativos formais reais e não promove ground truth operacional.",
            "",
            "## 2. Por que este protocolo é necessário",
            "O marco MV1 demonstrou uma infraestrutura auditável e um piloto label-free, mas também preservou bloqueios centrais. Sem um protocolo explícito, há risco de transformar evidência contextual, topologia DINOv2, ausência de evento ou contraste regional em alvo supervisionado.",
            "",
            "## 3. Relação com o marco label-free MV1",
            "A validação label-free MV1 permanece exploratória. Os embeddings DINOv2 congelados servem para topologia estrutural, vizinhança e priorização de revisão humana. Eles não provam inundação e não criam label.",
            "",
            "## 4. Relação com a curadoria externa em andamento",
            "A curadoria externa pode produzir insumos futuros, mas este pacote não depende de downloads externos, não altera artefatos de navegação e não altera quarentena externa. Qualquer evidência externa futura deve passar pelos gates antes de uso supervisionado.",
            "",
            "## 5. Ontologia de estados de label",
            "A ontologia pública define `positivo_ouro`, `positivo_prata`, `negativo_pareado`, `negativo_dificil`, `desconhecido`, `excluido` e `bloqueado`. Apenas estados futuros com evidência formal e todos os gates satisfeitos podem ser considerados para treino; no estado atual, nenhum treino está liberado.",
            "",
            "## 6. Política de evidência negativa",
            "Negativo formal exige evidência explícita de não inundação. Ausência de evidência, ausência de evento, unknown e Curitiba por default não satisfazem essa política.",
            "",
            "## 7. Separação entre fonte de label e fonte de feature",
            "Fonte de label deve ser independente de asset, embedding, feature, score contextual e métrica usada pelo modelo. Evidência contextual pode orientar revisão humana, mas não preencher alvo supervisionado.",
            "",
            "## 8. Gates de readiness para treino",
            "Os gates G0 a G8 controlam identidade de patch, identidade de asset, janela temporal, geometria, fonte de label independente, revisão humana, política de negativos, anti-leakage e liberação final.",
            "",
            "## 9. Estado atual dos gates",
            f"Gates bloqueados ou não satisfeitos: {gates_bloqueados}. Nenhum gate final libera treino supervisionado.",
            "",
            "## 10. Por que treino ainda está bloqueado",
            "Treino supervisionado permanece bloqueado porque ground truth operacional está ausente, positivos formais estão ausentes, negativos formais estão ausentes, revisão humana não está completa, geometria e janela temporal não estão fechadas e anti-leakage ainda não foi aprovado por amostra.",
            "",
            "## 11. Como evidências externas poderão ser usadas futuramente",
            "Evidências externas poderão ser usadas como fonte independente ou como contexto de revisão apenas se sua origem, data, geometria, licença, hash/proveniência e independência em relação às features forem auditáveis.",
            "",
            "## 12. Como evitar circularidade",
            "A matriz label-feature deve registrar, para cada amostra, se a fonte de label aparece como feature. Se houver sobreposição, a amostra deve ficar bloqueada.",
            "",
            "## 13. Como evitar `unknown = negative`",
            "`unknown` deve permanecer desconhecido. Lacuna documental, ausência de registro e ausência de evento não criam negativo formal.",
            "",
            "## 14. Como evitar Curitiba como negativo por default",
            "Curitiba pode ser usada como contraste estrutural label-free, mas não vira negativo formal por localização. Qualquer negativo futuro exige evidência explícita de não inundação.",
            "",
            "## 15. Como tratar landslide vs flood",
            "Eventos de movimento de massa e inundação devem permanecer separados na ontologia. Um evento landslide não deve ser usado como positivo de flood, nem como negativo automático de flood, sem política explícita e revisão humana.",
            "",
            "## 16. Próximos passos para ground truth real",
            "Expandir evidências independentes, fechar geometria e janela temporal, formalizar revisão humana/adjudicação, aplicar política de negativos, executar auditoria anti-leakage por amostra e só depois avaliar feature table multimodal.",
            "",
            "## 17. Guardrails preservados",
            *[f"- {item}" for item in GUARDRAILS],
            "",
            "## Status consolidado",
            f"- Branch: `{branch}`",
            f"- Status do protocolo: `{metricas['status_protocolo']}`",
            f"- Treino supervisionado: `{metricas['treino_supervisionado_status']}`",
            f"- Ground truth operacional: `{metricas['ground_truth_operacional_status']}`",
            f"- Positivos formais: `{metricas['positivos_formais_status']}`",
            f"- Negativos formais: `{metricas['negativos_formais_status']}`",
            "",
        ]
    )


def calcular_metricas() -> dict[str, object]:
    estados_treinaveis = [row["estado_label"] for row in ESTADOS_LABEL if row["pode_treinar"] == "true"]
    estados_bloqueados = [row["estado_label"] for row in ESTADOS_LABEL if row["pode_treinar"] == "false"]
    gates_satisfeitos = [row["gate"] for row in GATES if row["status_atual"] == "satisfeito"]
    gates_bloqueados = [row["gate"] for row in GATES if row["status_atual"] != "satisfeito"]
    return {
        "status_protocolo": "PROTOCOLO_FAIL_CLOSED_FORMALIZADO_SEM_LIBERAR_TREINO",
        "total_estados_label": len(ESTADOS_LABEL),
        "estados_que_podem_treinar": estados_treinaveis,
        "estados_bloqueados_para_treino": estados_bloqueados,
        "total_gates": len(GATES),
        "gates_satisfeitos_atualmente": gates_satisfeitos,
        "gates_bloqueados_atualmente": gates_bloqueados,
        "treino_supervisionado_status": "bloqueado",
        "ground_truth_operacional_status": "ausente",
        "negativos_formais_status": "ausente",
        "positivos_formais_status": "ausente",
        "revisao_humana_status": "bloqueado",
        "anti_leakage_status": "em_preparacao",
        "guardrails_preservados": GUARDRAILS,
        "proximo_passo_recomendado": "Aplicar este protocolo a cada candidato futuro, fechar fontes independentes, geometria, janela temporal, revisão humana, política de negativos e anti-leakage antes de qualquer avaliação supervisionada.",
    }


def validar_linguagem_estruturada(metricas: dict[str, object]) -> None:
    textos = []
    for colecao in (ESTADOS_LABEL, GATES, DASHBOARD):
        for row in colecao:
            textos.extend(str(valor).lower() for valor in row.values())
    textos.append(json.dumps(metricas, ensure_ascii=False).lower())
    combinado = "\n".join(textos)
    encontrados = [termo for termo in LINGUAGEM_PROIBIDA if termo.lower() in combinado]
    if encontrados:
        raise ValueError(f"Linguagem proibida em campos estruturados: {', '.join(encontrados)}")


def executar(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    branch = obter_branch(repo_root)
    metricas = calcular_metricas()
    validar_linguagem_estruturada(metricas)

    escrever_csv(repo_root / CAMINHO_ESTADOS, ESTADOS_LABEL, COLUNAS_ESTADOS)
    escrever_csv(repo_root / CAMINHO_GATES, GATES, COLUNAS_GATES)
    escrever_csv(repo_root / CAMINHO_DASHBOARD, DASHBOARD, COLUNAS_DASHBOARD)

    escrever_texto(repo_root / CAMINHO_DOC_ONTOLOGIA, markdown_ontologia())
    escrever_texto(repo_root / CAMINHO_DOC_NEGATIVOS, markdown_negativos())
    escrever_texto(repo_root / CAMINHO_DOC_ANTI_LEAKAGE, markdown_anti_leakage())
    escrever_texto(repo_root / CAMINHO_RELATORIO, markdown_relatorio(branch, metricas))

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
