# REV-P — Auditoria editorial do repositório público

Gerado: 2026-06-18
Propósito: mapear, antes de qualquer alteração, os problemas de linguagem, nomenclatura e tom nos arquivos públicos do repositório REV-P.

---

## 1. Critérios de classificação

Cada item recebe uma das seguintes classificações:

| Código | Significado |
|---|---|
| `KEEP_INTERNAL_NAME` | Nome interno necessário por compatibilidade; conteúdo pode ser editado |
| `REWRITE_TEXT_ONLY` | Nome permanece; texto interno deve ser humanizado |
| `ADD_PUBLIC_ALIAS` | Arquivo recebe nome público legível sem renomeação do original |
| `RENAME_SAFE` | Renomear é seguro (não há imports, testes ou orquestradores dependentes) |
| `DO_NOT_TOUCH` | Nenhuma alteração — dado bruto, formato técnico ou contrato de código |
| `ARCHIVE_LATER` | Conteúdo redundante; candidato a arquivamento futuro |
| `NEEDS_HUMAN_REVIEW` | Decisão requer avaliação humana antes de qualquer mudança |

---

## 2. Arquivos com linguagem robótica — outputs_public/execution_reports/

### 2.1 `final_delivery_artifact_index.md`

**Problema**: a coluna "Observacao metodologica" repete literalmente a mesma frase em 92 linhas:
> "Resultado estrutural destinado a revisao. Nao constitui ground truth operacional, confirmacao de evento observado, classe, label, predicao ou treinamento supervisionado."

Isso não acrescenta informação — qualquer leitor entende após a primeira ocorrência. É o padrão mais mecânico do repositório.

**Classificação**: `REWRITE_TEXT_ONLY`
**Ação**: mover o disclaimer para um parágrafo de cabeçalho único; remover da coluna de cada linha; simplificar a tabela.

---

### 2.2 `final_guardrails_report.md`

**Problema**: usa `C4_BLOCKED_NO_FORMAL_NEGATIVES` sem explicação em português. Leitores externos não sabem o que significa C4.

**Classificação**: `REWRITE_TEXT_ONLY`
**Ação**: manter o código técnico, mas adicionar frase explicativa em português.

---

### 2.3 `revp_curitiba_v2ca_v2cg_release_report_v2ch.md`

**Problema**: seção "Prontidão metodológica" lista statuses em inglês maiúsculo (`PRESENT_2`, `ABSENT`, `BLOCKED`, `COMPLETED`, `AVAILABLE_BUT_BLOCKED`) sem contexto em português. Parece output de máquina.

**Classificação**: `REWRITE_TEXT_ONLY`
**Ação**: manter os valores de status (são contratos técnicos), mas adicionar parágrafo explicativo em português antes da lista.

---

### 2.4 `revp_tp2_candidate_inventory_report_v2ci.md`

**Problema**: listas de "travas metodologicas" com `PASS (Nenhum label formal criado.)` em cada linha parecem checklists de máquina.

**Classificação**: `REWRITE_TEXT_ONLY`
**Ação**: condensar em parágrafo narrativo único.

---

### 2.5 `revp_curitiba_v2ca_v2cg_commit_checklist_v2ch.md`

**Problema**: título em inglês (`commit hygiene checklist`), itens em inglês sem tradução.

**Classificação**: `REWRITE_TEXT_ONLY`
**Ação**: traduzir título e adicionar parágrafo de contexto em português.

---

### 2.6 `revp_v2cn_to_v2cr_integrated_report.md`, `revp_v2cs_to_v2cw_integrated_report.md`, `revp_v2cj_to_v2cm_integrated_report.md`

**Problema**: relatórios integrados curtos e mecânicos; seções de execução listam apenas `PASS (executado)` sem contexto.

**Classificação**: `REWRITE_TEXT_ONLY` — nível baixo de prioridade.

---

### 2.7 `revp_tp2_candidate_commit_checklist_v2ci.md`

**Problema**: checklists com campos em inglês como `training_ready=BLOCKED`.

**Classificação**: `KEEP_INTERNAL_NAME` (nome seguro para compatibilidade de referência)
**Ação**: adicionar parágrafo de contexto em português antes dos itens.

---

### 2.8 `revp_v2cn_to_v2cr_commit_checklist.md`, `revp_v2cs_to_v2cw_commit_checklist.md`, `revp_v2cj_to_v2cm_commit_checklist.md`

**Problema**: checklists de commit em português misturado com inglês.

**Classificação**: `KEEP_INTERNAL_NAME`
**Ação**: baixa prioridade; manter como registro técnico interno.

---

### 2.9 `v2at_*.md` a `v2bh_*.md` (relatórios de estágios individuais)

**Problema**: títulos genéricos como "v2aw geometry source intake report" sem nome humano.

**Classificação**: `ADD_PUBLIC_ALIAS`
**Ação**: criar índice público de estágios com nomes legíveis em `revp_stage_index.md`.

---

### 2.10 `protocol_c_cross_region_reapplication_report.md`

**Problema**: tabelas de gates com status em inglês maiúsculo (`PASS_PUBLIC_PROVENANCE_RECORDED`, `NOT_CREATED_BLOCKED_FOR_TRAINING`) sem explicação contextual.

**Classificação**: `REWRITE_TEXT_ONLY`
**Ação**: o conteúdo já está bem escrito em português nas seções narrativas; os valores das tabelas são contratos técnicos e devem ser mantidos; adicionar nota de rodapé explicativa.

---

## 3. Arquivos com linguagem robótica — outputs_public/logs_summary/

### 3.1 `protocol_c_current_status_summary.md`

**Problema**: mistura inglês e português na mesma frase; usa código `C7 = NOT_CREATED_BLOCKED_FOR_TRAINING` sem contexto; `Operational label = 0 | negative = 0 | training = 0` parece linha de log.

**Classificação**: `REWRITE_TEXT_ONLY`

---

### 3.2 `protocol_c_cross_region_status_summary.md`

**Problema**: mesmo padrão — status em inglês maiúsculo (`PROTOCOL_VALIDATED_CANDIDATE_REFERENCE`, `NOT_CREATED_BLOCKED_FOR_TRAINING`) sem tradução.

**Classificação**: `REWRITE_TEXT_ONLY`

---

## 4. README.md principal

**Problema**: o README está bem escrito em português técnico, mas:
- é muito longo (480 linhas, 16 seções) para um arquivo de entrada pública;
- seções 12–14 têm listas defensivas mecânicas ("Interpretação não permitida: detecção operacional de inundação; predição de enchente; ...") que parecem checklists de guardrail;
- a estrutura não segue o formato de README científico de projeto de pesquisa.

**Classificação**: `REWRITE_TEXT_ONLY`
**Ação**: reestruturar no formato compacto solicitado; preservar todos os fatos; remover listas mecânicas e substituir por parágrafos narrativos.

---

## 5. outputs_public/README.md

**Estado**: bom. Linguagem concisa. Sem acento (arquivo escrito sem acentos por escolha). Estrutura clara.

**Classificação**: `REWRITE_TEXT_ONLY` (ajustes mínimos)

---

## 6. datasets/README.md

**Problema**: arquivo muito longo (240 linhas), com ~80 entradas de tabela descrevendo cada CSV individual. As entradas em si são necessárias como referência técnica. Há problemas de exibição de encoding na renderização Git (caracteres Ã, â€").

**Classificação**: `NEEDS_HUMAN_REVIEW`
**Ação**: verificar encoding do arquivo; se for UTF-8 válido, nenhum problema real. Não alterar conteúdo técnico das tabelas.

---

## 7. Nomes de scripts/testes (scripts/, tests/)

**Problema**: nomes como `test_revp_v1fw_dino_embedding_extraction_scaffold.py` contêm termos como `scaffold`. Esses nomes são internos e aparecem em imports Python.

**Classificação**: `DO_NOT_TOUCH` (renomear quebraria a suíte de testes)
**Ação**: criar camada de nomes públicos no `revp_stage_index.md`.

---

## 8. docs/metodologia_cientifica/ — 50+ arquivos .md

**Estado**: documentação técnica densa e especializada. A maioria usa linguagem portuguesa técnica adequada.

**Problema mais comum**: títulos com nomes de código (`protocolo_c_adjudicacao_negativo_feature_level_v1mo.md`) sem nome público.

**Classificação**: `ADD_PUBLIC_ALIAS` para a maioria; `DO_NOT_TOUCH` para o conteúdo dos arquivos (são documentos metodológicos detalhados, não relatórios públicos).

---

## 9. Termos excessivamente repetidos — ocorrências estimadas nos docs públicos

| Termo | Ocorrências estimadas | Tipo de uso | Recomendação |
|---|---|---|---|
| `scaffold` | ~15–20 | Nome de script, texto de relatório | Preservar em código; humanizar em texto |
| `guardrail` | ~30–40 | Texto de relatório e checklists | Substituir por "restrição metodológica" em texto |
| `readiness` | ~50–60 | Nome de script e texto | Preservar em código; traduzir em texto |
| `BLOCKED` | ~60–80 | Status técnico | Preservar como status; explicar em português |
| `candidate-only` | ~10 | Texto | Substituir por "somente candidato" em texto |
| `training-ready` | ~10 | Status técnico | Preservar como campo; explicar em texto |
| `pipeline` | ~80–100 | Texto genérico | Reduzir em documentação pública; manter em código |
| `ground-truth-ready` | ~5 | Status | Preservar como campo técnico |
| `comprehensive` | ~5 | Adjetivo desnecessário | Remover quando for floreio |
| `robust` | ~3 | Adjetivo desnecessário | Remover quando for floreio |
| `automated` | ~10 | Descrição técnica necessária | Manter quando preciso |

---

## 10. Arquivos que devem ser mantidos por compatibilidade interna

Os seguintes arquivos **não devem ser renomeados** porque têm dependências em testes, imports ou manifests:

- Todos os arquivos em `scripts/` e `tests/` (imports Python)
- Todos os arquivos em `datasets/` referenciados por múltiplos scripts
- Todos os arquivos em `manifests/` referenciados por orquestradores
- `configs/*.yaml` e `configs/*.json` referenciados por scripts de execução
- `requirements.txt`

---

## 11. Arquivos que podem receber nome público mais legível (sem renomear)

Criação de índice público em `revp_stage_index.md`:

| Código interno | Nome público sugerido |
|---|---|
| `v2at` | Vinculação evento-patch Recife |
| `v2au` | Sobreposição de geometria de evento |
| `v2av` | Construção de limites de patch |
| `v2aw` | Entrada de fonte geométrica |
| `v2ax` | Ingestão de geometria Recife |
| `v2ay` | Reconciliação de escopo do evento candidato |
| `v2az` | Orquestrador de replay de ponto de virada |
| `v2ba` | Bancada mínima de aquisição real |
| `v2bb` | Construtor de feed de recuperação geométrica pública |
| `v2bc` | Bancada de digitalização GIS Recife |
| `v2bd` | Recuperação de footprint de patch Sentinel |
| `v2be` | Integração de limites de patch TP1 |
| `v2bf` | Polígono de evento observado Recife TP2 |
| `v2bg` | Mineração de produto Charter 758 para TP2 |
| `v2bh` | Georreferenciamento e digitalização Charter 758 |
| `v2ca–v2cg` | Cadeia de evidências Curitiba |
| `v2ci` | Inventário de candidatos TP2 |
| `v2cj` | Priorização de candidatos TP2 |
| `v2ck` | Protocolo de digitalização manual |
| `v2cl` | Validação de geometria observada |
| `v2cm` | Replay de evento por patch |
| `v2cn` | Matriz de lacunas de evidência externa |
| `v2co` | Aquisição e auditoria de evidência externa |
| `v2cp` | Manifesto público de evidência externa |
| `v2cq` | QA geoespacial externo |
| `v2cr` | Pareamento de patch com evidência externa |
| `v2cs` | Semeadura de fontes externas reais |
| `v2ct` | Triagem de licença de fontes |
| `v2cu` | Sync de registro de fontes externas |
| `v2cv` | Checklist de descoberta de produtos externos |
| `v2cw` | Leitura regional de prontidão de evidência |

---

## 12. Risco de quebrar testes ao renomear

**Risco alto**: qualquer renomeação em `scripts/`, `tests/`, `datasets/` ou `configs/`.

**Risco baixo**: edição de conteúdo em `outputs_public/execution_reports/*.md` (não são importados por código).

**Risco zero**: criação de novos arquivos documentais (stage index, guia de estilo, relatórios de curadoria).

---

## 13. Resumo de prioridades

| Prioridade | Arquivo | Classificação | Tipo de mudança |
|---|---|---|---|
| 1 | `README.md` | `REWRITE_TEXT_ONLY` | Reestruturar em formato científico |
| 2 | `final_delivery_artifact_index.md` | `REWRITE_TEXT_ONLY` | Remover 92 disclaimers repetidos |
| 3 | `protocol_c_current_status_summary.md` | `REWRITE_TEXT_ONLY` | Humanizar status em inglês |
| 4 | `protocol_c_cross_region_status_summary.md` | `REWRITE_TEXT_ONLY` | Humanizar status em inglês |
| 5 | `final_guardrails_report.md` | `REWRITE_TEXT_ONLY` | Adicionar contexto em português |
| 6 | Criação de `revp_stage_index.md` | `ADD_PUBLIC_ALIAS` | Novo arquivo de índice |
| 7 | Criação de `revp_style_and_naming_guide.md` | Novo | Guia editorial canônico |
| 8 | Relatórios de sprint individuais | `REWRITE_TEXT_ONLY` | Baixa prioridade |
| 9 | `datasets/README.md` | `NEEDS_HUMAN_REVIEW` | Verificar encoding |
| 10 | `docs/metodologia_cientifica/*.md` | `DO_NOT_TOUCH` (maioria) | Conteúdo técnico correto |
