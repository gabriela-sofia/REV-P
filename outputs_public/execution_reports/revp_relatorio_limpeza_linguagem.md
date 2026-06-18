# REV-P — Relatório de limpeza de linguagem

Data: 2026-06-18
Fase: curadoria editorial pública — Fase 6

Este relatório registra a auditoria de termos com linguagem robótica ou artificial nos documentos públicos do repositório, as substituições aplicadas e os itens preservados por compatibilidade.

---

## 1. Termos auditados

| Termo | Ocorrências estimadas (antes) | Arquivos principais |
|---|---|---|
| `scaffold` | ~20 | nomes de scripts/testes, alguns relatórios |
| `guardrail` | ~40 | relatórios de execução, checklists |
| `readiness` | ~60 | nomes de scripts, relatórios, tabelas |
| `BLOCKED` (status isolado) | ~80 | relatórios, summaries, checklists |
| `candidate-only` | ~10 | relatórios de evidência |
| `training-ready` | ~8 | relatórios de prontidão |
| `pipeline` (repetido) | ~100 | README, relatórios |
| `ground-truth-ready` | ~5 | relatórios |
| `comprehensive` | ~5 | textos descritivos |
| `robust` | ~3 | textos descritivos |
| `Operational label = 0 \| negative = 0 \| training = 0` | 2 | summaries de status |
| disclaimer repetido 92× | 92 | `final_delivery_artifact_index.md` |
| `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` sem contexto | 4 | status summaries |

---

## 2. Substituições aplicadas

### 2.1 `final_delivery_artifact_index.md`

**Antes**: coluna "Observacao metodologica" repetia literalmente a mesma frase em 92 linhas da tabela:
> "Resultado estrutural destinado a revisao. Nao constitui ground truth operacional, confirmacao de evento observado, classe, label, predicao ou treinamento supervisionado."

**Depois**: a frase foi movida para um parágrafo único de cabeçalho. A coluna "Observacao metodologica" foi removida da tabela. O conteúdo semântico foi preservado.

**Ocorrências antes**: 92 | **Depois**: 1 (no cabeçalho)

---

### 2.2 `outputs_public/logs_summary/protocol_c_current_status_summary.md`

**Antes**:
```
- Recife (`REC_2022_05_24_30`): PROTOCOL_VALIDATED_CANDIDATE_REFERENCE (score 0.76, uncertainty MODERATE).
- Operational label = 0 | negative = 0 | training = 0 | C7 = NOT_CREATED_BLOCKED_FOR_TRAINING.
```

**Depois**: status convertidos para prosa em português. Códigos técnicos mantidos inline entre backticks com explicação em texto corrido.

---

### 2.3 `outputs_public/logs_summary/protocol_c_cross_region_status_summary.md`

**Antes**: lista de status em inglês maiúsculo sem contexto português.

**Depois**: tabela clara com tradução dos tipos de referência e nota explicativa do estado metodológico.

---

### 2.4 `outputs_public/execution_reports/final_guardrails_report.md`

**Antes**: título "Relatorio final de guardrails"; `C4_BLOCKED_NO_FORMAL_NEGATIVES` sem explicação; lista mecânica de restrições.

**Depois**: título renomeado para "Relatório de restrições metodológicas"; `C4_BLOCKED_NO_FORMAL_NEGATIVES` citado com explicação do que significa; texto reescrito em parágrafos narrativos.

---

### 2.5 `README.md`

**Antes**: 480 linhas, 16 seções com listas defensivas mecânicas ("Interpretação não permitida: detecção operacional de inundação; predição de enchente; ..."). Seção 12 com tabela de 14 parâmetros, seção 13 com duas listas de 10+ itens cada.

**Depois**: 9 seções em formato de README científico compacto. Restrições metodológicas expressas em dois parágrafos em vez de duas listas de 14 itens. Informação factual preservada integralmente.

---

## 3. Itens preservados por compatibilidade

Os termos abaixo foram identificados mas **não alterados** nos locais indicados, pois são contratos técnicos, nomes de arquivos com dependências, ou valores de campos em CSVs importados por código:

| Termo | Onde preservado | Motivo |
|---|---|---|
| `scaffold` | Nomes de scripts e testes (`v1fw_dino_embedding_extraction_scaffold.py`) | Renomear quebraria imports |
| `guardrail` | Campos em CSVs de datasets (`guardrail_satisfied`, etc.) | Contratos de schema |
| `readiness` | Nomes de scripts, tabelas e campos CSV | Renomear quebraria imports e schemas |
| `BLOCKED` | Valores de campo em CSVs e código Python | Enum técnico |
| `training_ready` | Campos CSV e código | Campo de schema |
| `C4_BLOCKED_NO_FORMAL_NEGATIVES` | `final_guardrails_report.md` | Citado entre backticks como código; mantido com explicação |
| `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` | Tabelas e summaries | Citado como código; mantido com explicação em prosa |
| `pipeline` | Todo o código Python | Termo técnico necessário |
| `readiness` (em tabelas) | `revp_curitiba_v2ca_v2cg_release_report_v2ch.md` | Status técnico do sistema; mantido com parágrafo explicativo |

---

## 4. Arquivos revisados nesta curadoria

| Arquivo | Tipo de mudança |
|---|---|
| `README.md` | Reestruturação completa — formato científico compacto |
| `outputs_public/execution_reports/final_delivery_artifact_index.md` | Remoção de 92 disclaimers repetidos; cabeçalho único |
| `outputs_public/logs_summary/protocol_c_current_status_summary.md` | Humanização de status em inglês |
| `outputs_public/logs_summary/protocol_c_cross_region_status_summary.md` | Humanização e estrutura de tabela clara |
| `outputs_public/execution_reports/final_guardrails_report.md` | Reescrita narrativa; título humanizado |

---

## 5. Arquivos criados como camada pública

| Arquivo | Função |
|---|---|
| `outputs_public/execution_reports/revp_auditoria_curadoria_repositorio_publico.md` | Auditoria editorial completa (Fase 1) |
| `docs/metodologia_cientifica/revp_guia_estilo_nomenclatura.md` | Guia editorial canônico (Fase 2) |
| `docs/metodologia_cientifica/revp_indice_etapas.md` | Índice público de estágios com nomes legíveis (Fase 4) |
| `outputs_public/tables/revp_indice_etapas_publicas.csv` | Versão CSV do índice de estágios (Fase 4) |
| `outputs_public/execution_reports/revp_relatorio_limpeza_linguagem.md` | Este relatório (Fase 6) |
| `outputs_public/execution_reports/revp_plano_organizacao_estado_git.md` | Diagnóstico Git (Fase 7) |
| `outputs_public/tables/revp_lista_arquivos_exportacao_publica.csv` | Lista de exportação pública (Fase 8) |
| `outputs_public/execution_reports/revp_relatorio_validacao_curadoria_publica.md` | Relatório de validação (Fase 9) |

---

## 6. Avaliação de impacto

Nenhuma mudança editorial altera:
- fatos científicos ou números;
- status metodológicos (bloqueado continua bloqueado);
- schemas de CSV ou contratos de código;
- rastreabilidade ou proveniência;
- limitações ou restrições do projeto.

A curadoria reduz repetição, adiciona contexto em português para status em inglês e organiza o repositório para leitura humana.
