# Critérios de publicação do REV-P

Este documento define o que pode entrar no repositório público do REV-P, o que deve permanecer local e o que vai para subdiretórios de arquivo interno. As regras aqui descritas valem para outputs novos e para revisão de arquivos existentes.

---

## O que pode ser publicado

**Documentação metodológica** — arquivos `.md` que descrevem a metodologia, os gates do Protocolo C, as limitações científicas e as decisões de projeto. Devem estar em português, sem jargão de automação, e não podem afirmar resultados que o pipeline não produziu.

**Manifests e registros auditáveis** — tabelas CSV que descrevem o que existe (corpus, patches, evidências, fontes), onde está e como foi produzido, sem replicar dados pesados. Incluem schemas e registros de rastreabilidade.

**Figuras derivadas** — figuras geradas a partir de dados públicos ou de métricas resumidas já publicadas. Não incluem visualizações diretas de rasters brutos ou de dados locais não publicáveis.

**Métricas descritivas** — resultados de análise estrutural (similaridade DINOv2, PCA, agrupamentos exploratórios) expressos como tabelas resumidas. Não representam desempenho de modelo operacional.

**Relatórios de execução e QA** — documentos que registram o processo, as validações executadas, os resultados dos testes e os pontos de bloqueio metodológico.

---

## O que deve permanecer local

Os seguintes materiais **não devem ser versionados** no repositório público:

- GeoTIFFs, rasters Sentinel, imagens brutas de satélite
- Shapefiles, GeoJSONs, geodatabases, vetores brutos com licença variável
- Embeddings `.npz` ou `.npy`
- Dados PE3D/MDE
- Outputs de execução local (`local_runs/`)
- Pesos de modelo ou checkpoints (não existem neste projeto, mas estão explicitamente proibidos)
- Logs brutos de execução com paths absolutos de máquina local
- Ambientes virtuais (`.venv/`, `__pycache__/`)
- Arquivos com credenciais ou tokens de acesso
- Qualquer dado que requeira licença de redistribuição não verificada

---

## O que vai para arquivo de etapas

Arquivos intermediários de sprint que já cumpriram sua função e não precisam ficar na raiz dos diretórios públicos podem ser movidos para subdiretórios de arquivo. Isso não significa exclusão — significa organização.

Subdiretórios de arquivo:

| Diretório de origem | Subdiretório de arquivo |
|---|---|
| `outputs_public/execution_reports/` | `outputs_public/execution_reports/arquivo_etapas/` |
| `outputs_public/logs_summary/` | `outputs_public/logs_summary/arquivo_etapas/` |
| `outputs_public/tables/` | `outputs_public/tables/saidas_intermediarias/` |
| `datasets/protocolo_c/` | `datasets/protocolo_c/saidas_intermediarias/` |
| `docs/` (notas de sprint) | `docs/arquivo_notas_etapas/` |

**Candidatos típicos para arquivo de etapas:**

- Listas de verificação de entrega (`*_commit_checklist.md`)
- Resumos JSON de sprint (`*_summary.json`)
- Resumos de validação por etapa (`*_guardrail_rollup.csv`, `*_test_rollup.csv`)
- Filas de reprocessamento (`*_queue.csv`, `*_recheck*.csv`)
- Registros de exemplares negativos (`*_negative_fixture*.csv`)
- Notas de transição de operador (`*_operator_handoff.md`)
- Guias de preenchimento interno (`*_how_to_fill*.md`)
- Notas de ponto de virada metodológico (`*_turning_point*.md`)
- Bancadas de sprint (`*_workbench*.md`)
- Relatórios integrados de sprint com código de versão no nome (`revp_v2*.md`, `revp_v1*.md`) — exceto os que fazem parte do índice público canônico

**O que nunca vai para arquivo de etapas:**

- Relatórios finais de entrega (`final_*.md`, `final_*.csv`, `final_*.json`)
- Figuras validadas de publicação
- Tabelas canônicas do corpus e do DINOv2
- Manifests de corpus e evidência
- Documentos de rastreabilidade central
- Este arquivo

---

## Linguagem permitida e proibida

**Permitido:**
- "evidência contextual", "suporte territorial externo"
- "referência candidata", "referência temporal"
- "análise visual-estrutural", "análise de embeddings"
- "codificador visual congelado", "representação estrutural"
- "revisão humana", "rastreabilidade", "limites metodológicos"
- "bloqueado por evidência insuficiente", "gate não satisfeito"
- "corpus de patches", "manifesto Sentinel-first"

**Proibido:**
- "detector de inundação", "preditor", "classificador supervisionado"
- "dataset validado", "ground truth confirmado", "sistema operacional"
- "modelo treinado", "acurácia de modelo", "performance"
- "automaticamente detectado", "confirmado pelo modelo"
- "rótulo positivo", "negativo formal" — a menos que explicitamente acompanhados da ressalva de que nenhum foi criado

---

## Regras para outputs de sprints futuros

Novos arquivos criados durante sprints de desenvolvimento seguem estas regras:

1. **Scripts e testes** (`scripts/`, `tests/`) — versionados normalmente. Não precisam de curadoria de linguagem, mas devem evitar strings de output com linguagem proibida.

2. **Datasets e registros** (`datasets/`) — CSV e schemas são públicos. Dados brutos ficam locais. Schemas não devem ser alterados sem atualizar os testes correspondentes.

3. **Manifests** (`manifests/`) — versionados como parte da rastreabilidade. Não alterar sem registrar a mudança em relatório.

4. **Relatórios de sprint** (`outputs_public/execution_reports/`) — relatórios integrados de sprint (ex: `revp_v2cj_to_v2cm_integrated_report.md`) são mantidos como rastreabilidade. Checklists de commit e summaries de sprint vão para archive após a entrega do sprint.

5. **Figuras** (`outputs_public/figures/`) — apenas figuras derivadas de dados públicos. Figuras validadas de publicação não devem ser sobrescritas sem controle de versão explícito.

6. **Métricas** (`outputs_public/metrics/`) — tabelas resumidas derivadas de análise local. Não publicar outputs brutos de embedding.

---

## Sobre arquivos intermediários v1/v2

Os arquivos com padrão `*_v1*.md`, `*_v2*.md` no nome são rastreabilidade de sprint — documentam o que foi feito em cada etapa do pipeline. Eles têm valor histórico e de reprodutibilidade.

A regra é: **não deletar, organizar**. Mover para archive quando não fazem parte da entrega pública principal. Manter na raiz apenas os que são referenciados diretamente pelo `final_delivery_artifact_index.md` ou pelos relatórios de entrega.

---

## Dados brutos: regra geral

Se um arquivo contém dados que:
- não podem ser redistribuídos sem verificação de licença,
- ou que requerem ambiente local específico para serem gerados,
- ou que têm tamanho incompatível com versionamento Git,

então esse arquivo **não entra no repositório público**. O manifesto ou registro auditável que descreve o dado é o que vai para o Git.
