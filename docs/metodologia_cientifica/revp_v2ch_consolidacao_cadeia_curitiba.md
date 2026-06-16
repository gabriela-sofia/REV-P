# v2ch — Consolidação da cadeia Curitiba, manifest de release e higiene de commit

Versão: `v2ch`
Modo: consolidação/empacotamento, offline. **Não cria nova modelagem**, não cria
label, não cria negativo formal, não libera treino, não copia `local_runs/` bruto nem
arquivos pesados para o Git, e não stageia mudanças não relacionadas.

## 1. Por que o v2ch existe

A cadeia Curitiba `v2ca–v2cg` está implementada e testada. O `v2ch` a fecha como um
**bloco auditável e pronto para commit**: inventaria tudo, faz rollup de
gates/guardrails/testes, consolida a prontidão científica, planeja os artefatos
públicos leves, monta um manifest de commit seguro e separa as mudanças não
relacionadas para que não contaminem o commit.

## 2. O que foi consolidado

- **Inventário de estágios** (`curitiba_chain_stage_inventory_v2ch.csv`): 7 estágios,
  cada um com script/teste/doc/output_dir e `creates_label=false`/`creates_negative=false`/
  `allows_training=false`.
- **Inventário de outputs** (`..._output_inventory_v2ch.csv`): arquivos por estágio,
  com flag `publishable` (só `.csv/.json/.md` leves).
- **Rollup de gates** (`..._gate_rollup_v2ch.csv`): chaves-chave de cada gate
  (labels/negativos/treino/blocked_reason/next_step) interpretadas.
- **Rollup de guardrails** (`..._guardrail_rollup_v2ch.csv`): overall + contagem por
  estágio (todos PASS).
- **Rollup de testes** (`..._test_rollup_v2ch.csv`): contagem estática de funções
  `test_` por estágio.
- **Prontidão científica** (`..._scientific_readiness_rollup_v2ch.csv`): 14 itens.

## 3. O que está pronto para commit

`curitiba_commit_file_manifest_v2ch.csv` — **24 arquivos intencionais**:

- 7 scripts `scripts/multimodal/revp_v2c[a-g]_*.py`
- 7 testes `tests/test_revp_v2c[a-g]_*.py`
- 7 docs `docs/metodologia_cientifica/revp_v2c[a-g]_*.md`
- 2 templates `.../templates/curitiba_external_event_evidence_template_v2cc.{csv,md}`
- 1 registry `docs/metodologia_cientifica/dino_command_registry.md`

Todos com `include_in_commit=true`.

## 4. O que NÃO deve entrar no commit

`curitiba_unrelated_working_tree_changes_v2ch.csv` — **37 mudanças não relacionadas**
sob `datasets/protocolo_c/v2bb_*` e `docs/protocolo_c/*`, todas
`include_in_commit=false`, `change_category=UNRELATED_WORKING_TREE_CHANGE`,
`recommended_action=review_or_restore_separately_before_commit`. Também excluídos por
política: `local_runs/`, `archive_drive/`, `downloaded_sources/`, `.html/.pdf/.zip/.tif/.shp`.

## 5. Como interpretar a prontidão (readiness)

- Presente: eventos reparados (2), patch boundaries (43), bindings (86), package
  externo, download live, deep crawl, dossiê oficial.
- Ausente/bloqueado: **geometria de evento válida (ABSENT)**, QA geometry (ABSENT),
  overlay executor (`AVAILABLE_BUT_BLOCKED`), labels/negativos formais (ABSENT),
  treino (`BLOCKED`).
- Plateau científico: `CURITIBA_VALID_EVENT_GEOMETRY_REQUIRED`.

## 6. Como usar o README patch

`curitiba_readme_patch_v2ch.md` é uma seção pronta para colar no README do projeto,
explicando a cadeia, onde estão os arquivos, o que está completo, por que o overlay
real está bloqueado, o que falta (CSV/GeoJSON/KML/WKT/bbox oficial), como usar o
template e como re-rodar `v2cf`/`v2cg` quando o dado chegar.

## 7. Como usar o commit manifest

Stagear apenas os 24 arquivos listados em `curitiba_commit_file_manifest_v2ch.csv`
(`include_in_commit=true`). Conferir o checklist
`curitiba_commit_hygiene_checklist_v2ch.csv` e usar a sugestão de mensagem em
`curitiba_commit_message_suggestion_v2ch.md`. **Não** stagear `git add -A` (puxaria as
37 mudanças não relacionadas) — stagear os arquivos do manifest explicitamente.

## 8. Por que o treino segue bloqueado

A cadeia está totalmente construída, mas o overlay real precisa de geometria de evento
explícita, que não existe no repositório. `allowed_for_training_count=0`,
`supervised_training_enabled=false`. 12 guardrails de release PASS.

## Outputs

`local_runs/release/v2ch/` (git-ignored): 15 arquivos (inventários, rollups,
readiness, public plan, commit manifest, hygiene checklist, unrelated changes, README
patch, commit message, guardrails, summary, report).

Artefatos públicos leves (rastreáveis): `outputs_public/execution_reports/`,
`outputs_public/logs_summary/`, `outputs_public/tables/` (6 arquivos `*_v2ch.*`).

## Nota de guardrail

Auditoria metodológica estruturada. Esta etapa não reivindica uso operacional,
validação operacional, métrica de acurácia operacional ou modelo operacional. É
documentação e manifest apenas; nenhum evento e nenhuma geometria foram inventados;
nenhum label, negativo ou treino foi criado; nenhum output local bruto foi publicado
e nenhuma mudança não relacionada foi stageada.
