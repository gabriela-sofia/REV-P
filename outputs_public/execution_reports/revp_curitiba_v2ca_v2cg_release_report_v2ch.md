# REV-P v2ch — Consolidação da cadeia Curitiba e higiene de commit

Versão: `v2ch`
Gerado: 2026-06-16T23:53:26.598214+00:00
Cadeia: `v2ca`-`v2cg`

## 1. Por que o v2ch existe

A cadeia Curitiba (v2ca-v2cg) está implementada e testada. O v2ch não adiciona
modelagem; ele consolida a cadeia em um bloco auditável e pronto para commit:
inventários, consolidações de critérios de bloqueio/travas/testes, prontidão
metodológica, plano de artefatos públicos, manifest seguro de commit, patch de
README e mensagem de commit. Mudanças não relacionadas do worktree ficam separadas
para não entrar no commit da cadeia.

## 2. Etapas consolidadas

- `v2ca` Vinculação do registro de eventos Curitiba e aquisição de evidência — PRESENT
- `v2cb` Aquisição de evidência de evento Curitiba, leitura e entrada de geometria QA — PRESENT
- `v2cc` Entrada de evidência externa Curitiba e pacote de aquisição — PRESENT
- `v2cd` Passada de download ao vivo e materialização de evidência externa Curitiba — PRESENT
- `v2ce` Varredura aprofundada de fontes oficiais Curitiba e extração estruturada — PRESENT
- `v2cf` Monitor de entrada de evidência externa Curitiba e construtor de geometria QA — PRESENT
- `v2cg` Execução de sobreposição espacial QA Curitiba e auditoria de sensibilidade — PRESENT

Scripts encontrados: 7; testes encontrados: 7;
documentos encontrados: 7.

## 3. Consolidação de travas

- `v2ca`: PASS (14/14)
- `v2cb`: PASS (16/16)
- `v2cc`: PASS (19/19)
- `v2cd`: PASS (20/20)
- `v2ce`: PASS (24/24)
- `v2cf`: PASS (17/17)
- `v2cg`: PASS (16/16)

## 4. Prontidão metodológica

- curitiba_events_repaired: PRESENT_2
- curitiba_patch_boundaries_available: PRESENT_43
- curitiba_patch_event_bindings_available: PRESENT_86
- external_evidence_package_available: PRESENT
- live_download_pass_completed: COMPLETED
- deep_crawl_completed: COMPLETED
- official_data_request_dossier_available: PRESENT
- valid_event_geometry_available: ABSENT
- qa_geometry_available: ABSENT
- overlay_executor_available: AVAILABLE_BUT_BLOCKED
- dry_run_positive_candidates_available: ABSENT
- formal_labels_available: ABSENT
- formal_negatives_available: ABSENT
- training_ready: BLOCKED

## 5. O que está pronto para commit

Os artefatos intencionais da cadeia Curitiba (scripts, testes, documentos, template
de evidência externa e registro de comandos atualizado). Ver
`curitiba_commit_file_manifest_v2ch.csv`.

## 6. O que não deve entrar no commit

37 mudanças não relacionadas do worktree sob `datasets/protocolo_c/v2bb_*`
e `docs/protocolo_c/*`. Elas não fazem parte da cadeia Curitiba e estão registradas
em `curitiba_unrelated_working_tree_changes_v2ch.csv` com `include_in_commit=false`.

## 7. Plateau metodológico

`CURITIBA_VALID_EVENT_GEOMETRY_REQUIRED`: a cadeia está construída, mas a sobreposição real
exige geometria explícita de evento. `allowed_for_training_count=0`,
`supervised_training_enabled=false`.

## 8. Como desbloquear

Fornecer CSV/GeoJSON/KML/WKT/bbox oficial para um evento de Curitiba (ver dossiê
v2ce), colocar em uma pasta de entrada e reexecutar v2cf, depois v2cg.

## Nota de trava metodológica

Auditoria metodológica estruturada. Esta etapa não reivindica uso operacional,
validação operacional, métrica de acurácia operacional ou modelo operacional. É
somente documentação e manifest: nenhum evento e nenhuma geometria foram inventados;
nenhum label, negativo ou treino foi criado; nenhum output local bruto foi publicado;
nenhuma mudança não relacionada foi stageada.
