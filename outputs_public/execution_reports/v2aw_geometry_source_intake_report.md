# v2aw - Geometry Source Intake + Recife Patch Boundary Filling Scaffold

## 1. Objetivo
Criar o canal auditavel para receber, validar, normalizar e auditar geometrias reais de
boundary de patch (e geometrias observadas de evento), comecando pelos 55
patches Recife P1 da fila da v2av. A etapa nao inventa geometria; cria templates, validacao,
prontidao e instrucoes para inserir dados reais sem contaminar o projeto.

## 2. Entradas lidas
- `v2av_patch_boundary_recovery_queue.csv`
- `v2at_event_patch_package_registry.csv`
- `ground_reference_event_registry.csv`

## 3. Arquivos criados
- `datasets/v2aw_patch_geometry_sources_template.csv`
- `datasets/v2aw_event_geometry_sources_template.csv`
- `datasets/v2aw_geometry_source_validation_registry.csv`
- `datasets/v2aw_recife_p1_geometry_readiness.csv`
- `docs/v2aw_geometry_source_intake_instructions.md`
- `datasets/examples/v2aw_geometry_intake/`
- `outputs_public/execution_reports/v2aw_geometry_source_intake_report.md`
- `outputs_public/execution_reports/v2aw_geometry_source_intake_summary.json`
- `outputs_public/logs_summary/v2aw_geometry_source_intake.txt`
- `outputs_public/execution_reports/v2aw_artifact_index_supplement.md`

## 4. Contagens
- Total de patches Recife P1 (template): **55**
- Fontes de patch fornecidas: **0**
- Fontes de patch validas (prontas p/ v2av): **0**
- Fontes de evento fornecidas/auditadas: **9**
- Fontes de evento validas como poligono (prontas p/ v2au): **0**
- Pontos-ancora de evento (CPRM) auditados: **9** (ancora, nao overlay)
- Patches prontos para v2av: **0**
- Patches prontos para overlay v2au: **0**

## 5. Principais blockers
- Geometria ausente: **0**
- CRS desconhecido: **0**
- Geometria invalida: **0**
- Ponto-ancora (nao overlay): **9**

## 6. O que precisa ser preenchido manualmente
Para cada patch Recife P1 em `datasets/v2aw_patch_geometry_sources_template.csv`:
`source_type` (bbox/wkt/geojson_inline/geojson_file), `geometry_value` ou `geometry_path`,
`crs` (obrigatorio), `provenance_type`, `provenance_note`, `digitized_by`, `digitized_at`,
`source_document`, `source_confidence`, `license_status`, `review_status`.
Instrucoes completas em `docs/v2aw_geometry_source_intake_instructions.md`.

## 7. Confirmacoes metodologicas explicitas
- Nenhum label operacional/binario foi criado (`can_create_operational_labels=false`).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth final foi declarado; ausencia de geometria virou blocker, nunca negativo.
- Geometria nunca foi inventada; ponto nunca virou boundary; ponto de evento permaneceu ancora.
- v2at/v2au/v2av nao foram sobrescritos.

## 8. Interpretacao metodologica
GEOMETRY_SOURCE_INTAKE_READY_NOT_FOR_TRAINING. Como ainda nao ha geometrias reais fornecidas, todos os
patches Recife P1 ficam bloqueados por geometria ausente. Isso e correto: a v2aw cria o canal
de entrada auditavel; ela nao resolve a falta de dado externo, mas garante que esse dado entre
de forma rastreavel, sem fabricar geometria nem criar label/ground truth/treino.
