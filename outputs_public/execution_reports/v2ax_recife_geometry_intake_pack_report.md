# v2ax - Recife Geometry Intake Pack

## Objetivo
Transformar a ausencia de geometria real em fluxo manual auditavel e fail-closed.

Escopo completo: `v2ax_recife_geometry_intake_pack`. O repositorio tambem possui uma trilha v2ax
hidrometeorologica preexistente em Protocolo C; os nomes completos mantem os escopos separados.

## Entradas
- `v2aw_patch_geometry_sources_template.csv`
- `v2aw_event_geometry_sources_template.csv`
- `v2aw_recife_p1_geometry_readiness.csv`
- `v2aw_geometry_source_validation_registry.csv`
- `v2av_patch_boundary_recovery_queue.csv`
- `v2at_event_patch_package_registry.csv`
- `v2au_overlay_review_queue.csv`

## Contagens
- Patches Recife P1: **55**
- Eventos Recife comprovados: **1** (esperados no config: 3)
- Pacotes cobertos: **55**
- Geometrias de patch fornecidas: **0**
- Geometrias de evento fornecidas: **0**
- Prontos para v2aw/v2av/v2au: **0 / 0 / 0**

## Blockers
- Pendentes de geometria manual: **56**
- CRS desconhecido: **0**
- Geometria invalida: **0**
- Divergencia de eventos Recife: `EXPECTED_RECIFE_EVENTS_NOT_FOUND`

Preencha `datasets/manual_intake/recife_p1/`, rode novamente a v2ax e use apenas exports validados.
Depois execute v2aw, v2av e v2au sob revisao humana.

## Guardrails
Nenhum label, modelo, treino supervisionado, ground truth final ou promocao C4 automatica foi criado.
`can_train_model=false`; `can_create_operational_labels=false`.
