# v2av - Patch Boundary Source Manifest + Patch Geometry Builder

## 1. Objetivo
Atacar o blocker dominante da v2au (`BLOCKED_MISSING_PATCH_GEOMETRY`): descobrir todos os
patch_id do REV-P, rastrear a proveniencia espacial de cada um, construir geometria de
boundary de patch (GeoJSON/WKT/bbox) APENAS quando ha metadado suficiente e CRS conhecido,
e gerar uma fila de recuperacao/digitalizacao para o resto.

Nada de label, ground truth final ou modelo. Geometria nunca e inventada; sem CRS nao ha
boundary. O significado maximo de um boundary construido e `READY_FOR_V2AU_OVERLAY`.

## 2. Entradas usadas
- `v2at_event_patch_package_registry.csv`: 1
- `protocolo_c/v1us_patch_registry_resolution.csv`: 1

## 3. Saidas geradas
- `datasets/v2av_patch_boundary_source_manifest.csv`: 1
- `datasets/v2av_patch_boundary_geometry_registry.csv`: 1
- `datasets/v2av_patch_boundary_build_audit.csv`: 1
- `datasets/v2av_patch_boundary_recovery_queue.csv`: 1
- `outputs_public/execution_reports/v2av_patch_boundary_geometry_builder_report.md`: 1
- `outputs_public/execution_reports/v2av_patch_boundary_geometry_builder_summary.json`: 1
- `outputs_public/logs_summary/v2av_patch_boundary_geometry_builder.txt`: 1
- `outputs_public/execution_reports/v2av_artifact_index_supplement.md`: 1

## 4. Contagens
- Patches unicos descobertos: **167**
- Patches com fonte espacial encontrada: **0**
- Patches com CRS: **0**
- Boundaries construidos: **0**
- Boundaries bloqueados: **167**
- Arquivos GeoJSON escritos: **0**
- Prontos para overlay v2au: **0**
- Patches prioridade Recife: **55**

## 5. Principais blocking_reason
- `NO_SPATIAL_METADATA_FOUND`: 167

## 6. Confirmacoes metodologicas explicitas
- Nenhum label operacional/binario foi criado (`can_create_operational_labels=false`; GATE_09 PASS).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth final foi declarado; ausencia de metadado nunca virou negativo.
- Geometria nunca foi inventada; sem CRS nao ha boundary; tamanho default nao foi usado.
- Center point so vira boundary com opt-in explicito (desligado por padrao).

## 7. Interpretacao metodologica
PATCH_BOUNDARY_RECOVERY_READY_FOR_OVERLAY_NOT_FOR_TRAINING.

Quando o repositorio nao tem metadado espacial de patch suficiente, o resultado correto e
bloquear os boundaries e gerar uma fila clara de recuperacao (prioridade Recife). Quando
bbox/WKT/GeoJSON real com CRS existir, os boundaries sao construidos e ficam prontos para a
v2au processar em nova execucao, sempre como candidato a overlay, nunca como verdade final.
