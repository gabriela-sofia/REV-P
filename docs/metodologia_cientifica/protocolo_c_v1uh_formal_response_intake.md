# Protocolo C — v1uh: Formal Response Intake and Observed Geometry Ingestion

## Objetivo

Criar infraestrutura auditavel e fail-closed para receber, organizar, validar
e auditar respostas formais de instituicoes, especialmente arquivos que possam
conter geometria observada de eventos de inundacao/alagamento/deslizamento.

## Escopo

- Ingestao de respostas formais (SHP, GPKG, GeoJSON, KML/KMZ, CSV, XLSX, PDF, ZIP)
- Inventario de assets recebidos
- Classificacao de candidatos a geometria observada
- Mapeamento de campos para formato canonico
- Auditoria de CRS e qualidade geometrica
- Auditoria de fenomeno e janela temporal
- Montagem de fila de revisao supervisora
- Matriz de blockers por evento
- Relatorio de conclusao

## Guardrails Permanentes

- ground_truth_operational=false
- can_create_ground_reference=false
- can_create_training_label=false
- can_reopen_protocol_b=false
- no_overlay_executed=true
- no_coordinates_invented=true
- supervisor_review_completed=false
- observed_geometry_candidate_only=true
- formal_response_intake_only=true

## Regras de Bloqueio

- Geometria sem data de evento nao passa
- Geometria sem fenomeno identificado nao passa
- Geometria sem CRS nao passa
- Suscetibilidade/modelagem nao e ocorrencia observada
- Quickview/mapa estatico nao e geometria validada
- PDF sem vetor e evidencia documental, nao geometria operacional
- Dados brutos ficam em local_only/
- Nenhuma geometria recebida vira ground reference automaticamente

## Pipeline

1. formal_response_intake — varredura de inbox, SHA256, staging/quarantine
2. response_asset_inventory — inventario de conteudo (ZIP, PDF, CSV, geodata)
3. observed_geometry_candidate_audit — classificacao de candidatos
4. event_field_mapper — mapeamento para campos canonicos
5. crs_and_geometry_quality_audit — auditoria de CRS, bounds, validade
6. phenomenon_temporal_gate_audit — gates de data e fenomeno
7. supervisor_review_queue_builder — fila para revisao humana
8. completion_report — relatorio, blocker matrix, manifest

## Eventos

- PET_2022_02_15 (Petropolis 2022 — mixed)
- PET_2024_03_21_28 (Petropolis 2024 — mixed)
- REC_2022_05_24_30 (Recife 2022 — urban_flooding)

## Instituicoes

- SGB/CPRM
- DRM-RJ / NADE
- Defesa Civil Petropolis
- COMPDEC / Defesa Civil PE
- Cemaden
- ANA / HidroWeb
