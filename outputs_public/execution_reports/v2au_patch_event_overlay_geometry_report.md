# v2au - Patch-Event Overlay Geometry Engine

## 1. Objetivo
Fechar programaticamente o blocker dominante da v2at (`NO_PATCH_EVENT_OVERLAY_GEOMETRY`)
construindo a infraestrutura de **interseccao geometrica patch x evento**:
`evento observado ∩ patch Sentinel`. Le geometrias reais quando existem, valida CRS,
calcula area e `intersection_ratio`, audita gates e atualiza o status dos pacotes em
arquivos derivados, sem nunca sobrescrever os CSVs da v2at.

A decisao maxima permitida e `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`. Nunca gera `C4_OPERATIONAL_LABEL`,
`TRAINING_LABEL` nem `GROUND_TRUTH_FINAL`.

## 2. Entradas usadas
- `v2at_event_patch_package_registry.csv`: 1
- `ground_reference_event_registry.csv`: 1

## 3. Saidas geradas
- `datasets/v2au_geometry_inventory.csv`: 1
- `datasets/v2au_patch_event_overlay_registry.csv`: 1
- `datasets/v2au_event_patch_package_overlay_update.csv`: 1
- `datasets/v2au_overlay_gate_decision_audit.csv`: 1
- `datasets/v2au_overlay_review_queue.csv`: 1
- `outputs_public/execution_reports/v2au_patch_event_overlay_geometry_report.md`: 1
- `outputs_public/execution_reports/v2au_patch_event_overlay_geometry_summary.json`: 1
- `outputs_public/logs_summary/v2au_patch_event_overlay_geometry.txt`: 1
- `outputs_public/execution_reports/v2au_artifact_index_supplement.md`: 1

## 4. Contagens
- Pacotes avaliados: **172**
- Geometrias inventariadas: **9**
- Overlays calculados: **172**
- Overlays confirmados (max C4 candidate): **0**
- Bloqueados por geometria de patch ausente: **172**
- Bloqueados por geometria de evento ausente: **0**
- Bloqueados por CRS desconhecido: **0**
- Bloqueados por geometria invalida: **0**
- Bloqueados por ponto sem buffer: **0**
- Bloqueados por geometria contextual: **0**
- Sem interseccao: **0**
- Em `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`: **0**

## 5. Distribuicao de overlay_status
- `BLOCKED_MISSING_PATCH_GEOMETRY`: 172

## 6. Confirmacoes metodologicas explicitas
- Nenhum label operacional foi criado (`can_create_operational_labels=false`; OVERLAY_GATE_11 PASS).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth final foi declarado; ausencia de geometria nunca virou negativo.
- Ponto nunca virou overlay sem buffer configurado; CRS desconhecido bloqueia (fail-closed).
- Geometria contextual/risco nunca promoveu C4; geometria nunca foi inventada.
- C4 so existe como candidato sob revisao humana (`C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`).

## 7. Interpretacao metodologica
GEOMETRY_OVERLAY_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING.

Quando nao ha geometria vetorial real, o resultado correto e: C4 permanece bloqueado e a
fila de revisao/digitalizacao e gerada. Quando geometria real existe, ela e processada,
mas a promocao maxima continua sendo um candidato C4 sob revisao humana, nunca um label final.
