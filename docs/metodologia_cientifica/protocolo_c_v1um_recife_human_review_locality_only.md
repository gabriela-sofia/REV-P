# Protocolo C v1um - Recife Human Review Locality-Only

O pipeline REV-P organiza, redige, amostra, ranqueia e prepara os candidatos locality-only para Revisao Humana auditavel.

## Escopo
- event_id: REC_2022_05_24_30
- status_maximo: RECIFE_LOCALITY_ONLY_HUMAN_REVIEW_CANDIDATE
- A Revisao Humana e uma etapa metodologica de organizacao e avaliacao de evidencias.
- A etapa produz pacotes, filas, amostras, agregacoes e matrizes sem overlay.

## Guardrails
- ground_truth_operational=false
- can_create_ground_reference=false
- can_create_training_label=false
- can_reopen_protocol_b=false
- dino_usage=SUPPORT_ONLY
- no_overlay_executed=true
- no_coordinates_invented=true
- human_review_package_created=true
- human_review_queue_ready=true
- human_review_status=PREPARED_NOT_OPERATIONAL
- supervisor_review_completed=false
