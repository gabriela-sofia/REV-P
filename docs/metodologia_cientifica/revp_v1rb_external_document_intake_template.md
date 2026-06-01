# v1rb — External Document Intake Template

## Objetivo

Fornecer um template em branco para inserir documentos externos coletados MANUALMENTE, mais um descritor de schema. Nenhum dado e preenchido; nenhuma URL e baixada.

## Como usar

1) Preencher o template com documentos coletados. 2) Apontar REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH para o CSV preenchido. 3) Rodar v1rc para validar.

## Campos

document_id, source_name, source_family, region, hazard_type, event_date_text, event_location_text, url_or_reference, local_document_hash, access_date, license_note, evidence_type, temporal_precision_claim, spatial_precision_claim, reviewer_notes, intake_status.

## Guardrails

Template review-only. Nao baixa nada. Nao cria label, target ou ground truth. url_or_reference nunca e acessada automaticamente.
