# v1pu — Visual Asset Eligibility Audit

## Objetivo

Auditar elegibilidade de assets visuais (referências de manifesto) para fila DINO review-only. Não requer scene_date nem temporal unlock. Apenas metadados — sem leitura de pixels.

## Fontes

v1fu Sentinel manifest (128 entradas), v1fm patch designation (59 entradas), v1pn inventory.

## Guardrails

can_create_label, can_train_model e target_created sempre false. DINO é representação visual review-only.

## Resultado

Assets auditados: 167. Elegíveis review-only: 128.
