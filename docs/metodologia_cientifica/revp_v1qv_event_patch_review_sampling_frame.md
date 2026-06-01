# v1qv — Event/Patch Review Sampling Frame

## Objetivo

Construir o quadro amostral de unidades evento-patch e sortear uma amostra estratificada para revisao humana. Prioriza C2, lacunas contextuais, fila DINO review-only e lacunas de fonte. Inclui bloqueados como controle metodologico.

## Parametros

REVP_PROTOCOL_C_REVIEW_SAMPLE_N=24; REVP_PROTOCOL_C_MIN_PER_REGION=4.

## Resultado

Frame: 8 unidades. Amostra: 8.

## Guardrails

DINO pode priorizar revisao mas nunca prova evento (dino_validates_event=false). Nenhuma linha cria label, target ou ground truth operacional.
