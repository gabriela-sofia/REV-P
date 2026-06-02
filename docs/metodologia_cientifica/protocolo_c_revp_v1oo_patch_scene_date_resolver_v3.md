# Protocolo C v1oo — Patch Scene Date Resolver v3

## Objetivo

v1oo resolve per-patch a hierarquia fechada de scene_date, usando os resultados de v1og-v1ol (grafo de proveniência), v1om (sidecar discovery) e v1on (product date parser).

## Hierarquia

A. PRODUCT_DATE_CONFIRMED: produto/SAFE/MTD/STAC com vínculo patch->asset->metadado oficial.
B. PRODUCT_DATE_PROBABLE_REVIEW_ONLY: metadado Sentinel forte mas vínculo incompleto.
C. Candidato only: filename/sidecar fraco, nunca desbloqueia temporal.
D. Bloqueado/desconhecido.

## Guardrail crítico

can_unlock_temporal=true SOMENTE se scene_date_status == PRODUCT_DATE_CONFIRMED e evidence_chain contém patch->asset->official_product_metadata/date.

## Resultado

Total patches: 2654. Confirmados: 0. Desbloqueiam temporal: 0.
