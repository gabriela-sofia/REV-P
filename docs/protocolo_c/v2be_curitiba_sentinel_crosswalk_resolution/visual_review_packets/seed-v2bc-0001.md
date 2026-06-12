# Curitiba Sentinel Visual Review Packet: SEED_v2bc_0001

## Evento e janela temporal
2023-10-28; janela 2023-10-21 a 2023-11-02.

## Evidencia INMET A807
Seed Curitiba/A807/LOCAL com suporte temporal forte herdado da v2bd.

## Ranking e asset principal
Principal `V1PU_VA_00001`; score 5; confianca `LOW`.
Alternativos: `V1PU_VA_00002|V1PU_VA_00003|V1PU_VA_00004`.

## Visual, espectral e DINO
Visual `ASSET_REFERENCE_ONLY`; espectral indisponivel; DINO `NOT_LINKED`.

## Lacuna geometrica e decisao
GEOMETRY_MISSING; `READY_FOR_HUMAN_VISUAL_REVIEW_ONLY`; promotion_allowed=false.

## Proxima acao humana
`MANUALLY_RESOLVE_PATCH_ASSET_LINK_FOR_CURITIBA`.

## Guardrails
Crosswalk candidato nao e truth; Sentinel/DINO nao sao truth; nao cria label, negativo, treino ou geometria.
