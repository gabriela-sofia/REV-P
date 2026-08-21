# Sintese executiva - REV-P (review-only)

## O que o REV-P e
Framework multimodal auditavel de suscetibilidade urbana a enchentes, com avaliacao
observacional review-only. Integra features fisicas, hidrologicas, espectrais,
territoriais, de chuva e evidencia documental por patch e produz o score_v6
candidato de suscetibilidade.

## O que o REV-P nao e
Nao e predicao operacional de enchente, nao e ground truth patch-level, nao e modelo
treinado, nao e benchmark supervisionado, nao e score_v7 e nao e sistema de alerta.
A avaliacao e review-only e nao substitui validacao operacional.

## Estado atual
300 patches consolidados (100 por regiao). Cobertura fisica,
espectral e de chuva completas; cobertura territorial parcial
(33.3%) por missingness de MapBiomas/solo exposto/agua/
impermeabilizacao. 7 patches com evidencia observacional review-only
(Recife 5, Curitiba 2); background nao rotulado
de 293 (nunca negativo).

## Principal contribuicao
Um pipeline multimodal auditavel por patch com separacao explicita entre
event_record, source_footprint, derived_patch_link, feature_evidence e
score_evaluation, e um diagnostico honesto das divergencias do score_v6 sem inflar
claims.

## Achados de Recife
Cinco canarios observacionais review-only com score_v6 medio acima do background
regional e coerencia urbana e topografica. Referencia mais solida da cadeia, ainda
assim review-only e de uma unica regiao e um unico evento.

## Achados de Curitiba
Dois overlays tecnicos SAR review-only. O SAR e footprint pos-evento e nao e
geometria oficial. Um patch com score alto e outro (curitiba_01101) com score baixo,
a divergencia observacional mais forte.

## Bloqueio de Petropolis
Fenomeno misto (deslizamento e inundacao) sem separacao. Permanece bloqueado e fora
do conjunto observado; nao e promovido.

## Por que 17B nao existe
Nao ha benchmark 17B: faltam geometria oficial de ocorrencia e eventos suficientes;
Curitiba e apenas tecnica SAR. benchmark_17b_criado=false.

## Por que score_v7 nao existe
O score_v7 permanece bloqueado por amostra minima (7 observados),
por missingness territorial e por ausencia de benchmark. O score_v6 permanece intacto.
