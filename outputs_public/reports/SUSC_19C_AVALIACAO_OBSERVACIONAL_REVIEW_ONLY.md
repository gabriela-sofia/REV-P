# SUSC-19C - Avaliacao observacional review-only da matriz multimodal

## Estado herdado do 19A/19B

O 19A consolidou 300 patches multimodais e o 19B encaminhou a lacuna territorial
ao pacote MapBiomas/GEE. Esta etapa compara os patches observacionais review-only
com o universo nao rotulado, sem treino e sem benchmark.

## Amostra observacional

7 patches observacionais: Recife 5
(canarios review-only) e Curitiba 2 (overlays tecnicos
SAR). Petropolis fica de fora (1 patch contextual
registrado como bloqueado). Background nao rotulado: 293.

## Por que o background nao e negativo

O universo sem evidencia documentada e `unlabeled_background`
(no_documented_observational_evidence). Ausencia de evidencia documentada nao e
evidencia de ausencia; nunca e negativo.

## Ranking score_v6

Score medio observado 0.5958 contra background
0.5208. Observados no top-30 global:
0 (enrichment 0.0);
no top-30 regional: 3. Os observados ficam no
terco medio-superior do score_v6, nao no extremo.

## Hit-rate / enrichment review-only

As metricas de hit-rate e enrichment sao exploratorias e review-only. Com
7 patches, a potencia amostral e baixa e nao ha
conclusao estatistica forte. Nao sao validacao operacional nem benchmark.

## Contraste de features (global)

| Feature | Direcao esperada | Media observada | Media background | Delta | Coerente |
| --- | --- | --- | --- | --- | --- |
| elevation_mean | menor | 295.5984 | 567.1102 | -271.5118 | true |
| slope_mean | menor | 4.0942 | 7.6109 | -3.5167 | true |
| HAND_mean | menor | 21.9898 | 39.3123 | -17.3225 | true |
| distance_to_water_mean | menor | 2887.0738 | 2132.6735 | 754.4004 | false |
| TWI_mean | maior | 41.4649 | 82.0239 | -40.5590 | false |
| flow_accumulation_mean | maior | 1.0321 | 1.7686 | -0.7365 | false |
| urban_prop | maior | 0.7116 | 0.4173 | 0.2944 | true |
| vegetation_prop | menor | 0.1551 | 0.2277 | -0.0726 | true |
| NDVI | menor | 0.3759 | 0.5639 | -0.1881 | true |
| NDWI | maior | -0.3811 | -0.5377 | 0.1565 | true |
| MNDWI | maior | -0.4244 | -0.4803 | 0.0559 | true |
| NDBI | maior | 0.0233 | -0.1020 | 0.1253 | true |
| CHIRPS_3d | maior | 8.7857 | 9.1216 | -0.3358 | false |
| CHIRPS_7d | maior | 34.4808 | 44.8388 | -10.3581 | false |
| CHIRPS_30d | maior | 69.4797 | 116.8464 | -47.3667 | false |
| score_v6 | maior | 0.5958 | 0.5208 | 0.0750 | true |

Direcoes coerentes globais: 10/16. Ha coerencia
urbana e topografica (elevacao, declividade, HAND, urbanizacao, indices espectrais)
e divergencia hidrologica e de chuva antecedente.

## Divergencias

Registradas em `matriz_divergencias_observacionais_19c.csv`: patches observados com
score baixo/medio, divergencia fisica, divergencia de chuva, evidencia apenas SAR e
limitacao de amostra. Nenhum score e corrigido e nenhum score_v7 e proposto.

## Recife

Cinco canarios review-only, score medio acima do background regional, coerencia
urbana e topografica, amostra pequena.

## Curitiba

Dois overlays tecnicos SAR review-only; um deles com score alto e outro com score
baixo (divergencia observacional relevante). Amostra minima; SAR nao e geometria oficial.

## Petropolis excluido

Fenomeno misto sem separacao; permanece contextual/bloqueado, fora de observado.

## Por que nao e ground truth, nem treino, nem benchmark

Evidencia review-only sem geometria de ocorrencia confirmada; background nao e
negativo; amostra pequena; nenhuma metrica e benchmark.

## Por que o score_v7 segue bloqueado

Estado `SCORE_V7_NAO_AUTORIZADO`: bloqueado por amostra, por missingness
territorial herdado do 19B e por ausencia de benchmark. O score_v6 permanece intacto.

## Sintese por regiao

| Regiao | Observados | Score obs | Score background | Top-k | Status |
| --- | --- | --- | --- | --- | --- |
| recife | 5 | 0.6287 | 0.5697 | 2/5_em_top30_regional | recife_review_only_com_amostra_pequena |
| curitiba | 2 | 0.5137 | 0.5151 | 1/2_em_top30_regional | curitiba_tecnica_sar_com_amostra_minima |
| petropolis | 0 | NA | 0.4800 | sem_observados | petropolis_excluido_por_fenomeno_misto |

## Proximo marco recomendado

**SUSC-19D - Pacote de comunicacao cientifica**: consolidar figuras, tabelas e
narrativa review-only para trabalho de conclusao e apresentacao.
