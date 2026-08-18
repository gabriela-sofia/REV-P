# Cartao de diagnostico - Recife

## Evidencia
Cinco canarios observacionais review-only (event_anchored_canary, documental).

## score_v6
Score medio observado (Recife) acima do background regional; nenhum patch no
top-30 global. Detalhe por patch:

| patch | score_v6 | classe | percentil global | aderencia | divergencia |
| --- | --- | --- | --- | --- | --- |
| recife_00019 | 0.618887 | high | 0.6667 | fisica_topografica | hidrologica |
| recife_00229 | 0.664999 | high | 0.7533 | fisica_topografica | nenhuma_familia_plenamente_divergente |
| recife_00276 | 0.698025 | high | 0.8033 | fisica_topografica | hidrologica |
| recife_00299 | 0.593865 | medium | 0.6067 | fisica_topografica | hidrologica |
| recife_00322 | 0.567661 | medium | 0.5500 | fisica_topografica | hidrologica |

## Aderencias
Familias fisica_topografica, urbana_territorial e espectral_umidade coerentes com
maior suscetibilidade.

## Divergencias
Familias hidrologica (distance_to_water/TWI/flow_accumulation) e
chuva_hidrometeorologica (CHIRPS 3d/7d/30d menores) divergem e limitam o score.

## Hipotese review-only
Separar evidencia documental de suscetibilidade e tratar chuva como gatilho
contextual (HIP_01, HIP_04); nenhuma vira score oficial.

## Por que nao e ground truth
Evidencia observacional review-only sem geometria de ocorrencia confirmada por patch; SAR e pos-evento; nao e verdade de campo.

## Por que nao e treino
eligible_for_training=false; nenhuma linha habilita treino; a amostra e minima.

## Por que nao cria score_v7
score_v7 bloqueado por amostra, missingness territorial e ausencia de benchmark; as hipoteses sao review-only e nenhuma vira score oficial.

## Por que nao cria 17B
Sem geometria oficial e sem eventos suficientes; nenhum benchmark 17B e criado.
