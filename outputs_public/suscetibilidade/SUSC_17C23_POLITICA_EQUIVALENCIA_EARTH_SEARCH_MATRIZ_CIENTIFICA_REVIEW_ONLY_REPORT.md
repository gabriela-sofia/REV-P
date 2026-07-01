# SUSC-17C23 - Politica de equivalencia Earth Search e matriz cientifica review-only

## Objetivo
Resolver objetivamente a trava Earth Search/STAC/COG e a politica de uso dos patches candidatos, usando apenas artefatos 17C18-17C22 (sem rede). Se a equivalencia passar, promover as 40 features pre-evento de accepted_review_only_sensor_feature para accepted_scientific_review_only_sensor_feature, sem liberar treino, ground truth, score v7 ou 17B.

## Equivalencia Earth Search
- Equivalencia resolvida: True.
- Dossies avaliados: 5; equivalencias aprovadas: 5.
- Blockers de politica de fallback restantes: 0.
- Fallback aceito para review-only cientifico: 5 patches.
- Criterios provados por patch: mesma missao, plataforma, data (2022-04-27), tile MGRS 25MBM, AOI coberta, bandas reais B03/B04/B08/B11, SHA256, replay, leitura COG/window e tamanho sob politica.

## Limitacao metodologica L1C vs L2A
A cena CDSE canonica e L1C e a materializacao Earth Search e L2A. Como missao, plataforma, data, tile e AOI sao equivalentes, a diferenca de nivel de processamento e registrada como limitacao metodologica (nao bloqueio) em susc_17c23_science_scope_limits.csv.

## Politica de patch candidato
- Politica concluida: True; patches aceitos: 5.
- Cada patch prova geometria/AOI, vinculo candidato com evento, binding temporal, feature real com hash/replay, fonte fallback resolvida e nao ser patch/patch-link oficial.

## Aceite cientifico review-only e matriz multimodal
- Features aceitas como scientific review-only: 40.
- Matriz multimodal candidata: 5 linhas (deltas em bloco separado, nunca feature pre-evento).

## Guardrails
- Fallback como Ground Reference: 0; treino: 0; score v7: 0; 17B: 0.
- Features promovidas a treino: 0; Ground Reference: 0; ground truth: 0; label: 0; score v6 intacto; score v7 inexistente; 17B bloqueado.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C24 Serie temporal pre-evento multi-cena e reducao de nuvem para robustez das features sensoriais cientificas review-only
