# SUSC-18A - Feature Store Multimodal Escalavel Patch-Level

## Objetivo
Consolidar os extratores reais desenvolvidos na cadeia de canarios (17C33-17C40) em uma FEATURE STORE MULTIMODAL patch-level unica, auditavel e reproduzivel, unindo populacao oficial, canarios ancorados em ocorrencia oficial hidrologica e controles de mismatch, com cobertura, lineage, disponibilidade e incerteza EXPLICITAS por patch.

## Pergunta cientifica
Conseguimos transformar os extratores reais dos canarios em uma feature store multimodal escalavel, com cobertura, lineage, disponibilidade e incerteza explicitas por patch? Sim, com missingness assimetrica honesta.

## Consumo de patches
- Oficiais consumidos: 300 (populacao Recife/Curitiba/Petropolis)
- Canarios consumidos: 11
- Controles consumidos: 5
- Registry unificado: 316 linhas

## Familias multimodais (contrato)
- 7 familias: sentinel2_spectral, chirps_rainfall, terrain_topography, hydrology_proximity, landcover_urban, transparent_review_components, lineage_quality

## Stores por familia
- Sentinel-2: 316 | CHIRPS: 316 | Terrain: 316 | Hydrology: 316 | Landcover: 316
- Matriz multimodal final: 316 linhas | Disponibilidade: 316 | Lineage: 1581 | Quality flags: 642

## Cobertura por familia (patches com feature)
- Sentinel-2: 316 | CHIRPS: 316 | Terrain: 311 | Hydrology: 311 | Landcover: 311 | Transparent: 16
- Completude media: 0.8339

## Missingness honesta
- Controles (5): sem terrain/hydrology/landcover (contexto de mismatch) -> completude 0.5.
- Oficiais (300): sem transparent_review_components (so 16 patches em 17C40) -> completude ~0.83.
- Canarios (11): 6/6 familias -> completude 1.0.
- Nada imputado; cada ausencia registrada em availability_matrix e quality_flags.

## Guardrails cientificos
- flow_acc oficial NAO tratado como equivalente (canario recomputado != oficial; oficial = valor nativo).
- Ocorrencia oficial NAO entrou como feature causal (apenas ancora observacional).
- Controle NAO virou negativo verdadeiro; ausencia NAO virou ausencia real.
- Score v6 oficial intacto; score v7 inexistente; ground truth nao criado; treino nao criado; 17B fail-closed.

## minimum_success_achieved: True | result_class: multimodal_feature_store_delivered

## Proximo marco recomendado
SUSC-18B Expandir os componentes transparentes review-only (17C40) para toda a populacao oficial usando a feature store 18A (analise por-patch review-only), mantendo missingness explicita; e/ou reabrir aquisicao de Ground Reference oficial com geometria patch-level para G4_full. Sem score v7, sem ground truth, sem treino, 17B fail-closed.
