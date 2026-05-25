# Protocolo C v1ji - Batch multimodal de anchors oficiais

A v1ji transforma as coordenadas oficiais recuperadas nas etapas v1je-v1jh em um lote multimodal auditavel. A etapa consolida anchors oficiais unicos, tenta gerar patches Sentinel-2, Sentinel-1 e DEM/terreno via GEE, e extrai embeddings DINOv2 frozen apenas para revisao quando o par Sentinel-2 passa em QA.

## Entrada

A etapa parte de tres registros:

- `official_coordinate_recovery_hardened_registry.csv`, com coordenadas explicitas recuperadas dos anexos CPRM;
- `ground_reference_candidate_master_registry.csv`, com candidatos de referencia e controles de revisao;
- `training_gate_decision_matrix.csv`, com o bloqueio formal de treino das etapas anteriores.

As 35 expressoes de coordenadas nao sao tratadas como 35 eventos independentes. Elas sao deduplicadas por unidade documental, proximidade, data, localidade e fenomeno. O resultado esperado e um conjunto de anchors oficiais unicos, mantendo rastreabilidade para as coordenadas originais.

## Patches multimodais

Para cada anchor confirmado, a v1ji tenta gerar:

- Sentinel-2 pre e pos, com B02, B03, B04, B08, B11, B12 e bandas de mascara quando disponiveis;
- Sentinel-1 pre e pos, com VV/VH quando houver cobertura;
- DEM, slope e aspect;
- QA local de forma, CRS, pixels validos e valores invalidos;
- embedding DINOv2 frozen para pares Sentinel-2 com QA suficiente.

Todos os rasters e arquivos pesados ficam somente em `local_runs`. Os CSVs publicos registram apenas metadados sanitizados.

## Fronteira cientifica

O anchor oficial com patch QA pode ser candidato positivo de referencia para revisao. Isso nao cria label operacional.

Controles candidatos continuam sendo controles de revisao. Eles nao sao negativos formais, porque ausencia de registro nao prova ausencia de evento.

DINO permanece frozen e review-only. Embedding valido indica que ha material estrutural para revisao multimodal, nao permissao para treino.

## Gate

Nesta etapa, os seguintes campos permanecem sempre bloqueados:

- `can_create_training_label=false`;
- `can_train_model=false`;
- `can_unfreeze_dino_for_scientific_claim=false`.

Treino supervisionado continua bloqueado enquanto faltarem negativos formais, split por evento/localidade, protocolo de vazamento e governanca de labels.
