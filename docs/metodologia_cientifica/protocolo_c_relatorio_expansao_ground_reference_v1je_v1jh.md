# Relatorio v1je-v1jh - Expansao de ground reference e gate de treino

## Escopo

A macroetapa v1je-v1jh ampliou a base auditavel do Protocolo C a partir dos anexos CPRM locais, dos candidatos de controle e dos artefatos Sentinel/DINO ja existentes. O trabalho foi feito sem criar label operacional, sem treinar modelo e sem descongelar DINO.

## Novas coordenadas oficiais

A v1je recuperou 35 coordenadas explicitas validas em 9 unidades documentais CPRM. O ganho principal foi sair de 1 unidade com coordenada explicita para 9 unidades com coordenada explicita.

Essas coordenadas sao candidatas oficiais de anchor porque vieram de documento oficial, com data, localidade, fenomeno e texto de coordenada. Mesmo assim, elas ainda sao referencia de revisao, nao labels de treino.

## Areas de revisao

A v1jf identificou 1 unidade que continua sem coordenada explicita e a manteve como `REVIEW_AREA_ONLY`. A localidade textual foi preservada para auditoria, mas nao foi transformada em ponto.

## Patches e batch multimodal

A v1jg registrou 42 candidatos de batch, incluindo coordenadas oficiais, uma area de revisao e controles candidatos.

O que esta pronto:

- Sentinel-2 do anchor ANEXO-II ja possui patch e QA previo;
- GEE esta autenticado;
- batch metadata para S2/S1/DEM foi criado.

O que nao foi criado:

- novos rasters S2 para todos os novos anchors;
- rasters S1;
- rasters DEM;
- labels.

Os novos candidatos agora justificam uma etapa posterior de exportacao Sentinel/SAR/DEM em lote, com QA por candidato.

## Gate de treino

A matriz v1jh ficou:

- positive_reference_candidates_count: 9;
- review_area_candidates_count: 1;
- control_candidates_count: 6;
- negative_labels_ready_count: 0;
- s2_ready_count: 4;
- s1_ready_count: 0;
- dem_ready_count: 0;
- dino_ready_count: 1;
- review_only_batch_status: `REVIEW_ONLY_BATCH`;
- weak_label_sandbox_status: `WEAK_LABEL_SANDBOX_ONLY`;
- supervised_training_gate_status: `SUPERVISED_TRAINING_BLOCKED`.

O projeto agora tem base melhor para revisao em lote, mas ainda nao tem base supervisionada. O bloqueio principal e a ausencia de negativos formais, split, protocolo de vazamento e QA multimodal completo para os novos candidatos.

## Proximo passo

O proximo passo tecnico defensavel e gerar patches S2/S1/DEM para as coordenadas recuperadas, calcular QA local e depois reavaliar o gate. Ate la, DINO permanece frozen, treino segue bloqueado e qualquer sandbox fraco fica local e sem claim cientifico.
