# SUSC-17C6 — Canário de Aplicabilidade Multimodal Review-Only

Status: somente para revisao. `review_only=true`; `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `official_patch_created=false`; `official_patch_link_created=false`; `eligible_for_17b_now=false`.

## Objetivo do marco

Este marco testa se a esteira tecnica multimodal do REV-P consegue representar um patch candidato por multiplas camadas de evidencia, sem criar verdade-terreno, sem treinar modelo e sem promover score v7.

## Cadeia validada

Geometria observacional -> patch candidato -> matriz multimodal -> contrato de embeddings -> saida somente para revisao.

## Por que o teste e sintetico e somente para revisao

Os patches criados aqui sao candidatos e nao entram na grade oficial SUSC. As features fisicas, urbanas, espectrais, chuva, SAR e embeddings reais ainda nao foram extraidas para esses candidatos. O smoke test de embedding e apenas um hash deterministico para validar schema, join e interface.

## Geometria Charter 758

O Charter 758 entra como geometria observacional candidata real do evento `REC_2022_05_24_30`. O SUSC-17C5 registrou que ela intersecta zero patches da grade SUSC Recife atual e fica a `1398.4 m` do patch oficial mais proximo. Por isso este marco cria uma grade candidata separada, com prefixo proprio, sem alterar a grade oficial.

`REC_00019` nao e usado como patch SUSC neste marco. Ele permanece no namespace historico do Protocolo C e nao vira equivalencia automatica com a grade SUSC.

## Patches candidatos

Foram criados `5` patches candidatos e `5` vinculos candidatos patch-evento. Todos usam prefixo `S17C6_CANARY_REC_`, `official_patch=false`, `official_patch_link=false`, `review_only=true`, `trainable=false` e `ground_truth=false`.

## Contrato de camadas multimodais

Camadas registradas: physical_static, urban_territorial, sentinel2_spectral, rainfall_trigger, documentary_evidence, sar_observational, embedding_representation.

Camadas disponiveis de verdade para os patches candidatos: documentary_evidence. Camadas missing/not_available: embedding_representation, physical_static, rainfall_trigger, sar_observational, sentinel2_spectral, urban_territorial.

## O que foi real

- Geometria Charter 758 candidata, herdada do SUSC-17C4.
- Grade base SUSC Recife, herdada do SUSC-17C5 apenas como referencia de alinhamento.
- Contrato baseado nas colunas e artefatos ja existentes no repositorio.
- Patches candidatos geometricos somente para revisao.

## O que foi sintetico

- Placeholders de disponibilidade multimodal para testar fluxo e missingness.
- Smoke test de interface de embeddings com hash deterministico, sem tile real e sem execucao DINOv2/SatMAE/Scale-MAE.

## Papel futuro da matriz multimodal

A matriz canaria registra, por patch candidato, quais camadas existem, quais faltam e por que o registro nao pode virar evidencia cientifica. O readiness score e operacional, nao e score de suscetibilidade, nao valida o score v6 e nao autoriza score v7.

## Bloqueadores para virar evidencia cientifica

Faltam patches oficiais ou politica explicita para grade candidata, extracao real de features, tile real, embedding real, QA aceito, runtime SAR, politica de score e pacotes formais P0.

## Por que 17B e score v7 seguem bloqueados

O 17B exige vinculo com patch oficial ou politica aprovada, QA aceito, features reais e evidencia observacional suficiente. Nada disso foi promovido neste marco. O score v7 continua inexistente e inelegivel.

## Proximo marco recomendado

`SUSC-17C7 Plano de Extracao de Features para Patches Candidatos`. Se o maior bloqueio operacional for entrada real de embedding, a alternativa e `SUSC-17C7 Preparacao de Tile Real / Entrada DINO`. Se o maior bloqueio institucional continuar sendo P0, a alternativa e `SUSC-17C7 Pacote de Solicitacao Formal`.
