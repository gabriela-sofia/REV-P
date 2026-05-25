# Protocolo C v1je-v1jh - Expansao de referencias, patches e gate de treino

Esta macroetapa ampliou a base de revisao do Protocolo C sem liberar treino. O objetivo foi sair de um unico anchor confirmado para um conjunto maior de candidatos positivos documentais, controles candidatos e uma matriz formal de prontidao.

## v1je - Recuperacao endurecida de coordenadas

A v1je instalou/verificou bibliotecas leves de PDF e reprocessou os anexos CPRM/DIGEAP locais. Foram usados texto nativo, blocos, tabelas e metadados quando disponiveis. Nao houve download de novos documentos.

Resultado:

- PDFs auditados: 10;
- expressoes de coordenadas recuperadas: 35;
- coordenadas validas na faixa de Petropolis: 35;
- unidades documentais com coordenada explicita: 9;
- novas unidades com coordenada explicita em relacao ao estado anterior: 8.

As coordenadas viram candidatos oficiais de anchor para revisao. Elas nao viram label.

## v1jf - Areas de revisao por localidade

A v1jf preservou as unidades que ainda nao possuem coordenada explicita como areas de revisao. Localidade textual foi usada apenas para planejamento e auditoria, sem geocodificacao.

Resultado:

- areas de revisao: 1;
- status: `REVIEW_AREA_ONLY`;
- patch search automatico: bloqueado por falta de geometria explicita.

Area de revisao nao e positivo, nao e label e nao substitui coordenada oficial.

## v1jg - Batch Sentinel/SAR/DEM

A v1jg consolidou candidatos para lote multimodal. O GEE estava autenticado. O par Sentinel-2 ja validado para o anchor ANEXO-II foi registrado como pronto. Para novos pontos, S2/S1/DEM ficaram como plano de exportacao/aquisicao, sem inventar raster local.

Resultado:

- candidatos de batch: 42;
- patches S2 prontos por QA existente: 4 registros;
- patches S1 gerados nesta etapa: 0;
- patches DEM gerados nesta etapa: 0;
- planos S1/DEM possiveis via GEE: registrados como metadata.

Nenhum patch de batch criou label.

## v1jh - Gate de treino

A v1jh consolidou a matriz de decisao:

- candidatos positivos de referencia: 9;
- areas de revisao: 1;
- controles candidatos: 6;
- negativos formais: 0;
- embeddings DINO prontos: 1;
- status review-only: `REVIEW_ONLY_BATCH`;
- sandbox fraco: `WEAK_LABEL_SANDBOX_ONLY`;
- gate supervisionado: `SUPERVISED_TRAINING_BLOCKED`.

O sandbox fraco e apenas uma possibilidade local de engenharia. Ele nao e resultado cientifico e nao autoriza descongelar DINO.

## Boundary

Permanece bloqueado:

- label positivo operacional;
- label negativo formal;
- treino supervisionado;
- descongelamento DINO para claim cientifico;
- reabertura do Protocolo B.

Para liberar treino real ainda faltam patch QA para multiplos positivos, controles formais, negativos com protocolo de ausencia, split por evento/localidade, protocolo de vazamento e metricas supervisionadas.
