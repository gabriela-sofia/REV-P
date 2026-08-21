# Estado do modelo por região

O REV-P não entrega um único modelo operacional para as três regiões — o estado é diferente em cada uma, e essa diferença é reportada explicitamente em vez de generalizada.

## Recife — modelo operacional entregue

Regressão logística penalizada de Firth (`v12`), treinada sobre 278 eventos reais (154 positivos / 124 negativos) confirmados por Defesa Civil, ANA e Diário Oficial. LOO-AUC = 0,68 (repetido 5-fold: 0,67 ± 0,01). Os 6 sinais de coeficiente preservam coerência física esperada; a variável de chuva antecedente (`rain_decay_index_api_chirps`) é o preditor mais forte e estatisticamente significativo (p < 0,0001).

Motor de inferência local e API de contrato foram implementados e auditados ponta a ponta — ver `outputs_public/data/linha_causal/susc_20d_motor_inferencia_local_mvp_recife/` e `outputs_public/data/linha_causal/susc_20e_api_contrato_inferencia_recife/`.

## Curitiba — modelo treinado, não operacional

O mesmo método (Firth) foi treinado para Curitiba (`SUSC-20N`) e apresenta AUC de 0,65 sob embaralhamento, mas colapsa para 0,52 em holdout temporal real de 2026 — não generaliza para prever eventos futuros. Sete diagnósticos independentes (`SUSC-20O` a `SUSC-20T`) descartaram vazamento espacial, sazonalidade, ruído de amostra, deriva administrativa e correlação com El Niño/La Niña como causa do colapso. Um GBM monotônico causal (`SUSC-21A`) confirma não linearidade real no fenômeno, mas não resolve o problema de generalização temporal.

Por não generalizar, este modelo não é declarado operacional. O resultado é reportado como achado negativo informativo, não omitido.

## Petrópolis — sem modelo

Enchente e deslizamento não estão separados nas fontes disponíveis para a região. Sem essa separação, não há base suficiente para treinar ou validar um modelo nesta entrega.

## DINOv2 — nunca foi usado como modelo causal

Os embeddings DINOv2 (encoder visual pré-treinado e congelado) foram testados como feature adicional ao modelo físico de Recife via comparação A/B e descartados — não melhoraram o modelo causal e romperiam o princípio de que a base do projeto é físico-hidrológica, não um padrão aprendido de imagem. Os embeddings seguem no repositório apenas como análise estrutural exploratória (similaridade, k-NN, PCA, medoids, outliers), não como parte do modelo de suscetibilidade.
