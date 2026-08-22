# Estado do modelo por região

O REV-P não entrega um único modelo operacional para as três regiões. O estado é diferente em cada uma, e essa diferença é reportada explicitamente em vez de generalizada. Os números abaixo são os da base harmonizada: tabela única com 65.070 pontos elegíveis ao ajuste fluvial, reduzidos a partir de seis fontes, com chuva de fonte única (Open-Meteo/ERA5-Land) em toda a base.

## Rótulo positivo e hierarquia do negativo

Positivos: SEDEC/Recife (154) e SIAC 156/Curitiba (1.045), mais Diário Oficial e bases internacionais.

O negativo é tratado em três níveis, declarados por linha:

| Nível | Fonte | Volume |
|---|---|---|
| Observado | Copernicus EMS | 25.249 pontos em 119 AOIs; ativação EMSR720 no Rio Grande do Sul com 216,55 km² na proporção 5,94:1 |
| Exclusão qualificada | Environment Agency/UK | 7.476 pontos (3.738 / 3.738) em 201 eventos independentes |
| Exclusão qualificada | SIAC 156/Curitiba | 114 pontos antes classificados por ausência, reclassificados pelo mesmo padrão de critérios |
| Ausência de registro | Recife e Petrópolis | — |

A unidade independente é o evento ou a AOI, nunca o ponto. É também a unidade que o produto devolve: o contrato responde por área.

## Recife — modelo operacional entregue

Regressão logística penalizada de Firth (`v12`), 278 pontos (154 positivos / 124 negativos) confirmados por Defesa Civil, ANA e Diário Oficial. LOO-AUC = 0,68 (5-fold repetido: 0,67 ± 0,01). Os seis sinais de coeficiente preservam a coerência física esperada. O motor de inferência local e o contrato de API foram implementados e auditados ponta a ponta — ver `outputs_public/data/susc_20d_motor_inferencia_local_mvp_recife/` e `susc_20e_api_contrato_inferencia_recife/`.

## Curitiba — modelo treinado, não operacional

O mesmo método foi treinado para Curitiba (`SUSC-20N`) e apresenta AUC 0,65 sob validação embaralhada, mas colapsa para 0,52 em holdout temporal real de 2026. Sete diagnósticos independentes (`SUSC-20O` a `SUSC-20T`) descartaram vazamento espacial, sazonalidade, ruído de amostra, deriva administrativa e correlação com El Niño/La Niña. Um GBM monotônico causal (`SUSC-21A`) confirma não linearidade real no fenômeno, mas não resolve a generalização temporal.

Na base harmonizada o conjunto tem 1.471 unidades de validação (grupos por evento, não pontos), e a trava de eventos por preditor mostra que Curitiba não sustenta holdout próprio: 114 negativos contra 1.238 positivos. Por não generalizar, o modelo não é declarado operacional; o resultado entra como achado negativo informativo.

## Petrópolis — servido por transferência, sem referência local

Não há ponto rotulado na região: enchente e movimento de massa não estão separados nas fontes disponíveis, e a região tem zero linhas na tabela única. Não há, portanto, modelo próprio nem validação local.

O que existe é aplicação por transferência. A cadeia de terreno já cobria a região, então a grade de suscetibilidade saiu sem aquisição nova, e 91,3% do território cabe na faixa de HAND que o modelo de serra viu nas AOIs europeias. O contrato responde com maturidade `transferencia_sem_referencia_local`: escore por semelhança de terreno, nunca afirmação de acerto.

## Ajuste por classe de relevo e holdout temporal

| Ajuste | AUC | Coeficientes |
|---|---|---|
| Serra | 0,7916 | `hand_m` −1,44 [−3,11; −0,83] |
| Planície | 0,7245 | `hand_m` −2,10 [−2,78; −1,56]; `twi_dinf` +0,40 [+0,33; +0,45] |
| Planície aplicada à serra | 0,7957 | — |

O modelo de planície aplicado à serra supera o que a serra alcança sozinha: a relação é a mesma nos dois terrenos, e o que separa é quão bem cada um está estimado. O estrato íngreme tem 19 eventos positivos e comporta uma variável, não quatro — a trava de eventos por preditor foi verificada antes de cada ajuste.

O holdout temporal roda sobre o piloto inglês em janela expansiva: 201 eventos em 110 datas entre 2000 e 2025, oito cortes, todos na faixa 0,70–0,88 fixada antes, AUC médio 0,7992, com IC95 por bootstrap de grupos em cada corte. Isso refuta que o colapso temporal seja propriedade do método; não prova estabilidade em serra tropical.

## Grade de aplicação

Grade a 120 m nas três regiões, derivada da cadeia de terreno já existente: 56.666 células em Recife, 65.275 em Curitiba, 172.015 em Petrópolis. A chuva entra como cenário, não como camada — na escala de ~11 km em que é medida ela desloca o escore e não muda o ordenamento. Célula fora do domínio do ajuste fica vazia no mapa em vez de receber escore baixo: recusar não é o mesmo que dizer que ali é seguro. Em Curitiba a elevação está a 5,05 desvios do domínio de ajuste e nenhuma célula cai na faixa vista.

## DINOv2 — nunca foi usado como modelo causal

Codificador visual congelado, nunca ajustado aos dados do projeto, usado só para medir similaridade entre áreas e alimentar a fila de revisão. Três tentativas independentes de promovê-lo a preditor fecharam sem sinal, e a rota está encerrada como candidata a variável. Os embeddings seguem no repositório apenas como análise estrutural exploratória.
