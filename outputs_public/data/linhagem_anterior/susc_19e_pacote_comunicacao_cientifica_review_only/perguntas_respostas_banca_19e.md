# Perguntas e respostas de banca - REV-P (review-only)

## Por que nao e ground truth?
Porque nao ha verdade de campo confirmada por patch. A evidencia e review-only, sem
geometria de ocorrencia confirmada; canarios e footprints SAR nao sao ground truth
patch-level.

## Por que nao e previsao?
Porque o REV-P estima suscetibilidade estrutural, nao a ocorrencia de um evento no
tempo. Nao ha modelo preditivo operacional nem sistema de alerta.

## Por que nao treinou modelo?
Porque nao ha rotulos nem targets: eligible_for_training=false em toda a cadeia. A
avaliacao e review-only e nao supervisionada.

## Por que usar SAR?
O SAR fornece footprint pos-evento em Curitiba como referencia observacional
review-only, util para revisao. Ele nao entra como feature pre-evento.

## Por que Curitiba SAR nao e oficial?
Porque o footprint SAR e derivado de sensor pos-evento e nao e geometria oficial de
ocorrencia; Curitiba aguarda resposta oficial (18D).

## Por que Petropolis ficou bloqueado?
Porque o fenomeno e misto (deslizamento e inundacao) sem separacao e sem geometria
forte. Permanece bloqueado e fora do conjunto observado.

## Por que score_v7 nao foi criado?
Porque ha bloqueio por amostra minima (7 observados), por missingness
territorial e por ausencia de benchmark. Criar score_v7 agora seria overclaim.

## O que significa review-only?
Significa uso restrito a revisao: a evidencia e as metricas servem para inspecao e
diagnostico, nao para treino, ground truth, score_v7 ou operacao.

## O que o REV-P entrega de util?
Um pipeline multimodal auditavel por patch, com separacao explicita das camadas e um
diagnostico honesto das divergencias do score_v6, reproduzivel e sem overclaim.

## Qual e a principal limitacao?
A amostra observacional minima (7 patches) combinada com missingness
territorial e ausencia de benchmark; com 7 patches nao ha conclusao
estatistica forte.

## Qual seria o proximo passo cientifico?
Ampliar a amostra observacional com geometria oficial, preencher a cobertura
territorial e separar o fenomeno de Petropolis, mantendo tudo review-only ate haver
base para benchmark.
