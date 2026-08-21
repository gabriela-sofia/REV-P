# Sintese tecnica para artigo - SUSC-18H

## O que o REV-P e agora

O REV-P e um estudo de suscetibilidade multimodal auditavel a inundacao,
organizado por patch e sustentado por evidencia observacional reproduzivel.
A cadeia 17C ate 18G consolidou referencias somente revisao (review-only) em tres
regioes com maturidade diferente.

## Por que nao e previsao operacional

Nao ha previsao operacional. Todas as saidas sao somente revisao, sem alvo de
treino e sem rotulo. O score_v6 permanece intacto e nenhum score_v7 foi criado.

## Por que nao e ground truth

Nenhuma matriz declara ground_truth. As referencias fortes de Recife e a
referencia tecnica SAR de Curitiba sao evidencia para revisao humana, nao verdade
de campo. Curitiba oficial segue bloqueada por ausencia de geometria de ocorrencia.

## Por que e suscetibilidade multimodal auditavel

Cada evidencia tem linhagem separada por entidade: registro de evento, footprint
de fonte, vinculo derivado de patch, evidencia de feature, avaliacao de score e
estado de gate. Tudo e rastreavel a arquivos publicos reproduziveis.

## O que Recife provou

Recife entregou 1 evento com 5 canarios fortes somente revisao, com features
fisicas diretas, espectrais e de chuva, e calibracao forte review-only. A
divergencia fisica (terreno elevado) foi preservada, nao mascarada.

## O que Curitiba tecnica acrescentou

Curitiba adicionou uma segunda regiao tecnica via SAR Sentinel-1: patch_stats
real com 43 linhas e um footprint tecnico compacto com 2 overlays somente
revisao (CUR_01050 e CUR_01101). Nao substitui a geometria oficial, ainda pendente.

## O que Petropolis bloqueia

Petropolis reune muitos registros de fenomeno misto (deslizamento e inundacao)
sem separacao e sem geometria forte. Nao entra como evidencia de inundacao ate a
separacao do fenomeno. Ha 1 candidato contextual de 2024.

## Por que o 17B ainda nao foi criado

O 17B exige no minimo 3 eventos distintos, 2 regioes e 20 vinculos fortes, com
controles definidos. O estado atual e `17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA`: duas
regioes tecnicas e dois eventos somente revisao, abaixo dos minimos. Nenhum
benchmark foi criado.

## Qual e a proxima frente cientifica

A proxima frente estrutural e a matriz multimodal escalavel por patch (SUSC-19A):
consolidar features fisicas, urbanas, espectrais e de chuva em todas as regioes,
com auditoria de cobertura, mantendo somente revisao e sem score_v7.
