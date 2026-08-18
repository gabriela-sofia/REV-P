# Sintese visual para apresentacao - SUSC-18H

## Slide 1 - O que e o REV-P
- Suscetibilidade multimodal auditavel a inundacao, por patch.
- Somente revisao: sem previsao operacional, sem rotulo, sem treino.

## Slide 2 - Regra de ouro
- score_v6 intacto; score_v7 nao criado; benchmark 17B nao criado.
- ground_truth = falso em toda a cadeia.

## Slide 3 - Recife (referencia forte)
- 1 evento, 5 canarios fortes somente revisao.
- Features fisicas diretas, espectrais e chuva; divergencia fisica preservada.

## Slide 4 - Curitiba (segunda regiao tecnica)
- SAR Sentinel-1: patch_stats real de 43 linhas.
- Footprint tecnico compacto com 2 overlays (CUR_01050, CUR_01101).
- Geometria oficial ainda pendente (18D aguardando resposta).

## Slide 5 - Petropolis (bloqueado)
- Fenomeno misto sem separacao e sem geometria forte.
- 1 candidato contextual de 2024.

## Slide 6 - Estado do 17B
- Status: `17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA`.
- Minimos: 3 eventos, 2 regioes, 20 vinculos fortes - ainda nao atingidos.

## Slide 7 - Decisao estrategica
- Deixar de depender do footprint como base central.
- Usar footprint e SAR como canario.
- Avancar para a matriz multimodal escalavel por patch.

## Slide 8 - Proximos marcos
- SUSC-18I: consolidacao tecnica SAR de Curitiba.
- SUSC-19A: matriz multimodal escalavel por patch (eixo principal).
- SUSC-19B, 19C, 19D: cobertura, avaliacao review-only e comunicacao.
