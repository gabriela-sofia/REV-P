# SUSC-17C40 - Indice Transparente Review-Only com Contrato de Substituicao do Flow Accumulation

## Objetivo
Produto operacional review-only e TRANSPARENTE para comparar patches canarios (ancorados em ocorrencia oficial hidrologica) vs controles originais de mismatch, sem depender do flow_acc oficial irrecuperavel (17C36-17C39) e SEM fingir replay v6.

## Contrato de substituicao do flow_acc
- flow_acc oficial NAO equivalente, NAO calibravel (17C39); usado apenas como flag de incerteza/incompletude; nunca imputado; replay v6 literal abandonado para canarios.

## Indice transparente (3 variantes)
- Matriz de componentes: 16 patches (11 canarios + 5 controles).
- Linhas de indice: 48 (3 variantes). Normalizacao review-only combinada (nao robust_minmax v6).
- Componentes causais: topografia, hidrologia_proximidade, urbano, espectral, pluvia. Ancora observacional e flow_acc NAO sao features causais.

## Robustez (canario vs controle)
  - V1_equal_weight_available_components: canary_median=0.705 control_median=0.25587 delta=0.44913 -> canario_maior_mediana
  - V2_v6_weight_inspired_without_flow_acc: canary_median=0.60598 control_median=0.18004 delta=0.42594 -> canario_maior_mediana
  - V3_rank_median_robust_index: canary_median=0.74891 control_median=0.32439 delta=0.42452 -> canario_maior_mediana
- Concordancia entre variantes: 3/3; direcao: canary_higher_all_variants.

## Interpretacao operacional (honesta)
As areas canarias com ocorrencia oficial hidrologica tendem a apresentar sinal relativo de suscetibilidade maior que os controles no indice transparente review-only, mas com LIMITACOES fortes: disponibilidade assimetrica de componentes (controles sem topografia/drenagem/landcover), CHIRPS region-level, escala nao-oficial, sem validacao supervisionada. Ancora observacional nao e feature causal nem ground truth. flow_acc oficial ausente.

## Guardrails
- NAO score v6, NAO score v7, NAO replay v6; ocorrencia so ancora observacional; controle nao e negativo verdadeiro; ausencia nao e ausencia real; flow_acc so flag; sem imputacao escondida; sem calibrar escala com canarios; score v6/matriz oficiais intactos; 17B fail-closed.

## minimum_success_achieved: True | result_class: operational_transparent_review_index_delivered

## Proximo marco recomendado
SUSC-17C41 Empacotar o indice transparente review-only para discussao de TCC (dossie/figuras/tabela), com disclaimers fortes (nao score v6/v7, nao ground truth); e/ou reabrir aquisicao de Ground Reference oficial com geometria patch-level para G4_full. 17B fail-closed.
