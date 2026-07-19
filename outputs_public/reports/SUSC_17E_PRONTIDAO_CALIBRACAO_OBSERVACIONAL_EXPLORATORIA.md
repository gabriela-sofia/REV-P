# SUSC-17E Prontidao e calibracao observacional exploratoria

## Estado herdado do 17D

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `968781a`
- Itens aceitos para avaliacao no 17D: 5
- Candidatos a calibracao no 17D: 0
- Status 17B herdado: 17B_PARCIAL_COM_VALIDACAO_TECNICA_SEM_PRONTIDAO_BENCHMARK
- Status 17E herdado: 17E_BLOQUEADO_SEM_CANDIDATO_CALIBRACAO
- `score_v6` alterado: False
- `score_v7` criado: False

O 17D deixou 5 itens aceitos para avaliacao review-only e nenhum candidato a calibracao,
com bloqueio unico: as features do patch candidato nao estavam materializadas naquela etapa.

## Metodologia de prontidao

Para cada item aceito calculamos qualidade de fonte, temporal, geometrica, de vinculo,
de fenomeno, de incerteza, disponibilidade e consistencia de features, estabilidade regional,
risco de vazamento temporal e severidade de bloqueios. A prontidao final classifica o item
entre calibracao forte, calibracao exploratoria review-only, prontidao parcial ou bloqueio.

As features usadas sao reais e pre-evento, ja presentes no repositorio: medias de banda
Sentinel-2 (cena de 2022-04-27, anterior ao evento) e CHIRPS diario (janela 2022-04-24 a
2022-05-23, encerrada no dia anterior ao evento de 2022-05-24 a 2022-05-30). Nenhuma feature
foi inventada; a topografia/hidrologia permanece indisponivel para os canarios.

## Criterios de calibracao forte

Exigem: item aceito, geometria real, vinculo forte com patch, incerteza aceitavel, data e
fenomeno compativeis, features minimas fisicas e urbanas/espectrais, sem contradicao critica,
sem vazamento temporal e confianca alta ou media. O componente dominante do score_v6 e a
topografia/hidrologia (peso 0.40), indisponivel para os canarios; por isso a calibracao forte
nao e atingida.

## Criterios de calibracao exploratoria review-only

Exigem: item aceito, ground_truth=false, treino=false, score_v7=false, patch e geometria
presentes, vinculo diferente de apenas-regiao, fenomeno e data compativeis, pelo menos dois
grupos de features (aqui espectral e chuva), sem vazamento temporal, justificativa preenchida
e lacunas explicitadas. Essa saida nunca substitui o score_v6 e so gera simulacao de
sensibilidade, hipotese de ajuste de peso, diagnostico de feature e recomendacao de coleta.

## Diagnostico dos 5 itens

- S17D_VAL_0001 / S17C6_CANARY_REC_00002: pronto_para_calibracao_exploratoria_review_only (bloqueio: bloqueado_por_features; acao: obter_feature_fisica)
- S17D_VAL_0002 / S17C6_CANARY_REC_00001: pronto_para_calibracao_exploratoria_review_only (bloqueio: bloqueado_por_features; acao: obter_feature_fisica)
- S17D_VAL_0003 / S17C6_CANARY_REC_00003: pronto_para_calibracao_exploratoria_review_only (bloqueio: bloqueado_por_features; acao: obter_feature_fisica)
- S17D_VAL_0004 / S17C6_CANARY_REC_00004: pronto_para_calibracao_exploratoria_review_only (bloqueio: bloqueado_por_features; acao: obter_feature_fisica)
- S17D_VAL_0005 / S17C6_CANARY_REC_00005: pronto_para_calibracao_exploratoria_review_only (bloqueio: bloqueado_por_features; acao: obter_feature_fisica)

- Prontos para calibracao forte: 0
- Prontos para calibracao exploratoria review-only: 5
- Prontidao parcial: 0
- Bloqueados: 0

## Contagem por status

- pronto_para_calibracao_exploratoria_review_only: 5

## Bloqueios principais

O bloqueio principal comum e a ausencia do grupo fisico/topografico (HAND, declividade,
elevacao, TWI), que domina o score_v6. Bloqueios secundarios: amostra regional pequena
(evento unico) e incerteza espacial que ainda requer revisao.

## Hipoteses de ajuste de pesos

- H0_baseline_pesos_documentados: pesos documentados do score_v6 sobre componentes disponiveis (permitida=true, aplicada=true)
- H1_separar_confianca_documental: separar confianca documental de suscetibilidade fisica (permitida=true, aplicada=false)
- H2_reduzir_penalidade_documental_com_evidencia_observacional: reduzir penalidade documental quando ha evidencia observacional (permitida=true, aplicada=false)
- H3_reforco_urbano: aumentar peso urbano se urban_prop/NDBI sustentarem alagamento urbano (permitida=true, aplicada=true)
- H4_reforco_chuva: aumentar peso chuva se CHIRPS/runoff sustentarem gatilho (permitida=true, aplicada=true)
- H5_reforco_agua_umidade: aumentar peso agua/umidade se MNDWI/NDWI sustentarem sinal (permitida=true, aplicada=true)
- H6_manter_bloqueado_feature_contraditoria: manter bloqueado se feature contraditoria (permitida=false, aplicada=false)

## Simulacao de sensibilidade

- H0_baseline_pesos_documentados: low=4, medium=1
- H3_reforco_urbano: low=4, medium=1
- H4_reforco_chuva: low=4, medium=1
- H5_reforco_agua_umidade: low=4, medium=1

A simulacao mantem o score_v6 intacto e cria apenas o score exploratorio review-only,
comparando sua classe com a classe do patch oficial mais proximo. Em todas as linhas:
score_oficial=false, substituir_score_v6=false, usar_em_treino=false, ground_truth=false.

## Acoes minimas para desbloqueio

- obter_feature_fisica (HAND/declividade/elevacao/TWI) para habilitar calibracao forte.
- obter_feature_urbana (urban_prop/cobertura da terra) para reforcar o componente urbano.
- ampliar_amostra_regional com eventos datados em outras regioes.
- reduzir_incerteza_geometrica com metrica quantitativa do footprint.

As recomendacoes prospectivas de calibracao futura estao em
`recomendacoes_calibracao_futura.csv` e as acoes minimas por item em
`acoes_minimas_desbloqueio.csv`.

## Gate final 17E

- itens_aceitos_avaliados: passou=true (5 / 1)
- com_geometria_real: passou=true (5 / 5)
- com_vinculo_forte: passou=true (5 / 5)
- com_pelo_menos_2_grupos_features: passou=true (5 / 5)
- prontos_calibracao_forte: passou=true (0 / >=0)
- prontos_calibracao_exploratoria: passou=true (5 / >=1)
- prontidao_parcial: passou=true (0 / >=0)
- ground_truth_false: passou=true (0 / 0)
- trainable_false: passou=true (0 / 0)
- score_v7_ausente: passou=true (true / true)
- caminho_funcional_entregue: passou=true (true / true)
- status_final_17e: passou=true (17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY / enum)

- Status final: **17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY**
- Caminho funcional: **calibracao_observacional_exploratoria_review_only**

## Impacto no 17B

O 17B permanece sem prontidao de benchmark: nao ha ground truth, a amostra esta concentrada
em um unico evento e regiao e nao ha score_v7. O estado 17B passa a
`17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK`, refletindo que existe calibracao exploratoria review-only, mas nao
benchmark.

## Conclusao

- O que finalmente funcionou: a calibracao observacional exploratoria review-only, sustentada
  por features espectrais e de chuva reais e pre-evento, com simulacao de sensibilidade e
  hipoteses de ajuste de peso, sem tocar no score_v6 oficial.
- O que ainda segue bloqueado: calibracao forte, ground truth, treino, score_v7 e benchmark 17B,
  todos por ausencia de features fisicas/topograficas e por amostra concentrada.
- Proximo marco recomendado: SUSC-17F para extrair as features fisicas/topograficas reais dos
  patches canario e ampliar a amostra regional, condicoes minimas para promover a calibracao de
  exploratoria para forte.
