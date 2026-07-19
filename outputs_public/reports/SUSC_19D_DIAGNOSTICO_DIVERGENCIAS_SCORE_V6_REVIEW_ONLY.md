# SUSC-19D - Diagnostico de divergencias e hipoteses review-only do score_v6

## Estado herdado do 19C

O 19C avaliou 7 patches observacionais review-only (Recife 5 canarios; Curitiba 2
overlays tecnicos SAR) contra o universo nao rotulado de 293
patches. Achados herdados: score medio observado 0.5958 maior
que o background 0.5208, porem
0 observados no top-30 global; coerencia
urbana/topografica; divergencia hidrologica/de chuva; `curitiba_01101` como divergencia
forte. O universo sem evidencia e `unlabeled_background` e nunca e negativo.

## Por que o 19D existe

O 19C mostrou um paradoxo: os observados tem score medio maior mas nao aparecem no
extremo do ranking. O 19D explica tecnicamente esse resultado, sem alterar o score_v6,
sem criar score_v7 e sem criar benchmark 17B.

## Diagnostico do score_v6

O score_v6 tem 5 componentes com pesos explicitos (contrato 17c35) e e
`robust_minmax(combinacao_linear_ponderada)`. A combinacao linear reproduz o ranking
oficial exatamente (transformacao monotonica), o que permite analise de sensibilidade
por reordenacao sem recalcular nem substituir o score_v6.

| Componente | Peso | Documental | Chuva | Urbano/espectral | Papel no diagnostico |
| --- | --- | --- | --- | --- | --- |
| topography_hydrology_index | 0.400 | false | false | false | sustenta_via_topografia_e_derruba_via_hidrologia |
| rainfall_trigger_index | 0.250 | false | true | false | derruba_observados_chuva_antecedente_menor |
| urban_spectral_index | 0.200 | false | false | true | sustenta_aderencia |
| vegetation_mitigation_index | -0.100 | false | false | true | mitigacao_peso_negativo |
| evidence_support_index | 0.050 | true | false | false | confianca_documental_confundida_com_suscetibilidade |

## Patches observacionais

| patch | regiao | score_v6 | classe | percentil global | top30 global | divergencia principal |
| --- | --- | --- | --- | --- | --- | --- |
| recife_00019 | recife | 0.618887 | high | 0.6667 | false | hidrologica |
| recife_00229 | recife | 0.664999 | high | 0.7533 | false | nenhuma_familia_plenamente_divergente |
| recife_00276 | recife | 0.698025 | high | 0.8033 | false | hidrologica |
| recife_00299 | recife | 0.593865 | medium | 0.6067 | false | hidrologica |
| recife_00322 | recife | 0.567661 | medium | 0.5500 | false | hidrologica |
| curitiba_01050 | curitiba | 0.669632 | high | 0.7600 | false | hidrologica |
| curitiba_01101 | curitiba | 0.357789 | low | 0.2000 | false | fisica_topografica |

## Decomposicao por familia

| Familia | Nº features | Media observada | Media background | Coerencia | Contribuicao |
| --- | --- | --- | --- | --- | --- |
| fisica_topografica | 3 | 107.2275 | 204.6778 | 1.000 | sustenta_aderencia |
| hidrologica | 3 | 976.5236 | 738.8220 | 0.000 | derruba_ou_diverge |
| urbana_territorial | 2 | 0.4334 | 0.3225 | 1.000 | sustenta_aderencia |
| espectral_umidade | 4 | -0.1016 | -0.1390 | 1.000 | sustenta_aderencia |
| chuva_hidrometeorologica | 3 | 37.5821 | 56.9356 | 0.000 | derruba_ou_diverge |
| documental_observacional | 1 | 0.4286 | 0.0017 | 1.000 | confianca_documental_separavel_da_suscetibilidade |
| score_v6 | 1 | 0.5958 | 0.5208 | 1.000 | score_medio_observado_maior_que_background_mas_sem_top30_global |

Familias que sustentam a aderencia: fisica_topografica, urbana_territorial, espectral_umidade.
Familias que derrubam os observados: hidrologica, chuva_hidrometeorologica.

## Divergencias principais

- **Hidrologica**: distance_to_water maior, TWI e flow_accumulation menores nos
  observados - direcao contraria a esperada. A hidrologia natural nao captura drenagem
  urbana (limitacao ja registrada na cadeia 17C36-40).
- **Chuva**: CHIRPS 3d/7d/30d menores nos observados. A chuva antecedente entra como
  suscetibilidade estatica e derruba o ranking, quando deveria ser gatilho contextual.
- Com topografia_hidrologia (0.4) e chuva (0.25) somando a maior parte do peso, a
  divergencia hidrologica e de chuva limita os observados mesmo com topografia, urbano e
  espectral coerentes.

## Caso curitiba_01101

score_v6 baixo (classe low), menor percentil da amostra. Componente
topografia_hidrologia e o mais baixo entre os observados e documental=0.0. E a
divergencia observacional mais forte e permanece baixo em todos os cenarios de
sensibilidade nao oficiais.

## Sensibilidade nao oficial (review-only, nao substitui score_v6)

| Cenario | Executado | Mudanca de rank observada | Interpretacao |
| --- | --- | --- | --- |
| SENS_01_remover_documental | true | obs_top30_global=0;delta_medio_rank_vs_score_v6=-38.1 | Recife perde parte do score que vinha da documentacao; Curitiba (documental 0.0) quase nao muda; separa confianca de suscetibilidade |
| SENS_02_boost_urbano_espectral | true | obs_top30_global=3;delta_medio_rank_vs_score_v6=+33.1 | parte dos observados urbanos sobe (alguns ao top-30); patches de topografia forte perdem posicao; sobreajuste possivel |
| SENS_03_chuva_como_gatilho | true | obs_top30_global=1;delta_medio_rank_vs_score_v6=+39.6 | observados sobem no ranking porque tinham chuva antecedente menor; sugere chuva como gatilho, nao suscetibilidade estatica |
| SENS_04_sem_missingness_territorial | false | NA | nao executavel: features territoriais nao materializadas (19B); preenchimento inventado seria vies |
| SENS_05_mapbiomas_pendente | false | NA | nao executavel: pacote MapBiomas/GEE preparado no 19B ainda nao executado |

Os cenarios sao interpretativos e review-only; nao substituem o score_v6, nao sao score
oficial e nao criam score_v7.

## Hipoteses review-only

Sete hipoteses em `hipoteses_ajuste_review_only_19d.csv` (separar documental de
suscetibilidade; reduzir penalizacao por baixa documentacao; aumentar peso
urbano/espectral; chuva como gatilho; hidrologia urbana; preencher territorial; ampliar
amostra). Todas com `pode_virar_score_v7_agora=false`.

## Por que o score_v7 segue bloqueado

`SCORE_V7_NAO_AUTORIZADO`: bloqueado por amostra minima, por missingness territorial
herdado do 19B e por ausencia de benchmark. As hipoteses sao review-only e nenhuma vira
score oficial. O score_v6 permanece intacto.

## Por que o 17B segue nao criado

`17B_NAO_CRIADO`: sem geometria oficial de ocorrencia e sem eventos suficientes;
Curitiba e apenas tecnica SAR. Nenhum benchmark 17B e criado.

## Por que nao e ground truth nem treino

Evidencia review-only sem geometria de ocorrencia confirmada; SAR e pos-evento;
background nunca e negativo; amostra minima. ground_truth=false e
eligible_for_training=false em todas as linhas.

## Proximo marco recomendado

**SUSC-19E - Comunicacao cientifica review-only**: consolidar figuras, tabelas e
narrativa para trabalho de conclusao, com ressalvas de amostra e missingness territorial.
Estado: `19E_PRONTO_COM_RESSALVAS`.
