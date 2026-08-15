# SUSC-17H Calibracao observacional forte somente revisao

## Estado herdado do 17G

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `72dec09`
- Status final 17G herdado: 17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL
- Canarios com calibracao forte review-only possivel: 5
- `score_v6` alterado: False
- `score_v7` criado: False

## Por que o 17H e possivel

Os 5 canarios `S17C6_CANARY_REC_00001..00005` reunem features fisicas diretas (17G), espectrais e
de chuva pre-evento (17E), vinculo espacial aceito (17D) e qualidade de evidencia. Isso permite uma
calibracao forte por componentes, estritamente somente revisao.

## Metodologia de componentes

Cinco componentes normalizados a 0-1: fisico_topografico (peso 0.4),
chuva_gatilho (0.25), urbano_espectral (0.2),
umidade_espectral (0.1) e qualidade_evidencia (0.05).
Os pesos espelham o score_v6, mas o indice e somente revisao e nunca substitui o score oficial.

## Matriz integrada

- S17C6_CANARY_REC_00002: indice=0.4688 (medium); fisico=0.4569; aderencia=divergencia_fisica; dif_vs_v6=-0.1360
- S17C6_CANARY_REC_00001: indice=0.4022 (low); fisico=0.2682; aderencia=divergencia_fisica; dif_vs_v6=-0.2026
- S17C6_CANARY_REC_00003: indice=0.4114 (low); fisico=0.3514; aderencia=divergencia_multicomponente; dif_vs_v6=-0.1934
- S17C6_CANARY_REC_00004: indice=0.4045 (low); fisico=0.3069; aderencia=divergencia_multicomponente; dif_vs_v6=-0.2003
- S17C6_CANARY_REC_00005: indice=0.4309 (medium); fisico=0.4208; aderencia=divergencia_multicomponente; dif_vs_v6=-0.1739

## Simulacoes de pesos

- cenario_base_v6_compativel: indice_medio=0.4236 (low); mudanca_media=0.0000
- cenario_sem_documental_penalizante: indice_medio=0.4069 (low); mudanca_media=-0.0166
- cenario_gatilho_chuva_reforcado: indice_medio=0.4167 (low); mudanca_media=-0.0069
- cenario_umidade_espectral_reforcada: indice_medio=0.4471 (medium); mudanca_media=0.0236
- cenario_fisico_dominante: indice_medio=0.4017 (low); mudanca_media=-0.0219
- cenario_urbano_espectral_reforcado: indice_medio=0.4327 (low); mudanca_media=0.0092

## Diagnostico de aderencia

- divergencia_fisica: 2
- divergencia_multicomponente: 3

## Achado principal

A topografia direta dos canarios (HAND e elevacao altos em parte deles) NAO sustenta alta
suscetibilidade, ainda que o footprint intersecte o patch. O componente fisico medio e baixo e o
indice observacional fica abaixo da referencia score_v6 do patch oficial mais proximo. A calibracao
forte somente revisao preserva essa divergencia em vez de ajustar os pesos para forcar aderencia.

## Impacto sobre a interpretacao do 17F

O 17F usou uma referencia comparativa distante (patch baixado) que superestimava a suscetibilidade
topografica. A extracao direta (17G) e a calibracao (17H) corrigem essa leitura: a evidencia fisica
propria dos canarios aponta suscetibilidade topografica menor.

## Gate final 17H

- canarios_processados: passou=true (5 / 5)
- features_fisicas_diretas_completas: passou=true (5 / 5)
- features_espectrais_disponiveis: passou=true (5 / 5)
- chuva_disponivel: passou=true (5 / 5)
- vinculo_espacial_aceito: passou=true (5 / 5)
- componentes_normalizados_0_1: passou=true (true / true)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_allowed_zero: passou=true (0 / 0)
- score_oficial_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)
- benchmark_17b_nao_criado: passou=true (true / true)
- caminho_funcional_entregue: passou=true (true / true)
- status_final_17h: passou=true (17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA / enum)

- Status final: **17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA**
- Caminho funcional: **calibracao_forte_com_divergencia_fisica**

## Por que segue sem ground truth

O indice descreve suscetibilidade observacional review-only; nao confirma ocorrencia e nao e verdade de referencia.

## Por que segue sem treino

Sem rotulo validado, nenhum canario alimenta treino supervisionado.

## Por que segue sem score_v7

Nao ha score oficial nem score_v7; o indice e somente revisao e o score_v6 permanece intacto.

## Por que 17B ainda nao esta pronto

- minimo_3_eventos_distintos: passou=false (1 / 3)
- minimo_2_regioes: passou=false (1 / 2)
- minimo_20_patch_links_fortes: passou=false (5 / 20)
- separacao_temporal: passou=false (false / true)
- controles_definidos: passou=true (true / true)
- ground_truth_false: passou=true (0 / 0)
- trainable_false: passou=true (0 / 0)

Os 5 canarios sao do mesmo evento e regiao, sem separacao temporal e sem o minimo de vinculos
fortes; o estado 17B permanece `17B_AINDA_SEM_PRONTIDAO_BENCHMARK_AMOSTRA_LOCAL`.

## Proximo marco recomendado

SUSC-17I: ampliar a amostra observacional para outros eventos e regioes (Curitiba, Petropolis) com
extracao direta equivalente, condicao necessaria para avaliar prontidao de benchmark, sempre
somente revisao e sem ground truth.
