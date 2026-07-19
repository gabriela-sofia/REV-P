# SUSC-17F Extracao e consolidacao de atributos fisicos/topograficos dos canarios observacionais

## Estado herdado do 17E

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `f9000b9`
- Status final 17E herdado: 17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY
- Itens em calibracao exploratoria review-only: 5
- Status 17B herdado: 17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK
- `score_v6` alterado: False
- `score_v7` criado: False

## Problema: ausencia do grupo fisico/topografico

Os 5 canarios `S17C6_CANARY_REC_00001..00005` tinham espectral e chuva reais, mas nenhuma
feature fisica/topografica (HAND, elevacao, declividade, TWI, acumulacao de fluxo, proximidade
hidrica). Esse grupo domina o score_v6 (peso 0.4) e sua ausencia bloqueava a
calibracao forte.

## Inventario das fontes encontradas

- S17F_FONTE_0001 (features_oficiais_por_patch): status_uso=referencia_comparativa; datasets/suscetibilidade/susc_features_by_patch_v1.csv
- S17F_FONTE_0002 (topografia_recomputada_dem): status_uso=referencia_comparativa; outputs_public/suscetibilidade/susc_17c38_official_patch_basin_aware_topography_features.csv
- S17F_FONTE_0003 (terreno_canario_ancorado_evento): status_uso=referencia_comparativa; outputs_public/suscetibilidade/susc_17c34_terrain_features.csv
- S17F_FONTE_0004 (manifesto_dem_local): status_uso=insuficiente; outputs_public/suscetibilidade/susc_17c36_dem_artifact_manifest.csv
- S17F_FONTE_0005 (drenagem_oficial_por_patch): status_uso=referencia_comparativa; outputs_public/suscetibilidade/susc_17c35_official_drainage_features.csv
- S17F_FONTE_0006 (feature_store_fisico_direto_canario_S17C6): status_uso=ausente; nao_aplicavel

Nenhuma fonte oficial cobre a geometria dos canarios S17C6; a janela DEM local do 17C36 fica ao
sul (ate -8.01855) e nao alcanca a AOI dos canarios. Portanto nao ha feature fisica direta, e
sim referencia comparativa e necessidade de extracao.

## Resultado por canario

- S17C6_CANARY_REC_00002: modo=referencia_comparativa_review_only; referencia=recife_00552 a 1967.5000 m; forte_agora=false
- S17C6_CANARY_REC_00001: modo=referencia_comparativa_review_only; referencia=recife_00552 a 984.4000 m; forte_agora=false
- S17C6_CANARY_REC_00003: modo=referencia_comparativa_review_only; referencia=recife_00552 a 2791.7000 m; forte_agora=false
- S17C6_CANARY_REC_00004: modo=referencia_comparativa_review_only; referencia=recife_00552 a 3112.4000 m; forte_agora=false
- S17C6_CANARY_REC_00005: modo=referencia_comparativa_review_only; referencia=recife_00552 a 3553.7000 m; forte_agora=false

- Feature fisica direta: 0
- Referencia comparativa consolidada: 5
- Fila de extracao gerada: 5
- Calibracao exploratoria aprimorada: 5
- Pode calibracao forte agora: 0

## Features diretas encontradas

Nenhuma feature fisica direta dos canarios S17C6 foi encontrada no repositorio.

## Referencias comparativas encontradas

O patch oficial `recife_00552` tem features fisicas reais (oficiais e recomputadas por DEM no
17C38), a distancias de aproximadamente 984 a 3554 m dos canarios, sem sobreposicao. Uso
estritamente comparativo review-only, com penalidade de incerteza pela distancia.

## Lacunas restantes

- HAND, elevacao, declividade, TWI, acumulacao de fluxo e proximidade hidrica diretas dos
  canarios seguem ausentes.
- A referencia comparativa e distante e nao habilita calibracao forte.

## Simulacao exploratoria aprimorada

- HF0_sem_componente_fisico: low=4, medium=1
- HF2_referencia_comparativa_penalizada: low=1, medium=4
- HF4a_sensibilidade_topografia_leve: low=4, medium=1
- HF4c_sensibilidade_topografia_forte: medium=5

O componente fisico comparativo, penalizado pela distancia, eleva o score exploratorio em
direcao a classe de referencia, mais para os canarios mais proximos. Em todas as linhas:
score_oficial=false, substituir_score_v6=false, usar_em_treino=false, ground_truth=false.

## Impacto na prontidao de calibracao

A calibracao exploratoria foi aprimorada com um componente fisico comparativo, mas a calibracao
forte permanece bloqueada porque a referencia fisica e comparativa e distante. O caminho para a
calibracao forte e executar a fila de extracao das features fisicas diretas.

## Gate final 17F

- canarios_avaliados: passou=true (5 / 5)
- feature_fisica_direta: passou=true (0 / >=0)
- referencia_comparativa: passou=true (5 / >=0)
- fila_extracao: passou=true (5 / 5)
- calibracao_exploratoria_aprimorada: passou=true (5 / >=1)
- pode_calibracao_forte_agora: passou=true (0 / >=0)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_allowed_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)
- caminho_funcional_entregue: passou=true (true / true)
- status_final_17f: passou=true (17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO / enum)

- Status final: **17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO**
- Caminho funcional: **calibracao_exploratoria_aprimorada_mais_fila_extracao**

## Impacto no 17E e no 17B

- 17E: 17E_CALIBRACAO_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO_COMPARATIVO (a calibracao exploratoria ganhou componente fisico comparativo).
- 17B: 17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK (segue sem prontidao de benchmark: sem ground truth, amostra
  concentrada, sem score_v7).

## Proximo marco recomendado

SUSC-17G: executar a fila de extracao (`fila_extracao_features_fisicas_pendentes.csv`) para
obter HAND/elevacao/declividade/TWI/acumulacao de fluxo e proximidade hidrica diretas dos
canarios sobre nova janela DEM Copernicus que cubra a AOI, habilitando a reavaliacao de
calibracao forte review-only.
