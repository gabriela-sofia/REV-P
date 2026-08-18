# SUSC-17C35 - Feature Parity, HAND/Hidrografia/Landcover e Readiness de Replay Score v6

## Objetivo
Ponte review-only para comparabilidade entre canarios (17C34) e patches oficiais do score v6, usando a MESMA politica de features e normalizacao POPULACIONAL. Nao cria v7, nao altera score v6, nao normaliza canarios por eles mesmos.

## Contrato + normalizacao
- Contrato da formula v6 criado (pesos topo0.40/rain0.25/urban0.20/veg-0.10/evid0.05; winsorize(0.01,0.99)+robust_minmax populacional).
- Populacao de normalizacao RECONSTRUIDA da matriz oficial: True (19 features com bounds recuperados).

## Aquisicao oficial
- Hidrografia/drenagem: 7 tentativas; drenagem OFICIAL disponivel: 11; proxy OSM: 0.
- HAND: 11 tentativas; HAND proxy local: 11; HAND full: 0.
- Landcover: 3 tentativas; landcover OFICIAL disponivel: 11 (Dados Recife cobertura-da-terra: construida/fv/agua); built-up proxy NDBI: 11.

## Paridade e replay
- Feature parity matrix: 11 linhas; familias disponiveis: 6; paridade media: 0.4286.
- Score v6 full replay computavel: False; partial component replay: True.
- Score v6 review-only rows: 11; scores finais computados: 0.

## Resposta cientifica (Resultado B)
A normalizacao populacional foi reconstruida e drenagem+landcover OFICIAIS foram adquiridos (distance_to_water e urban_prop/vegetation_prop oficiais). Os componentes vegetation_mitigation e urban_spectral sao computaveis de forma comparavel review-only. Porem o sub-indice topography_hydrology (peso 0.40, dominante) exige twi/tpi/flow_acc e HAND full, ausentes -> o SCORE FINAL v6 permanece NAO computavel. Nenhum numero de score final enganoso e emitido. HAND fica como proxy local (nao full); OSM fica como proxy (nao oficial); score v6 oficial intacto; sem v7/GT/treino.

## Guardrails
- Score v6 oficial intacto (hash before=after); sem v7; sem GT/treino; ocorrencia nao e feature; sem pos-evento como pre-evento; PROIBIDO normalizar so nos canarios (usada populacao oficial); HAND proxy != HAND full; landcover oficial separado de NDBI proxy; OSM so proxy; controle nao vira negativo verdadeiro.

## minimum_success_achieved: True | result_class: B_normalization_reconstructed_topography_component_blocked

## Proximo marco recomendado
SUSC-17C36 Pipeline hidrologico local (DEM+drenagem oficial) para TWI/TPI/flow_acc/HAND full nos canarios e completar o sub-indice topography_hydrology para um replay v6 comparavel review-only; manter score v6 oficial intacto e 17B fail-closed.
