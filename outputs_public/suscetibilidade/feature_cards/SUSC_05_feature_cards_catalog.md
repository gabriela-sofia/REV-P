# SUSC-05 — Catalogo de Feature Cards Cientificos

> O SUSC-05 nao calcula score, nao treina modelo e nao cria ground truth. Cada card e uma unidade metodologica explicavel. Suscetibilidade != ocorrencia confirmada.

Total de cards: **61**.

## Resumo por status

| status | nº | features |
|--------|----|----------|
| ready_for_methodology | 23 | elevation_mean, slope_mean, hand_mean, twi_mean, tpi_250m_mean, curvature_laplacian_mean, distance_to_water_mean, water_occurrence_patch, flow_acc_log_mean, flow_acc_log_p75, chirps_3d_mm, chirps_7d_mm ... |
| usable_with_caution | 5 | elevation_std, slope_std, s1_vv_mean_clean, s1_vh_mean_clean, s1_vv_minus_vh_mean_clean |
| proxy_only | 30 | urban_water_interaction, urban_drainage_interaction, proxy_v5_hand_low, proxy_v5_distance_water_low, proxy_v5_flow_accumulation, proxy_v5_twi_wetness, proxy_v5_flat_terrain, proxy_v5_low_elevation, proxy_v5_water_history, proxy_v5_rainfall_context, proxy_v5_runoff_context, proxy_v5_rain_concentration ... |
| blocked_until_recomputed | 3 | chirps_3d_to_30d_ratio, chirps_7d_to_30d_ratio, runoff_score |
| unresolved | 0 |  |

## elevation_mean  `(ready_for_methodology)`

- **Grupo:** topography · **Papel:** physical_core · **Peso:** high
- **Conceito:** Mean elevation (DEM)
- **Formula/derivacao:** Media da elevacao (DEM) sobre os pixels do patch.
- **Unidade:** m · **Resolucao/CRS:** EPSG:31983;EPSG:31985;EPSG:4326;20m · **Referencia temporal:** detected
- **Direcao esperada:** lower_increases
- **Racional:** Cotas mais baixas tendem a acumular agua e receber escoamento de montante.
- **Relacao com suscetibilidade:** Valores mais baixos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante direto no SPGAM (declividade, elevacao, orientacao de vertentes).
- **Baixo Jaguaribe/UFC:** Contexto de relevo complementar ao mapeamento de areas inundaveis.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, GEE, JRC, MDE, MDT, MERIT, PE3D, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Source likely GEE DEM or local PE3D/SGB MDE. Provenance script must be audited.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## elevation_std  `(usable_with_caution)`

- **Grupo:** topography · **Papel:** physical_core · **Peso:** low
- **Conceito:** Elevation standard deviation
- **Formula/derivacao:** Desvio-padrao da elevacao (DEM) dentro do patch.
- **Unidade:** m · **Resolucao/CRS:** EPSG:31983;EPSG:31985 · **Referencia temporal:** detected
- **Direcao esperada:** ambiguous
- **Racional:** Isolada e ambigua: baixo desvio sobre cota baixa indica planicie de acumulacao, mas o sinal depende da cota absoluta.
- **Relacao com suscetibilidade:** Evidencia complementar sem direcao monotonica; nao isolar como condicionante.
- **SPGAM/INPE:** Condicionante direto no SPGAM (declividade, elevacao, orientacao de vertentes).
- **Baixo Jaguaribe/UFC:** Contexto de relevo complementar ao mapeamento de areas inundaveis.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Direcao nao-monotonica/ambigua; nao usar com peso alto. | Same provenance concern as elevation_mean.
- **Governanca:** score_v6=False · spgam=False · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## slope_mean  `(ready_for_methodology)`

- **Grupo:** topography · **Papel:** physical_core · **Peso:** high
- **Conceito:** Mean terrain slope
- **Formula/derivacao:** Media da declividade derivada do DEM (gradiente do terreno).
- **Unidade:** degrees · **Resolucao/CRS:** EPSG:31983;EPSG:31985;EPSG:4326;20m · **Referencia temporal:** detected
- **Direcao esperada:** lower_increases
- **Racional:** Terreno plano drena devagar e favorece alagamento; alta declividade escoa rapido (mas pode gerar fluxo torrencial).
- **Relacao com suscetibilidade:** Valores mais baixos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante direto no SPGAM (declividade, elevacao, orientacao de vertentes).
- **Baixo Jaguaribe/UFC:** Contexto de relevo complementar ao mapeamento de areas inundaveis.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, GEE, JRC, MDT, MERIT, PE3D, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Derived from DEM. Spec confirmed in PROJETO/src/revp/features_topo_hydro.py.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## slope_std  `(usable_with_caution)`

- **Grupo:** topography · **Papel:** physical_core · **Peso:** low
- **Conceito:** Slope standard deviation
- **Formula/derivacao:** Desvio-padrao da declividade dentro do patch.
- **Unidade:** degrees · **Resolucao/CRS:** EPSG:31983;EPSG:31985 · **Referencia temporal:** detected
- **Direcao esperada:** ambiguous
- **Racional:** Heterogeneidade de declividade nao tem direcao monotonica clara para alagamento.
- **Relacao com suscetibilidade:** Evidencia complementar sem direcao monotonica; nao isolar como condicionante.
- **SPGAM/INPE:** Condicionante direto no SPGAM (declividade, elevacao, orientacao de vertentes).
- **Baixo Jaguaribe/UFC:** Contexto de relevo complementar ao mapeamento de areas inundaveis.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Direcao nao-monotonica/ambigua; nao usar com peso alto. | Same provenance as slope_mean.
- **Governanca:** score_v6=False · spgam=False · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## hand_mean  `(ready_for_methodology)`

- **Grupo:** topography_hydrology · **Papel:** physical_core · **Peso:** high
- **Conceito:** Height Above Nearest Drainage
- **Formula/derivacao:** Height Above Nearest Drainage: altura vertical de cada pixel acima do nivel da drenagem mais proxima; media no patch. Spec design_only em features_topo_hydro.py.
- **Unidade:** m · **Resolucao/CRS:** EPSG:31985;EPSG:4326;20m · **Referencia temporal:** detected
- **Direcao esperada:** lower_increases
- **Racional:** Areas mais baixas em relacao a drenagem tendem a maior propensao de inundacao.
- **Relacao com suscetibilidade:** Valores mais baixos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante hidro-topografico alinhado ao SPGAM (proximidade vertical/horizontal a drenagem).
- **Baixo Jaguaribe/UFC:** Apoia a delimitacao de areas baixas/inundaveis do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MDT, MERIT, PE3D, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Critical hydrological feature. Source likely GEE HAND dataset or derived from DEM+drainage. Provenance audit required.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## twi_mean  `(ready_for_methodology)`

- **Grupo:** topography_hydrology · **Papel:** physical_core · **Peso:** high
- **Conceito:** Topographic Wetness Index
- **Formula/derivacao:** Topographic Wetness Index: ln(a / tan(b)), a=area de contribuicao, b=declividade; media no patch. Spec design_only.
- **Unidade:** dimensionless · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** TWI alto indica maior potencial de saturacao e acumulo de agua.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante hidro-topografico alinhado ao SPGAM (proximidade vertical/horizontal a drenagem).
- **Baixo Jaguaribe/UFC:** Apoia a delimitacao de areas baixas/inundaveis do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MERIT, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Derived from DEM slope and contributing area. Source script must be audited.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## tpi_250m_mean  `(ready_for_methodology)`

- **Grupo:** topography · **Papel:** physical_core · **Peso:** high
- **Conceito:** Topographic Position Index (250m window)
- **Formula/derivacao:** Topographic Position Index: elevacao do pixel menos media da vizinhanca (janela 250m); media no patch.
- **Unidade:** m · **Resolucao/CRS:** unresolved · **Referencia temporal:** detected
- **Direcao esperada:** lower_increases
- **Racional:** Valores negativos indicam vales/depressoes propensas ao acumulo de agua.
- **Relacao com suscetibilidade:** Valores mais baixos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante direto no SPGAM (declividade, elevacao, orientacao de vertentes).
- **Baixo Jaguaribe/UFC:** Contexto de relevo complementar ao mapeamento de areas inundaveis.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Derived from DEM. Window size 250m documented.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## curvature_laplacian_mean  `(ready_for_methodology)`

- **Grupo:** topography · **Papel:** physical_core · **Peso:** high
- **Conceito:** Laplacian curvature
- **Formula/derivacao:** Curvatura laplaciana: segunda derivada da superficie do DEM; media no patch (negativo=concavo).
- **Unidade:** 1/m · **Resolucao/CRS:** unresolved · **Referencia temporal:** detected
- **Direcao esperada:** lower_increases
- **Racional:** Curvatura concava (negativa) indica convergencia de fluxo.
- **Relacao com suscetibilidade:** Valores mais baixos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante direto no SPGAM (declividade, elevacao, orientacao de vertentes).
- **Baixo Jaguaribe/UFC:** Contexto de relevo complementar ao mapeamento de areas inundaveis.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Derived from DEM second derivatives.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## distance_to_water_mean  `(ready_for_methodology)`

- **Grupo:** hydrology · **Papel:** hydrological_core · **Peso:** high
- **Conceito:** Mean distance to nearest drainage/water body
- **Formula/derivacao:** Distancia euclidiana de cada pixel a hidrografia/drenagem mais proxima; media no patch.
- **Unidade:** m · **Resolucao/CRS:** EPSG:31982;EPSG:31983;EPSG:31985;EPSG:32722;EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** lower_increases
- **Racional:** Distancia menor a drenagem aumenta a suscetibilidade.
- **Relacao com suscetibilidade:** Valores mais baixos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante hidrologico (distancia a drenagem, fluxo) alinhado ao SPGAM.
- **Baixo Jaguaribe/UFC:** Proximidade a agua/drenagem alinhada a logica cheia/seca do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** GEE, GlobalSurfaceWater, JRC, MERIT, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | REV-P/scripts/dino/revp_v1gq uses GeoJSON hydrography layers for distance computation.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## water_occurrence_patch  `(ready_for_methodology)`

- **Grupo:** hydrology · **Papel:** hydrological_core · **Peso:** high
- **Conceito:** Historical surface water occurrence fraction
- **Formula/derivacao:** Fracao da area do patch com ocorrencia historica de agua superficial (provavel JRC Global Surface Water).
- **Unidade:** fraction_0_to_1 · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Maior ocorrencia historica de agua indica presenca recorrente.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante hidrologico (distancia a drenagem, fluxo) alinhado ao SPGAM.
- **Baixo Jaguaribe/UFC:** Proximidade a agua/drenagem alinhada a logica cheia/seca do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, GlobalSurfaceWater, JRC, MERIT, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Source likely JRC/Google Global Surface Water dataset via GEE.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## flow_acc_log_mean  `(ready_for_methodology)`

- **Grupo:** hydrology · **Papel:** hydrological_core · **Peso:** high
- **Conceito:** Log flow accumulation (mean)
- **Formula/derivacao:** Log da acumulacao de fluxo (roteamento do DEM); media no patch.
- **Unidade:** log_cells · **Resolucao/CRS:** EPSG:31985;EPSG:4326;20m · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Maior acumulacao de fluxo indica convergencia do escoamento superficial.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante hidrologico (distancia a drenagem, fluxo) alinhado ao SPGAM.
- **Baixo Jaguaribe/UFC:** Proximidade a agua/drenagem alinhada a logica cheia/seca do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MDT, MERIT, PE3D, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Derived from DEM flow routing. Provenance script must be audited.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## flow_acc_log_p75  `(ready_for_methodology)`

- **Grupo:** hydrology · **Papel:** hydrological_core · **Peso:** high
- **Conceito:** Log flow accumulation (75th percentile)
- **Formula/derivacao:** Percentil 75 do log da acumulacao de fluxo no patch. Nome solicitado 'flow_acc_p75'.
- **Unidade:** log_cells · **Resolucao/CRS:** EPSG:31985;EPSG:4326;20m · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Percentil alto de acumulacao reforca convergencia de fluxo no patch.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Condicionante hidrologico (distancia a drenagem, fluxo) alinhado ao SPGAM.
- **Baixo Jaguaribe/UFC:** Proximidade a agua/drenagem alinhada a logica cheia/seca do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MDT, MERIT, PE3D, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Same provenance as flow_acc_log_mean.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## chirps_3d_mm  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** CHIRPS 3-day accumulated precipitation
- **Formula/derivacao:** Precipitacao acumulada CHIRPS em 3 dias relativos a reference_date.
- **Unidade:** mm · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Chuva recente intensa aumenta o gatilho hidrologico.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MERIT, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Source: CHIRPS via GEE. Temporal window relative to reference_date.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## chirps_7d_mm  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** CHIRPS 7-day accumulated precipitation
- **Formula/derivacao:** Precipitacao acumulada CHIRPS em 7 dias.
- **Unidade:** mm · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Acumulado de 7 dias representa gatilho recente sustentado.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MERIT, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Same source as chirps_3d_mm.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## chirps_30d_mm  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** CHIRPS 30-day accumulated precipitation
- **Formula/derivacao:** Precipitacao acumulada CHIRPS em 30 dias.
- **Unidade:** mm · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Umidade antecedente de 30 dias amplifica o risco.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Same source as chirps_3d_mm.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## chirps_3d_to_30d_ratio  `(blocked_until_recomputed)`

- **Grupo:** precipitation · **Papel:** exclude_until_audited · **Peso:** blocked
- **Conceito:** Requested ratio NOT present in SUSC-03
- **Formula/derivacao:** Razao solicitada CHIRPS 3d/30d. AUSENTE da matriz SUSC-03; real mais proxima rain_3d_7d_ratio (definicao diferente).
- **Unidade:** ratio · **Resolucao/CRS:** unresolved · **Referencia temporal:** unresolved
- **Direcao esperada:** not_scored
- **Racional:** Coluna solicitada inexistente na matriz migrada; exigiria recomputacao.
- **Relacao com suscetibilidade:** Nao pontuado como condicionante direto.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** unresolved
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Coluna AUSENTE da matriz SUSC-03; exige recomputacao ou uso da coluna real equivalente. | AUSENTE em SUSC-03. Coluna real mais proxima conceitualmente: rain_3d_7d_ratio (definicao diferente). Nao inventar.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## chirps_7d_to_30d_ratio  `(blocked_until_recomputed)`

- **Grupo:** precipitation · **Papel:** exclude_until_audited · **Peso:** blocked
- **Conceito:** Requested ratio NOT present in SUSC-03
- **Formula/derivacao:** Razao solicitada CHIRPS 7d/30d. AUSENTE da matriz; real mais proxima rain_7d_30d_ratio.
- **Unidade:** ratio · **Resolucao/CRS:** unresolved · **Referencia temporal:** unresolved
- **Direcao esperada:** not_scored
- **Racional:** Coluna solicitada inexistente na matriz migrada; exigiria recomputacao.
- **Relacao com suscetibilidade:** Nao pontuado como condicionante direto.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** unresolved
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Coluna AUSENTE da matriz SUSC-03; exige recomputacao ou uso da coluna real equivalente. | AUSENTE em SUSC-03. Coluna real mais proxima: rain_7d_30d_ratio. Nao inventar.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## rain_3d_7d_ratio  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** Ratio of 3-day to 7-day rainfall
- **Formula/derivacao:** Razao chuva 3 dias / 7 dias (concentracao recente).
- **Unidade:** ratio · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Razao alta indica chuva concentrada nos ultimos dias (gatilho agudo).
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Derived from chirps columns. Computation is straightforward division.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## rain_7d_30d_ratio  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** Ratio of 7-day to 30-day rainfall
- **Formula/derivacao:** Razao chuva 7 dias / 30 dias (recente vs antecedente).
- **Unidade:** ratio · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Balanco recente vs antecedente; razao alta indica concentracao recente.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Derived from chirps columns.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## rain_persistence_index  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** Rainfall persistence index
- **Formula/derivacao:** Indice composto de persistencia da chuva ao longo do tempo. Nome solicitado 'rain_persistence'. Formula a auditar.
- **Unidade:** dimensionless · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Persistencia de chuva sustenta condicao umida e aumenta o gatilho.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Composite metric. Formula must be audited.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## runoff_context_7d  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** 7-day runoff context (rainfall + terrain)
- **Formula/derivacao:** Contexto de escoamento de 7 dias combinando chuva e terreno. Formula composta a auditar.
- **Unidade:** dimensionless · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Maior contexto de escoamento de 7 dias aumenta potencial de inundacao.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MapBiomas, SRTM
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Composite metric. Provenance script must be audited.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## runoff_context_30d  `(ready_for_methodology)`

- **Grupo:** precipitation · **Papel:** rainfall_trigger · **Peso:** medium
- **Conceito:** 30-day runoff context
- **Formula/derivacao:** Contexto de escoamento de 30 dias. Formula composta a auditar.
- **Unidade:** dimensionless · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Contexto de escoamento sustentado amplifica risco.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, JRC, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Same provenance concern as runoff_context_7d.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## runoff_score  `(blocked_until_recomputed)`

- **Grupo:** precipitation · **Papel:** exclude_until_audited · **Peso:** blocked
- **Conceito:** Requested runoff score NOT present in SUSC-03
- **Formula/derivacao:** Nome solicitado inexistente na matriz; reais sao runoff_context_7d/30d.
- **Unidade:** dimensionless · **Resolucao/CRS:** unresolved · **Referencia temporal:** unresolved
- **Direcao esperada:** not_scored
- **Racional:** Nome solicitado inexistente; as colunas reais sao runoff_context_7d e runoff_context_30d.
- **Relacao com suscetibilidade:** Nao pontuado como condicionante direto.
- **SPGAM/INPE:** Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.
- **Baixo Jaguaribe/UFC:** Precipitacao/cota fluviometrica como contexto hidrologico do estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** unresolved
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Coluna AUSENTE da matriz SUSC-03; exige recomputacao ou uso da coluna real equivalente. | AUSENTE em SUSC-03 com este nome. Auditar runoff_context_* no lugar.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## s1_vv_mean_clean  `(usable_with_caution)`

- **Grupo:** sar · **Papel:** sar_support · **Peso:** low
- **Conceito:** Sentinel-1 VV backscatter (cleaned)
- **Formula/derivacao:** Media do retroespalhamento Sentinel-1 VV (limpo) no patch. Metodo de limpeza/pareamento temporal a auditar.
- **Unidade:** dB · **Resolucao/CRS:** unresolved · **Referencia temporal:** detected
- **Direcao esperada:** ambiguous
- **Racional:** Retroespalhamento radar e evidencia complementar; VV baixo pode indicar superficie lisa/umida mas nao e monotonico nem verdade de evento.
- **Relacao com suscetibilidade:** Evidencia complementar sem direcao monotonica; nao isolar como condicionante.
- **SPGAM/INPE:** Nao usado no SPGAM classico (que e optico/topografico).
- **Baixo Jaguaribe/UFC:** Nucleo do metodo: retroespalhamento Sentinel-1, limiar cheia/seca, cubo multitemporal.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, Copernicus, Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Direcao nao-monotonica/ambigua; nao usar com peso alto. | Cleaned SAR. Cleaning method and temporal matching must be audited.
- **Governanca:** score_v6=False · spgam=False · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## s1_vh_mean_clean  `(usable_with_caution)`

- **Grupo:** sar · **Papel:** sar_support · **Peso:** low
- **Conceito:** Sentinel-1 VH backscatter (cleaned)
- **Formula/derivacao:** Media do retroespalhamento Sentinel-1 VH (limpo) no patch.
- **Unidade:** dB · **Resolucao/CRS:** unresolved · **Referencia temporal:** detected
- **Direcao esperada:** ambiguous
- **Racional:** VH sensivel a estrutura/umidade; sem direcao monotonica para alagamento.
- **Relacao com suscetibilidade:** Evidencia complementar sem direcao monotonica; nao isolar como condicionante.
- **SPGAM/INPE:** Nao usado no SPGAM classico (que e optico/topografico).
- **Baixo Jaguaribe/UFC:** Nucleo do metodo: retroespalhamento Sentinel-1, limiar cheia/seca, cubo multitemporal.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, Copernicus, Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Direcao nao-monotonica/ambigua; nao usar com peso alto. | Same provenance concern as s1_vv_mean_clean.
- **Governanca:** score_v6=False · spgam=False · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## s1_vv_minus_vh_mean_clean  `(usable_with_caution)`

- **Grupo:** sar · **Papel:** sar_support · **Peso:** low
- **Conceito:** Sentinel-1 VV-VH polarization difference
- **Formula/derivacao:** Diferenca de polarizacao VV-VH. Nome solicitado 's1_vv_minus_vh'.
- **Unidade:** dB · **Resolucao/CRS:** unresolved · **Referencia temporal:** detected
- **Direcao esperada:** ambiguous
- **Racional:** Diferenca de polarizacao reflete rugosidade/umidade; nao monotonica para suscetibilidade.
- **Relacao com suscetibilidade:** Evidencia complementar sem direcao monotonica; nao isolar como condicionante.
- **SPGAM/INPE:** Nao usado no SPGAM classico (que e optico/topografico).
- **Baixo Jaguaribe/UFC:** Nucleo do metodo: retroespalhamento Sentinel-1, limiar cheia/seca, cubo multitemporal.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, Copernicus, Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Direcao nao-monotonica/ambigua; nao usar com peso alto. | Derived from s1_vv and s1_vh.
- **Governanca:** score_v6=False · spgam=False · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## ndvi_mean  `(ready_for_methodology)`

- **Grupo:** spectral_index · **Papel:** spectral_support · **Peso:** high
- **Conceito:** Normalized Difference Vegetation Index
- **Formula/derivacao:** NDVI = (NIR-RED)/(NIR+RED); media no patch. Formula confirmada em features_optical.py ndvi_spec().
- **Unidade:** dimensionless_-1_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** detected
- **Direcao esperada:** higher_decreases
- **Racional:** Mais vegetacao tende a maior permeabilidade e menor escoamento.
- **Relacao com suscetibilidade:** Valores mais altos tendem a REDUZIR a suscetibilidade.
- **SPGAM/INPE:** Indices espectrais (NDBI, NDVI) usados como condicionantes no SPGAM.
- **Baixo Jaguaribe/UFC:** Sentinel-2 (indices) usado para validacao espectral no estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, GEE, Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Formula confirmed in PROJETO/src/revp/features_optical.py ndvi_spec().
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## mndwi_mean  `(ready_for_methodology)`

- **Grupo:** spectral_index · **Papel:** spectral_support · **Peso:** high
- **Conceito:** Modified Normalized Difference Water Index
- **Formula/derivacao:** MNDWI = (GREEN-SWIR)/(GREEN+SWIR); media. Confirmada em features_optical.py mndwi_spec().
- **Unidade:** dimensionless_-1_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** MNDWI alto indica agua/umidade na superficie.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Indices espectrais (NDBI, NDVI) usados como condicionantes no SPGAM.
- **Baixo Jaguaribe/UFC:** Sentinel-2 (indices) usado para validacao espectral no estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** ANA, CHIRPS, GEE, JRC, MapBiomas, Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Formula confirmed in PROJETO/src/revp/features_optical.py mndwi_spec().
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## ndbi_mean  `(ready_for_methodology)`

- **Grupo:** spectral_index · **Papel:** spectral_support · **Peso:** high
- **Conceito:** Normalized Difference Built-up Index
- **Formula/derivacao:** NDBI = (SWIR-NIR)/(SWIR+NIR); media. Confirmada em features_optical.py ndbi_spec().
- **Unidade:** dimensionless_-1_to_1 · **Resolucao/CRS:** EPSG:32723;EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** NDBI alto indica area construida/impermeavel, reduzindo infiltracao.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Indices espectrais (NDBI, NDVI) usados como condicionantes no SPGAM.
- **Baixo Jaguaribe/UFC:** Sentinel-2 (indices) usado para validacao espectral no estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, GEE, Sentinel
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao. | Formula confirmed in PROJETO/src/revp/features_optical.py ndbi_spec().
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## urban_prop  `(ready_for_methodology)`

- **Grupo:** land_use · **Papel:** urbanization_core · **Peso:** high
- **Conceito:** Urban/built-up fraction
- **Formula/derivacao:** Fracao do patch classificada como urbano/construido. Fonte publica indeterminada (MapBiomas vs classificacao GEE).
- **Unidade:** fraction_0_to_1 · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Maior fracao urbana aumenta impermeabilizacao.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Impermeabilizacao/uso do solo: NDBI/BU e cobertura vegetal sao condicionantes no SPGAM.
- **Baixo Jaguaribe/UFC:** Uso/cobertura do solo e variavel do estudo do Baixo Jaguaribe.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, GlobalSurfaceWater, JRC, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Manifesto de proveniencia: public_source_known=false (fonte publica nao confirmada). | Source uncertain: possibly MapBiomas-derived or GEE classification. Provenance audit critical.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## vegetation_prop  `(ready_for_methodology)`

- **Grupo:** land_use · **Papel:** urbanization_core · **Peso:** high
- **Conceito:** Vegetation cover fraction
- **Formula/derivacao:** Fracao do patch com cobertura vegetal. Mesma fonte indeterminada de urban_prop.
- **Unidade:** fraction_0_to_1 · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_decreases
- **Racional:** Mais vegetacao indica maior permeabilidade, reduzindo escoamento.
- **Relacao com suscetibilidade:** Valores mais altos tendem a REDUZIR a suscetibilidade.
- **SPGAM/INPE:** Impermeabilizacao/uso do solo: NDBI/BU e cobertura vegetal sao condicionantes no SPGAM.
- **Baixo Jaguaribe/UFC:** Uso/cobertura do solo e variavel do estudo do Baixo Jaguaribe.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, GlobalSurfaceWater, JRC, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Manifesto de proveniencia: public_source_known=false (fonte publica nao confirmada). | Same provenance concern as urban_prop.
- **Governanca:** score_v6=True · spgam=True · dino=True · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## urban_water_interaction  `(proxy_only)`

- **Grupo:** interaction · **Papel:** urbanization_core · **Peso:** medium
- **Conceito:** Urban x water proximity interaction
- **Formula/derivacao:** Termo composto v5: proporcao urbana combinada com proximidade/ocorrencia de agua. Formula a auditar.
- **Unidade:** dimensionless · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Area urbana proxima a agua combina impermeabilizacao e proximidade hidrica.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Termo composto que combina condicionantes do SPGAM (urbanizacao x agua/drenagem).
- **Baixo Jaguaribe/UFC:** Combina uso do solo e agua, ambos presentes no estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Termo composto v5; auditar formula antes de pesar. | Manifesto de proveniencia: public_source_known=false (fonte publica nao confirmada). | Composite interaction. Formula must be audited.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## urban_drainage_interaction  `(proxy_only)`

- **Grupo:** interaction · **Papel:** urbanization_core · **Peso:** medium
- **Conceito:** Urban x drainage proximity interaction
- **Formula/derivacao:** Termo composto v5: proporcao urbana combinada com proximidade a drenagem. Formula a auditar.
- **Unidade:** dimensionless · **Resolucao/CRS:** EPSG:4326 · **Referencia temporal:** detected
- **Direcao esperada:** higher_increases
- **Racional:** Area impermeavel proxima a drenagem combina dois condicionantes.
- **Relacao com suscetibilidade:** Valores mais altos tendem a AUMENTAR a suscetibilidade.
- **SPGAM/INPE:** Termo composto que combina condicionantes do SPGAM (urbanizacao x agua/drenagem).
- **Baixo Jaguaribe/UFC:** Combina uso do solo e agua, ambos presentes no estudo.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** CHIRPS, MapBiomas
- **Limitacoes:** Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04). | Termo composto v5; auditar formula antes de pesar. | Manifesto de proveniencia: public_source_known=false (fonte publica nao confirmada). | Composite interaction. Formula must be audited.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_hand_low  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: HAND below threshold indicating physical flood predisposition.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Threshold and derivation from score v5 system. Not GT.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_distance_water_low  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: distance to water below threshold.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_flow_accumulation  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: high flow accumulation.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_twi_wetness  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: high TWI indicating wet terrain.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_flat_terrain  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: flat terrain prone to water accumulation.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_low_elevation  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: low absolute elevation.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_water_history  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: historical water occurrence above threshold.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_rainfall_context  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: significant rainfall context.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_runoff_context  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: significant runoff context.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_rain_concentration  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: rainfall concentrated in recent days.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_rain_persistence  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: persistent rainfall over time.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_urban_exposure  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: high urban exposure.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_vegetation_low  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: low vegetation cover.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_urban_water_interaction  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: urban area near water.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_urban_drainage_interaction  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: urban area near drainage.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_ndbi_built  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: high NDBI indicating built-up area.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_ndvi_low  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: low NDVI indicating lack of vegetation.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## proxy_v5_mndwi_wet  `(proxy_only)`

- **Grupo:** proxy_v5 · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Binary flag: high MNDWI indicating wet surface.
- **Formula/derivacao:** Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Part of score v5 proxy system.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## score_predisposicao_hidrotopografica_v5  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Composite hydrotopographic predisposition score. Combines elevation, slope, HAND, TWI, TPI, curvature, distance to water, flow accumulation.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Heuristic composite. Not a feature (it is derived from features). Not GT.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## score_gatilho_hidroclimatico_v5  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Hydroclimatic trigger score. Combines rainfall intensity, persistence, and runoff context.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Heuristic composite. Not GT.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## score_amplificacao_urbana_v5  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Urban amplification score. Combines urban proportion, impermeabilization proxies, and interaction terms.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Heuristic composite. Not GT.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## score_superficie_optica_v5_diagnostic  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Optical surface diagnostic score. Combines NDVI, MNDWI, NDBI indicators.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Heuristic composite. Diagnostic only.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## score_umidade_antecedente_v5  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Antecedent moisture score.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Heuristic composite. Not GT.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## score_impulso_chuva_v5  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Rainfall impulse score.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Heuristic composite. Not GT.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## score_evento_enchente_potencial_v5_core  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Core potential flood event score combining all sub-scores.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Top-level heuristic. NOT ground truth. NOT validated against observed events.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## label_evento_enchente_potencial_v5_core_regional_p75  `(proxy_only)`

- **Grupo:** heuristic_label · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Heuristic binary label: 1 if score_evento_enchente_potencial_v5_core exceeds regional 75th percentile. THIS IS NOT GROUND TRUTH.
- **Formula/derivacao:** Label heuristico derivado de score por limiar (NAO ground truth).
- **Unidade:** binary_0_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | CRITICAL: This is a heuristic label derived from composite scores, NOT from observed flood events. Must NEVER be treated as ground truth for supervised training.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## label_confidence_v5  `(proxy_only)`

- **Grupo:** heuristic_label · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Confidence level of the heuristic label.
- **Formula/derivacao:** Label heuristico derivado de score por limiar (NAO ground truth).
- **Unidade:** score_0_to_1 · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Describes confidence in the heuristic, not in reality.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True

## regime_evento_enchente_v5  `(proxy_only)`

- **Grupo:** score · **Papel:** proxy_only · **Peso:** not_weighted
- **Conceito:** Categorical regime classification of flood event potential.
- **Formula/derivacao:** Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).
- **Unidade:** unknown · **Resolucao/CRS:** unresolved · **Referencia temporal:** reference_date_2022-12-31
- **Direcao esperada:** not_scored
- **Racional:** Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.
- **Relacao com suscetibilidade:** Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.
- **SPGAM/INPE:** Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.
- **Baixo Jaguaribe/UFC:** Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.
- **DINO:** DINOv2 compara padroes visuais/espaciais latentes como camada complementar de generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO e ground truth.
- **Fontes (candidatas):** score_v5_system
- **Limitacoes:** Heuristico, NAO ground truth. | Derivado de scores/limiar regional (p75), nao de evento observado. | Heuristic classification. Not GT.
- **Governanca:** score_v6=False · spgam=False · dino=False · ground_truth=False · training=False · review_only=True · requires_manual_review=True
