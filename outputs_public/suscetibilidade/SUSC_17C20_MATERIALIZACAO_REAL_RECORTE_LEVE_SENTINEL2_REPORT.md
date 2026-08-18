# SUSC-17C20 - Materializacao real de recorte leve Sentinel-2 por AOI

## Objetivo
Sair de 'consulta de metadados' para 'artefato materializado': obter recortes leves reais de Sentinel-2 por AOI candidata a partir da cena canonica pre-evento do 17C19 (2022-04-27, tile 25MBM, S2A), com hash, manifesto e replay offline.

## Estrategia em cascata
- Caminho A (CDSE/OData oficial): so serve produto completo .SAFE/ZIP sem asset publico por banda -> registrado como `cdse_product_level_only`.
- Caminho B (fallback STAC/COG): o mesmo tile/data/plataforma foi materializado a partir do catalogo publico Earth Search (colecao sentinel-2-l2a), lendo COGs por banda por janela HTTP range no bucket sentinel-cogs, sem baixar o produto completo. Marcado como `source_role=materialization_fallback`, review-only.

## Resultado
- minimum_success_achieved: True.
- Patches com artefato leve: 5 de 5.
- Artefatos leves commitados: 15 (manifestos: 15).
- Estatisticas por banda: 80; estatisticas de indice: 60.
- Features sensoriais reais: 40.
- Produtos Sentinel completos baixados: 0; rasters pesados: 0; tiles: 0; embeddings: 0.

## Artefatos materializados
- S17C6_CANARY_REC_00001 band_stats_csv `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00001_band_stats.csv` sha256=cf759e32de74356d... 1818 bytes
- S17C6_CANARY_REC_00001 preview_rgb_png `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00001_preview.png` sha256=fec4aa382d833c4f... 14330 bytes
- S17C6_CANARY_REC_00001 materialization_stats_json `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00001_stats.json` sha256=6b2e80a710f447a4... 2159 bytes
- S17C6_CANARY_REC_00002 band_stats_csv `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00002_band_stats.csv` sha256=ac5f4af8bdc64e9e... 1818 bytes
- S17C6_CANARY_REC_00002 preview_rgb_png `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00002_preview.png` sha256=d0f565b1aacce9a0... 15412 bytes
- S17C6_CANARY_REC_00002 materialization_stats_json `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00002_stats.json` sha256=122c2088bed50ee1... 2161 bytes
- S17C6_CANARY_REC_00003 band_stats_csv `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00003_band_stats.csv` sha256=31b5a63add93f462... 1815 bytes
- S17C6_CANARY_REC_00003 preview_rgb_png `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00003_preview.png` sha256=6e90d9371f80e343... 18207 bytes
- S17C6_CANARY_REC_00003 materialization_stats_json `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00003_stats.json` sha256=47cc2bd99b4e5023... 2159 bytes
- S17C6_CANARY_REC_00004 band_stats_csv `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00004_band_stats.csv` sha256=ff7454e1173da7ad... 1818 bytes
- S17C6_CANARY_REC_00004 preview_rgb_png `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00004_preview.png` sha256=5d3ed42e0046fdbd... 15617 bytes
- S17C6_CANARY_REC_00004 materialization_stats_json `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00004_stats.json` sha256=da91b4cc5beb4894... 2161 bytes
- S17C6_CANARY_REC_00005 band_stats_csv `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00005_band_stats.csv` sha256=a482518a314a85fd... 1815 bytes
- S17C6_CANARY_REC_00005 preview_rgb_png `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00005_preview.png` sha256=46fbc6d0c5bba112... 19112 bytes
- S17C6_CANARY_REC_00005 materialization_stats_json `outputs_public/suscetibilidade/susc_17c20_light_artifacts/s17c6_canary_rec_00005_stats.json` sha256=72b2b06f9a9a4459... 2159 bytes

## Bandas e indices
- Bandas lidas: B03;B04;B08;B11 (B11 reamostrada por vizinho ao grid de 10 m).
- Indices reais: NDVI (B08,B04), NDWI (B03,B08), MNDWI (B03,B11), NDBI (B11,B08).

## Prova de pre-evento
A cena canonica e de 2022-04-27, anterior ao inicio do evento (2022-05-24). Nenhuma cena durante (2022-05-24..30) ou pos-evento foi usada como feature pre-evento.

## Por que nada vira Ground Reference
O recorte leve e o fallback STAC/COG sao materializacao sensorial review-only. Nao passam G4/G5 como evento observado, nao viram Ground Reference Candidate, ground truth ou label. O fallback nunca e usado para validar evento.

## Score v6, score v7 e 17B
Score v6 intacto, score v7 inexistente, 17B bloqueado ate existir artefato de evento com geometria e fenomeno, QA de patch candidato e revisao de politica do fallback.

## Proximo marco recomendado
SUSC-17C21 QA de patch candidato e comparacao multi-temporal pre/pos-evento das features sensoriais materializadas review-only
