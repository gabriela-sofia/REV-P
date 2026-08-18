# Dossiê multimodal

## Identificação
- Patch candidato: S17C6_CANARY_REC_00004
- Evento: REC_2022_05_24_30
- Status: scientific_review_only_not_ground_reference

## Evento e janela temporal
- Período do evento: 2022-05-24 a 2022-05-30
- Janela pré-evento: 2022-04-24 a 2022-05-23

## AOI e patch candidato
- BBox: -34.935481481,-7.986973178,-34.926498328,-7.978079592
- Patch candidato não oficial, review-only.

## Sentinel-2 multitemporal
- NDVI mediana temporal: 0.108950
- NDWI mediana temporal: -0.080286
- MNDWI mediana temporal: 0.086020
- NDBI mediana temporal: -0.152921

## CHIRPS antecedente
- CHIRPS_3d soma (mm): 20.5343
- CHIRPS_7d soma (mm): 20.5343
- CHIRPS_30d soma (mm): 139.6736

## Delta observacional pré/pós
- delta NDVI: 0.375404
- delta MNDWI: -0.494738
- Delta é mudança observacional review-only, nunca feature pré-evento nem label.

## Artefatos e hashes
- outputs_public/suscetibilidade/susc_17c25_chirps_artifacts/s17c6_canary_rec_00004_chirps_daily.csv sha256=fc90d2f82d31f455...
- outputs_public/suscetibilidade/susc_17c25_chirps_artifacts/s17c6_canary_rec_00004_chirps_source.json sha256=fcac40aa647e302f...

## Interpretação permitida
- Analisar contexto sensorial (Sentinel-2 multitemporal) e chuva antecedente (CHIRPS) como camadas review-only.
- Comparar mudança observacional pré/pós apenas como inspeção temporal.

## Interpretação proibida
- Tratar sensor ou chuva como evento observado.
- Usar como Ground Reference, ground truth, label ou treino.
- Usar delta como rótulo. Usar pós-evento como feature pré-evento.

## Lacunas G4/G5
- G4 (vínculo espacial de evento observado): ausente.
- G5 (separação de fenômeno confirmada): ausente.
- Falta artefato de evento observado com geometria e classificação de fenômeno.

## Campos necessários para Ground Reference
- event_date_or_period, observed_location, geometry_or_geocodable_address, phenomenon_class, source_name, source_hash, officiality_level.

## Próximas buscas objetivas
- busca_por_ocorrencia_oficial_com_local (fonte sugerida: defesa_civil_recife)
- busca_por_geometria_poligono_extensao (fonte sugerida: apac_pernambuco)
- busca_por_fonte_que_separa_alagamento_de_deslizamento (fonte sugerida: cemaden)

## Status final
scientific_review_only_not_ground_reference
