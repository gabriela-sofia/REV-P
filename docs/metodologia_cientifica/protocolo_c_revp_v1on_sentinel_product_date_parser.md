# Protocolo C v1on — Sentinel Product Date Parser

## Objetivo

v1on parseia datas Sentinel exclusivamente de fontes de metadado legítimas.

## Fontes aceitas

- S2A/S2B SAFE product name com timestamp (YYYYMMDDTHHMMSS)
- S1A/S1B SAFE product name com timestamp
- MTD XML: PRODUCT_START_TIME, SENSING_TIME, DATATAKE_SENSING_START, GENERATION_TIME
- STAC JSON: datetime, start_datetime, end_datetime, properties.datetime
- Sentinel product ID genérico com timestamp embutido

## Fontes explicitamente bloqueadas

- Manifest field genérico (creationdate, processingdate)
- IDs de evento REC (REC-YYYYMMDD, RECIFE_XXXXX)
- Event window (event_window_*)
- Nome derivado de patch (patch_derived_*)
- Data de modificação de arquivo (mtime)
- Data de execução de pipeline
- YYYYMMDD isolado sem contexto Sentinel

## Categorias

- PRODUCT_DATE_CONFIRMED
- PRODUCT_DATE_PROBABLE_REVIEW_ONLY
- FILENAME_DATE_CANDIDATE_ONLY
- SIDECAR_DATE_CANDIDATE_ONLY
- DERIVED_NAME_LOW_CONFIDENCE
- NO_PRODUCT_DATE
- BLOCKED_NON_SCENE_DATE

## Resultado

Total de linhas: 0. Resumo por categoria em recife_sentinel_product_date_summary_v1on.csv.
