# Protocolo C v1om — Sentinel Sidecar Discovery

## Objetivo

v1om escaneia diretórios locais em busca de sidecars Sentinel (SAFE, MTD XML, STAC JSON, manifest.safe, .dim, .txt). Nunca abre raster para pixels. Extrai apenas datas de fontes de metadado legítimas.

## Fontes aceitas

- MTD_MSIL*.xml / MTD_TL.xml: campos PRODUCT_START_TIME, SENSING_TIME, DATATAKE_SENSING_START, etc.
- STAC JSON / GeoJSON: campos datetime, start_datetime, end_datetime, properties.datetime.
- Nome de produto SAFE: S2A_MSIL1C_YYYYMMDDTHHMMSS_... / S1A_IW_GRDH_..._YYYYMMDDTHHMMSS_...
- ID de produto Sentinel com timestamp embutido.

## Fontes bloqueadas

- MANIFEST_FIELD genérico
- IDs de evento REC
- Janela temporal de evento
- Nome derivado de patch
- Data de criação/modificação de arquivo
- Data de execução de pipeline
- YYYYMMDD isolado sem contexto Sentinel

## Vínculo patch->asset

Este script não resolve vínculo patch-asset. Os candidatos aqui são encaminhados para v1on (parser) e v1oo (resolver v3).

## Guardrails

- Nenhum pixel lido.
- Nenhum label criado.
- can_unlock_temporal não é setado aqui.
- Nenhum path absoluto nos outputs.
