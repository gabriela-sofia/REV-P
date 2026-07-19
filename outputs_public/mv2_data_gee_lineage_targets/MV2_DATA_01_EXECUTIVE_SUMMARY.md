# MV2-DATA-01 — Sumário Executivo

**Vertente B (desbloqueio científico de dados).** Os 128 bindings MEDIUM do MV2-13/MV2-14
viraram uma **fila objetiva e validada de recuperação de lineage Sentinel/GEE
metadata-only**.

## Em uma frase

Hoje cada um dos 128 alvos tem `asset_id`, `patch_id`, `bbox` e `CRS`, mas **nenhum** tem
`scene_id`, `acquisition_datetime`, `MGRS_TILE` nem `cloud_cover`; este pacote organiza —
sem baixar nada e sem chamar API real — **como** recuperar esses campos manualmente e por
consulta de metadados.

## Números

- **128** targets (Petrópolis 48 · Curitiba 43 · Recife 37)
- **0** com scene_id / datetime / tile / cloud (nada inventado)
- **48** prontos p/ lookup manual GEE · **80** prontos p/ plano de consulta metadata
- **128** linhas de template manual · **128** planos GEE · **384** planos STAC/OData
- **1536** passos de checklist · **393** linhas de risco (12/12 tipos)

## Guardrails (todos verificados)

GEE=0 · HTTP/STAC=0 · downloads=0 · crops=0 · rasters nativos=0 · features espectrais=0 ·
labels/silver/gold/negativos=0 · Dia 10 **não** desbloqueado · treino **bloqueado** ·
sandbox **bloqueado**.

## Dia 10

**Continua bloqueado.** Só destrava após recuperação de lineage real revisada
(PRODUCT_ID + datetime + tile + cloud + proveniência de export), confirmada por consulta
metadata-only e vínculo cena↔patch — nunca por tile/zona/cidade.
