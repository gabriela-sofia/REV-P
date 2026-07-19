# MV2-DATA-01 — GEE Lineage Target Pack

**Vertente B — Desbloqueio científico de dados.** Primeira infraestrutura programática
para recuperar o *lineage* Sentinel/GEE dos 128 bindings MEDIUM identificados por
MV2-13/MV2-14, de forma **metadata-only**.

> Este marco **não** baixa raster, **não** executa GEE/STAC/OData real, **não** cria crop,
> **não** calcula feature espectral, **não** cria label/silver/gold/negativo e **não**
> desbloqueia o Dia 10.

## Contexto e divergência de branch

- Branch executada: `marco/validacao-label-free-evidencia-estrutural-mv1`
- Branch esperada (linha MV2 Vertente B): `marco/mv2-data-vertente-b-desbloqueio-dados`
- **Divergência registrada**: o trabalho foi feito na branch atual (não houve troca de
  branch, conforme regra do marco). O `summary.json` marca `divergencia_branch=true`.

## Entradas

| Arquivo | Origem | Uso |
|---|---|---|
| `outputs_public/mv2_gee_lineage_recovery/mv2_14_medium_binding_index.csv` | LOCAL | 128 bindings MEDIUM (fonte primária) |
| `outputs_public/mv2_gee_lineage_recovery/mv2_14_lineage_binding_matrix.csv` | LOCAL | lineage recuperado (todos `recovered_*` vazios) |

Estado de entrada: **0** scene_id / datetime / tile / cloud recuperados; o lado espacial
(asset_id, patch_id, bbox, CRS) está pronto para todos os 128.

## O que existe hoje (por binding)

- `asset_id`, `patch_id`, `bbox`, `crs` — **presentes** nos 128.

## O que falta (gargalo de lineage)

- `scene_id` / `PRODUCT_ID`, `acquisition_datetime`, `MGRS_TILE`, `cloud_cover`,
  coleção GEE, proveniência de export (task/script) e evidência de vínculo GEE/export.

## Resultado — fila objetiva de recuperação

| Métrica | Valor |
|---|---|
| total de targets | **128** |
| com asset_id / patch_id / bbox / crs | 128 / 128 / 128 / 128 |
| com scene_id / datetime / tile / cloud_cover | 0 / 0 / 0 / 0 |
| prontos p/ lookup manual GEE | 48 (Petrópolis, cenas candidatas locais) |
| prontos p/ plano de consulta metadata | 80 (Recife 37 + Curitiba 43) |
| bloqueados / inválidos | 0 / 0 |
| linhas no template manual | 128 |
| planos GEE metadata-only | 128 |
| planos STAC/OData metadata-only | 384 (128 × 3 providers) |
| passos no checklist manual | 1536 (12/target) |
| linhas na matriz de risco | 393 (12/12 tipos obrigatórios) |

### Prioridade por região e completude

| Prioridade | Qtd | Região | Justificativa |
|---|---|---|---|
| `P0_HIGH_SENSOR_CONFIRMED_QUERY` | 37 | Recife | sensor S2_PROBABLE confirmado + geometria pronta |
| `P1_MEDIUM_LOCAL_CANDIDATE_REVIEW` | 48 | Petrópolis | cenas candidatas locais para revisão manual |
| `P2_LOW_NO_LOCAL_SCENE_QUERY` | 43 | Curitiba | sem cena local e sensor desconhecido |

## Guardrails fail-closed (verificados)

- chamadas GEE: **0** · chamadas HTTP/STAC/OData: **0** · downloads: **0**
- crops: **0** · rasters nativos: **0** · features espectrais: **0**
- labels / silver / gold / negativos: **0**
- `can_unlock_day10_now`: **false** · `can_train`: **false** · `sandbox_status`: **bloqueado**
- nenhum `scene_id`/`datetime`/`tile`/`cloud` inventado (template vazio nesses campos)
- nenhum tile auto-vinculado (T23KPR não é join automático; só hipótese registrada como risco)
- cidade/região nunca estabelece lineage forte
- todos os planos: `would_call_gee` / `would_execute_http` / `would_download` = false;
  `execution_status = NOT_EXECUTED_PLAN_ONLY`

## O que falta para o Dia 10

O Dia 10 (STAC real → crop privado → baseline espectral) permanece **bloqueado**. Para
desbloquear, em revisão humana e nesta ordem:

1. Preencher o template manual com evidência GEE real (PRODUCT_ID, datetime, tile, cloud,
   coleção, proveniência de export) — sem inferência.
2. Executar os planos de consulta **metadata-only** (GEE/STAC/OData) para confirmar o
   `scene_id` e o `acquisition_datetime` de cada patch.
3. Estabelecer o vínculo cena↔patch revisado (lineage forte), nunca por tile/zona.
4. Só então avaliar STAC real, crop e baseline espectral — fora deste marco.

## Saídas

- `mv2_data_01_lineage_targets.csv`
- `mv2_data_01_manual_gee_lineage_template.csv`
- `mv2_data_01_gee_metadata_query_plan.csv`
- `mv2_data_01_stac_odata_metadata_plan.csv`
- `mv2_data_01_manual_recovery_checklist.csv`
- `mv2_data_01_risk_matrix.csv`
- `mv2_data_01_summary.json`
- `commands.txt`
