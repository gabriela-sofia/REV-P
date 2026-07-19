# MV2-DATA-02 — Sentinel API & Raster Access Harness

**Vertente B — Desbloqueio científico de dados.** Primeira infraestrutura REAL de acesso a
API/raster Sentinel (GEE / CDSE STAC / CDSE OData) sobre os 128 targets do MV2-DATA-01, com
guardrails fail-closed FORTES.

> Este marco **não** faz download em lote, **não** grava raster em `outputs_public`, **não**
> executa STAC real em lote, **não** cria crop/feature espectral/label/silver/gold/negativo,
> **não** treina e **não** desbloqueia o Dia 10 sem lineage forte + raster privado validado.

## Contexto e branch

- Branch executada: `analysis/temporal-asset-readiness-mv1` (troca externa herdada do fim do
  DATA-01; **nenhuma** troca feita por este marco).
- Branch esperada (linha Vertente B): `marco/mv2-data-vertente-b-desbloqueio-dados`.
- Top commit: `67d8cfd`. Staged vazio. DATA-01 outputs presentes e válidos.

## Configuração (fail-closed)

Arquivo de exemplo público (sem segredo): `mv2_data_02_api_config.example.json`. Defaults:

| Flag | Default |
|---|---|
| `gee_enabled` / `cdse_stac_enabled` / `cdse_odata_enabled` | false |
| `allow_network` | **false** |
| `allow_metadata_calls` | **false** |
| `allow_raster_download` | **false** |
| `allow_canary_download` | **false** |
| `max_download_mb` | 50 |
| `private_output_dir` | `local_only/mv2_data_api_raster_harness` (git-ignored) |
| `cdse_token_env_var` / `gee_service_account_env_var` | apenas o NOME da env var |

Credenciais **só por variável de ambiente**; nenhum token vai para arquivo público (chaves
de segredo são filtradas na leitura da config). Config privada opcional via
`REV_P_MV2_DATA_02_CONFIG` ou `local_only/.../mv2_data_02_api_config.json`.

## Providers registrados

| Provider | Metadata | Raster download | Auth | Default mode |
|---|---|---|---|---|
| GEE | sim | não | service account (env) | metadata_only |
| CDSE_STAC | sim | não | none p/ busca | metadata_only |
| CDSE_ODATA | sim | sim (canary) | bearer token (env) | metadata_only_download_opt_in |

## Execução default (offline, fail-closed)

| Métrica | Valor |
|---|---|
| total de targets | 128 |
| chamadas GEE metadata | **0** |
| chamadas STAC metadata | **0** |
| chamadas OData metadata | **0** |
| api_lineage_not_executed | 128 (resolved_strong/partial/review/conflict/not_found = 0) |
| canary candidates | 2 (âncoras oficiais PET 2022 com scene_id documentado) |
| canary executados | **0** |
| downloads | **0** · crops **0** · raster nativo **0** |
| vazamento público de raster | **0** |
| rasters privados validados | 0 → Dia 10 bloqueado |
| labels/silver/gold/negativos | 0 |
| `can_train` | false · `sandbox_status` | bloqueado |

### Candidatos canary (bloqueados por config)

Os 2 candidatos vêm do **registry oficial** (`official_anchor_sentinel_patch_registry.csv`),
não dos 128 targets: âncoras CPRM Moinho Preto / Petrópolis 2022 com scene_id Sentinel-2
documentado (`...T23KPR`, EPSG:32723, coleção `COPERNICUS/S2_SR_HARMONIZED`, bandas, cloud).
Aqui o tile T23KPR é **documentado oficialmente** — não é auto-join por zona. Mesmo com
`canary_allowed_by_data=true`, `download_allowed_now=false` e `crop_allowed_now=false`
porque a config bloqueia (fail-closed).

## Guardrails fail-closed (verificados pelo validador)

- nenhum segredo, raster ou caminho privado absoluto em `outputs_public`;
- nenhuma chamada de rede com config bloqueada; OData nunca baixa no cliente;
- `can_promote_to_bulk_download=false` em 100% das resoluções (bulk sempre proibido);
- canary só com lineage forte; executor exige `REV_P_ALLOW_RASTER_CANARY=YES` + todas as flags;
- raster só em diretório privado git-ignored; intake privado detecta vazamento;
- Dia 10 só desbloqueia com raster privado validado (0 agora).

## O que falta para o Dia 10

1. Habilitar metadados (config + credencial em env var) e recuperar lineage forte real via
   GEE/STAC/OData → `API_LINEAGE_RESOLVED_STRONG`.
2. Aprovar 1–3 alvos no plano canary e executar o download canary opt-in para diretório
   **privado** (`REV_P_ALLOW_RASTER_CANARY=YES`).
3. Validar o raster privado (bandas mínimas + SCL + checksum + CRS/geotransform).
4. Só então STAC real/crop/baseline espectral — fora deste marco.

## Saídas públicas (leves)

`mv2_data_02_api_config.example.json`, `_api_provider_registry.csv`,
`_cdse_stac_metadata_results.csv`, `_cdse_odata_metadata_results.csv`,
`_gee_metadata_results.csv`, `_api_lineage_resolution.csv`, `_raster_canary_plan.csv`,
`_raster_canary_execution_manifest.csv`, `_private_raster_validation.csv`,
`_api_raster_risk_matrix.csv`, `_summary.json`, `commands.txt`.
