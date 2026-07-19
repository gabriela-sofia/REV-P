# MV2-DATA-03 — Live Metadata Probe & Private Raster Canary

**Vertente B — Desbloqueio científico de dados.** Transição controlada do harness
fail-closed (DATA-02) para a camada REAL de API/metadados Sentinel e, opcionalmente, no
máximo **1** raster canary privado/local-only — usando exclusivamente as 2 âncoras OFICIAIS
(Protocolo-C), nunca os 128 patch do corpus.

> Mesmo que o canary funcione, ele é **prova técnica do pipeline**, não baseline do corpus.
> Este marco **não** baixa os 128 targets, **não** faz bulk download, **não** publica raster,
> **não** cria feature espectral/label/silver/gold/negativo, **não** treina e **não**
> desbloqueia o Dia 10 nem o sandbox do corpus.

## Contexto e branch

- Branch executada: `analysis/temporal-asset-readiness-mv1` (sem troca nesta sessão).
- Top commit: `67d8cfd`. Staged vazio. DATA-01 e DATA-02 presentes e válidos.

## Config local privada

**Não encontrada** (`CONFIG_MISSING`). Procurada em:
`local_only/`, `data_local/`, `private_outputs/` → `mv2_data_api_raster_harness/api_config.local.json`.
Como não há config local, **nenhuma chamada de rede** foi executada (fail-closed), mesmo que
env vars estivessem setadas. Um runbook foi gerado: `MV2_DATA_03_CONFIG_RUNBOOK.md`.

## Flags efetivas (execução atual)

| Flag | Valor |
|---|---|
| config_found | false (CONFIG_MISSING) |
| allow_network | false |
| allow_metadata_calls | false |
| allow_raster_download | false |
| allow_canary_download | false |

Credenciais só por env var; **valores nunca são impressos/gravados** (`secret_value_logged=false`).

## Candidates oficiais (TECHNICAL_CANARY_ONLY)

2 âncoras do registry oficial `official_anchor_sentinel_patch_registry.csv` (CPRM Moinho
Preto / Petrópolis 2022), scene_id Sentinel-2 documentado, EPSG:32723, coleção
`COPERNICUS/S2_SR_HARMONIZED`, bandas B02..B12, cloud documentado:

| candidate | scene_id | data | tile | nuvem |
|---|---|---|---|---|
| CC_01 (pré-evento) | `20220202T130251_..._T23KPR` | 2022-02-02 | T23KPR | 90.36% |
| CC_02 (pós-evento) | `20220304T130251_..._T23KPR` | 2022-03-04 | T23KPR | 2.39% |

`can_use_for_corpus_day10=false`; `corpus_patch_id` vazio (NÃO são patch do corpus). O tile
T23KPR vem do **scene_id documentado**, não de auto-join por zona/cidade.

## Probes e consenso (execução atual)

| Métrica | Valor |
|---|---|
| probes GEE executados | **0** |
| probes STAC executados | **0** |
| probes OData executados | **0** |
| consenso API_CONFIRMED_STRONG | 0 |
| consenso REGISTRY_ONLY_STRONG | **2** |
| conflitos | 0 |

Sem API, o lineage forte vem só do registry oficial → `REGISTRY_ONLY_STRONG` (data_ready),
mas `canary_download_allowed=false` porque a config/env bloqueiam.

## Raster canary (execução atual)

| Métrica | Valor |
|---|---|
| raster canary executado | **0** (`NOT_EXECUTED_GUARDRAIL`) |
| downloads | **0** · crops **0** · raster nativo **0** |
| vazamento público de raster | **0** |
| validação privada | `NO_RASTER_EXECUTED_OK` |
| corpus targets baixados | **0** |
| Dia 10 do corpus desbloqueado | **não** |
| sandbox desbloqueado | **não** |

Manifest com 1 entrada (máx 1), `NOT_EXECUTED_GUARDRAIL`, requisito faltante
`REV_P_ALLOW_RASTER_CANARY!=YES` (entre outros). Nenhum diretório privado foi criado.

## Matriz de impacto no corpus

8 componentes: auth/probe/private-path/checksum/band-SCL = canary valida o **mecanismo**;
corpus-lineage/Dia 10/sandbox = continuam bloqueados. Todas as linhas:
`affects_128_targets=false`, `affects_day10=false`, `affects_sandbox=false`.

## O que falta para o Dia 10 do corpus

1. Recuperar lineage forte real dos 128 (scene_id/datetime/tile/cloud) via DATA-01/02 + API.
2. Baixar/validar raster **do corpus** (não do canary oficial) em fluxo controlado.
3. Só então STAC real/crop/baseline espectral — fora deste marco.

O canary oficial valida que o caminho técnico (auth → probe → download privado → checksum →
bandas/SCL) funciona; ele **não** substitui a recuperação de lineage do corpus.

## Saídas públicas (leves)

preflight.csv, official_canary_candidates.csv, live_gee/stac/odata_metadata_probe.csv,
canary_metadata_consensus.csv, private_raster_canary_execution_manifest.csv,
private_raster_canary_validation.csv, corpus_impact_matrix.csv, risk_matrix.csv,
summary.json, CONFIG_RUNBOOK.md, commands.txt.
