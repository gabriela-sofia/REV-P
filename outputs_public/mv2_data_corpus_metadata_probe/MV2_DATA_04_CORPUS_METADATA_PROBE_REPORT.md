# MV2-DATA-04 — Corpus Metadata Probe

**Vertente B — Desbloqueio científico de dados.** Sai do canary técnico Protocolo-C
(DATA-03) e começa a mirar o **corpus oficial de 128 targets**, ainda em modo
**metadata-only**: sem raster, sem crop, sem download, sem feature espectral.

> Prova programaticamente que (1) o batch do corpus pode ser selecionado; (2) probes
> metadata-only podem ser preparados; (3) chamadas abertas sem janela temporal são
> bloqueadas; (4) OData bloqueia sem product_id; (5) o próximo dado crítico é a **janela
> temporal por target**.

## Contexto e branch

- Branch executada: `analysis/temporal-asset-readiness-mv1` (sem troca nesta sessão).
- Top commit: `67d8cfd`. Staged vazio. DATA-01/02/03 presentes.
- Config local privada: **não encontrada** (`CONFIG_MISSING`) → nenhuma rede; metadata
  bloqueada. Raster/download estão **sempre** bloqueados neste marco, independentemente de config.

## Batch do corpus (metadata-only)

15 targets, balanceado por região (até 5 cada), maior prioridade primeiro, **sem** canary
Protocolo-C e **sem** targets sem bbox/CRS:

| região | selecionados | prioridade dominante |
|---|---|---|
| Recife | 5 | P0_HIGH_SENSOR_CONFIRMED_QUERY |
| Petrópolis | 5 | P1_MEDIUM_LOCAL_CANDIDATE_REVIEW |
| Curitiba | 5 | P2_LOW_NO_LOCAL_SCENE_QUERY |

## Probes (execução atual — bloqueio limpo)

| probe | executados | bloqueio |
|---|---|---|
| GEE metadata | **0** | 15 `BLOCKED_NO_TEMPORAL_WINDOW` |
| CDSE STAC metadata | **0** | 15 `BLOCKED_NO_TEMPORAL_WINDOW` |
| CDSE OData metadata | **0** | 15 `BLOCKED_NO_PRODUCT_ID` |

Nenhuma query aberta ampla foi emitida: sem janela temporal fechada por target, GEE/STAC
**bloqueiam** em vez de consultar. OData não tem product_id/scene_id (corpus tem 0) e bloqueia.

## Resolução de lineage de corpus

15 targets → todos `CORPUS_LINEAGE_BLOCKED_NO_TEMPORAL_WINDOW`. `can_stac_dry_run=false`,
`can_raster_download_next_step=false`, `can_unlock_day10_now=false` para todos.

## Lacuna temporal (o dado crítico)

A matriz de lacuna mostra, para os 15: `has_bbox=true`, `has_crs=true`,
`has_temporal_window=false`, `has_event_date=false`, `has_export_date_hint=false`,
`has_scene_id=false`. O **template de janela temporal** foi gerado com os campos temporais
vazios (nenhuma data inventada); re-execução preserva valores preenchidos por revisão humana.

## Guardrails fail-closed (verificados)

probes executados=0 · downloads=0 · crops=0 · raster nativo=0 · features espectrais=0 ·
labels/silver/gold/negativos=0 · `corpus_day10_unlocked=false` · `sandbox_unlocked=false` ·
`can_train=false` · nenhum canary Protocolo-C no batch · T23KPR não auto-vinculado ao corpus ·
sem segredo/caminho privado/raster em `outputs_public`.

## O que falta para o próximo marco

**Preencher a janela temporal por target** (`mv2_data_04_temporal_window_template.csv`), a
partir da data do evento ou do export GEE original — nunca inventada. Com a janela presente
(+ config/credencial), os probes GEE/STAC deixam de bloquear por janela temporal e passam a
resolver `scene_id`/`datetime`/`tile`/`cloud`, abrindo o caminho (em marco futuro) para
download/crop controlado. O Dia 10 do corpus permanece bloqueado até lineage forte + raster
validado.

## Saídas públicas (leves)

config_env_preflight.csv, corpus_metadata_batch.csv, gee/stac/odata_corpus_metadata_probe.csv,
corpus_api_lineage_resolution.csv, temporal_window_gap_matrix.csv, temporal_window_template.csv,
risk_matrix.csv, summary.json, 2 relatórios .md, commands.txt.
