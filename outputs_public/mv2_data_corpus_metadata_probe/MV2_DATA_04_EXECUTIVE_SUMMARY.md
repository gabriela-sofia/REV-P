# MV2-DATA-04 — Sumário Executivo

**Vertente B (desbloqueio científico de dados).** Primeiro probe metadata-only mirando o
**corpus oficial de 128 targets** (não mais o canary Protocolo-C), provando o bloqueio limpo
enquanto não houver janela temporal.

## Em uma frase

O corpus pode ser selecionado em batch balanceado e ter probes preparados, mas **sem janela
temporal por target toda consulta GEE/STAC é bloqueada** (nunca query aberta) e o OData
bloqueia sem product_id — o próximo dado crítico é a janela temporal.

## Números (execução atual)

- batch: **15** (Recife 5 · Petrópolis 5 · Curitiba 5), sem canary
- config local: **não encontrada** → metadata bloqueada; raster/download sempre bloqueados
- probes GEE/STAC/OData executados: **0 / 0 / 0**
- bloqueios: **15** por falta de janela temporal (GEE+STAC), **15** por falta de product_id (OData)
- corpus lineage: 15 `BLOCKED_NO_TEMPORAL_WINDOW`; strong/partial/review = 0
- `can_stac_dry_run`=0 · `can_raster_download_next_step`=0
- downloads/crops/rasters/features = 0 · labels/silver/gold/negatives = 0
- Dia 10 do corpus: **bloqueado** · sandbox: **bloqueado** · treino: **bloqueado**

## Guardrails

batch ≤15 e ≤5/região · nenhum canary no corpus · sem janela temporal → bloqueio (nunca
query aberta) · sem product_id → OData bloqueia · sem segredo/caminho privado/raster em
`outputs_public` · Dia 10/sandbox intactos.

## Próximo passo

Preencher `mv2_data_04_temporal_window_template.csv` com a janela temporal por target (data
de evento/export, nunca inventada). Isso é o que destrava os probes do corpus.
