# MV2-DATA-05 — Temporal Window Intake & Validation

**Vertente B — Desbloqueio científico de dados.** Cria o intake, a validação e o gate de
promoção das **janelas temporais por target** do corpus, a partir do template gerado no
MV2-DATA-04. Constrói a ponte válida
`target_id + patch_id + asset_id + bbox + CRS + temporal_window_start/end + source_ref + review_status`
que DATA-06 usará para liberar o metadata probe.

> Este marco **não** executa GEE/STAC/OData, **não** baixa raster, **não** cria crop/feature,
> **não** inventa data e **não** desbloqueia o Dia 10.

## Contexto e branch

- Branch executada: `analysis/temporal-asset-readiness-mv1` (sem troca nesta sessão).
- Top commit: `67d8cfd`. Staged vazio. DATA-01/02/03/04 presentes.
- Template preenchido privado: **não encontrado** → `NO_FILLED_TEMPLATE_FOUND`; fallback no
  template público DATA-04 (vazio) → **0 promovidas**.

## Pipeline e resultado (execução atual)

| etapa | resultado |
|---|---|
| input discovery | 4 inputs (1 template público + 3 candidatos privados ausentes) |
| linhas de entrada | 15 (batch DATA-04) |
| normalização | 15 `EMPTY` (datas vazias preservadas, nada inventado) |
| validação de evidência | 15 `TEMPORAL_EVIDENCE_EMPTY` |
| gate de promoção | 15 `TEMPORAL_WINDOW_BLOCKED_EMPTY` |
| probe-ready (GEE/STAC/OData) | 0 / 0 / 0 |
| template de correção | 15 linhas, 15 com `correction_needed=true` |

## Lógica de validação (fail-closed)

Classificação por target (`evaluate()`): **STRONG** exige target existente no corpus,
asset/patch batendo, datas ISO válidas e **não futuras**, `start<=end`, janela ≤45 dias (ou
justificada), `temporal_window_source` + `source_ref` e `review_status` aprovado. **PARTIAL**:
datas e `source_ref` válidos, faltando review final. **Rejeição** (`INVALID`/`CONFLICT`):
data futura, `start>end`, formato inválido, sem fonte, target inexistente, asset/patch
mismatch. Janela ampla demais sem justificativa → `REVIEW_REQUIRED` (nunca STRONG).

## Gate de promoção

GEE/STAC metadata probe só são habilitados com promoção **STRONG/PARTIAL**. OData **nunca** é
habilitado aqui (ainda exige `product_id`/`scene_id`). `can_unlock_day10_now=false` sempre.

## Guardrails (verificados)

template vazio → 0 promovidas; nenhuma data inventada; GEE/STAC/OData executados=0;
downloads/crops/raster/features=0; labels/silver/gold/negatives=0; `corpus_day10_unlocked=false`;
`sandbox_unlocked=false`; `can_train=false`; sem segredo/caminho privado/raster em `outputs_public`.

## O que falta para DATA-06

Depositar um **template preenchido privado** (git-ignored) em
`local_only/mv2_data_temporal_window/mv2_data_05_temporal_window_filled.csv` com a janela
temporal por target — preenchida **só com evidência** (data de evento/export, nunca inventada),
janela ≤45 dias (ou justificada) e `review_status=APPROVED`. Re-executando, as janelas
STRONG/PARTIAL são promovidas e habilitam o metadata probe do corpus em DATA-06. O Dia 10
permanece bloqueado até lineage forte + raster validado.

## Saídas públicas (leves)

input_discovery.csv, normalized_temporal_windows.csv, temporal_evidence_validation.csv,
temporal_promotion_gate.csv, probe_ready_batch.csv, temporal_window_correction_template.csv,
risk_matrix.csv, summary.json, 2 relatórios .md, commands.txt.
