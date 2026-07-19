# MV2-14 — GEE Scene Lineage Recovery & Temporal Metadata Binding

**Data:** 2026-06-23
**Branch executada:** `marco/validacao-label-free-evidencia-estrutural-mv1`
**Branch esperada:** `marco/mv2-12-reconstrucao-espectral-sentinel-baseline` (divergente — ver nota final)

Marco **review-only, fail-closed**. NÃO baixa raster, NÃO executa STAC, NÃO cria crop,
NÃO calcula feature espectral, NÃO cria label/silver/gold/negativo, NÃO treina.
Ataca o gargalo do MV2-13: os 128 bindings com lado espacial pronto e tempo/cena ausente.

---

## 1. Contexto MV2-10 → MV2-13

- **MV2-13** provou a separação estrutural: 128 bindings MEDIUM têm `asset_id+patch_id+bbox+CRS`
  mas faltam `scene_id+datetime+tile+cloud_cover`; 10 anchors WEAK têm cena Sentinel mas sem patch/asset.
- **MV2-14** tenta recuperar lineage temporal/cena dos 128 MEDIUM a partir de fontes locais
  (histórico/scripts GEE, scene searches, registries, manifests), sem baixar raster nem STAC.

## 2. O que foi descoberto sobre as fontes GEE locais

Existem fontes GEE/export reais **localmente** — mas pertencem ao **track de âncora oficial
do Protocolo C** (Petrópolis CPRM, tile **T23KPR**, `COPERNICUS/S2_SR_HARMONIZED`), com lineage
completo (scene_id, scene_date, cloud_cover, bandas, CRS). Fontes principais:
`local_runs/protocolo_c/v1iv|v1ix|v1iz/*scene_search*.csv`, `*_gee_export_plan.py`,
`datasets/official_anchor_sentinel_patch_*.csv`.

**Achado central:** essas fontes estão num **namespace diferente** do corpus. Os candidatos de
cena usam `reference_patch_id` (REFPATCH_*/ANCHOR_*), **não** os `patch_id` do corpus
(`CUR_00038`, `PET_00360`, …). E **todas as 11 cenas locais são T23KPR (zona 23 / Petrópolis)** —
**zero** cenas para Curitiba (zona 22) ou Recife (zona 25).

## 3. Totais

- **Bindings MEDIUM (alvos):** 128 — Curitiba 43, Petrópolis 48, Recife 37
- **Fontes GEE/export descobertas:** 841 (819 locais; 22 lidas read-only de outro worktree)
- **Candidatos de lineage extraídos:** 111 (89 estruturados em CSV; 22 por regex textual; 11 cenas distintas, todas T23KPR)

### Por status de lineage
| lineage_status | total | quem |
|----------------|-------|------|
| LINEAGE_RECOVERED_STRONG | 0 | — |
| LINEAGE_RECOVERED_PARTIAL | 0 | — |
| LINEAGE_CANDIDATE_REVIEW | 48 | Petrópolis (cenas T23 locais existem, mas vínculo por região+zona é só hipótese de revisão humana) |
| LINEAGE_CONFLICT | 0 | — |
| LINEAGE_INVALID | 0 | — |
| LINEAGE_NOT_FOUND | 80 | Curitiba (43) + Recife (37) — nenhuma cena local para zonas 22/25 |

## 4. As 10 cenas com scene_id (T23KPR) — revisão sem auto-join

As 10 âncoras WEAK do MV2-13 (2 cenas distintas: `20220118…T23KPR`, `20220202…T23KPR`) foram
enriquecidas com proveniência local (`COPERNICUS/S2_SR_HARMONIZED`, datas, cloud). **auto_join_allowed=false
para todas as 10** — coincidência de tile/zona com Petrópolis é hipótese, não vínculo.
**auto_joins_performed = 0.** Ver `mv2_14_strong_scene_anchor_review.csv`.

## 5. Prontidão STAC e Dia 10 após lineage

- `READY_FOR_STAC_DRY_RUN` = **0** ; `READY_FOR_STAC_REAL_REVIEW_REQUIRED` = **0**.
  (Nenhum binding alcançou LINEAGE_RECOVERED_STRONG porque nenhum candidato referencia o `patch_id`
  do corpus explicitamente.)
- **Dia 10 não desbloqueado:** `can_unlock_day10_now=false` em todos (0 raster nativo); `can_unlock_day10_next_step` = 0.

## 6. Por que STAC real e Dia 10 seguem bloqueados

O lineage que existe localmente está num namespace de âncora (Protocolo C), não no namespace
do corpus de 128 patches. Não se pode juntar por tile/zona (proibido). Logo:
- **Petrópolis (48):** as cenas existem localmente — falta confirmar **humano por patch** quais cenas
  correspondem a quais patches do corpus.
- **Curitiba (43) + Recife (37):** falta recuperar o **export GEE próprio** desses patches (sem cena local).

## 7. O que precisa ser feito manualmente no GEE

Ver `mv2_14_gee_manual_recovery_queue.csv`. Ações P0:
1. Abrir o **histórico de tasks GEE** e localizar os exports dos patches do corpus (CUR/PET/REC).
2. Abrir o **script de export** e ler `COLLECTION`, `filterDate`, bandas.
3. Recuperar **scene_id/PRODUCT_ID** (`system:index`/`PRODUCT_ID`) por patch.
4. Recuperar **acquisition_datetime** real (não a data de processamento), **cloud_cover** (`CLOUDY_PIXEL_PERCENTAGE`) e **tile** (`MGRS_TILE`).
5. Exportar **apenas metadado leve** — nunca raster.

## 8. Campos a preencher no template manual

`mv2_14_manual_lineage_template.csv` traz 1 linha por patch do corpus (asset_id/patch_id já preenchidos,
fatos conhecidos) e os campos a completar após checagem no GEE, todos **vazios por design**:
`scene_id`, `acquisition_datetime`, `tile`, `cloud_cover`, `sensor`, `platform`, `gee_collection`,
`gee_export_task`, `gee_script_path_or_name`, `evidence_url_or_local_ref`, `filled_by`, `reviewed_by`.
`review_status=PENDING_GEE_CHECK`. **Não preencher dados inventados.**

## 9. Impacto no cronograma

| Dia | Status | Efeito do MV2-14 |
|-----|--------|------------------|
| 8 | parcial | inalterado |
| 10 | **bloqueado** | lineage formalizado; ainda 0 raster nativo e 0 binding forte |
| 18 | bloqueado | inalterado (evidência observacional) |
| 19 | bloqueado | inalterado (silver formal zero) |
| 21 | bloqueado | inalterado (sem split treinável) |
| 22 | bloqueado | inalterado (sandbox não inicia) |

## 10. Guardrails confirmados

`stac_real_executed=0`, `downloads_executed=0`, `crops_created=0`, `native_rasters_created=0`,
`spectral_features_created=0`, `labels_created=0`, `silver_created=0`, `gold_created=0`,
`negatives_created=0`, `auto_joins_performed=0`, `can_train=false`, `can_unlock_day10_now=false`,
`heavy_public_outputs=0`. Ausência (80 NOT_FOUND) nunca vira negativo; cidade/região nunca vira
vínculo (no máximo CANDIDATE_REVIEW); datetime futuro descartado; PNG/NPZ/render nunca é raster nativo.

## 11. Nota de divergência de branch e leitura cross-worktree

A branch esperada está em outro worktree; **não houve checkout/troca de branch**. As entradas do
MV2-12 Spectral Reconstruction / lineage que não estão locais foram lidas como
`READ_ONLY_FROM_OTHER_WORKTREE`. MV2-13 e MV2-12 Data Readiness estão presentes localmente.
Nenhuma colisão: o MV2-14 escreve apenas em `outputs_public/mv2_gee_lineage_recovery/`.

## Artefatos

`mv2_14_gee_source_discovery.csv`, `mv2_14_lineage_candidates.csv`, `mv2_14_medium_binding_index.csv`,
`mv2_14_lineage_binding_matrix.csv`, `mv2_14_strong_scene_anchor_review.csv`,
`mv2_14_post_lineage_stac_gate.csv`, `mv2_14_day10_post_lineage_gate.csv`,
`mv2_14_gee_manual_recovery_queue.csv`, `mv2_14_manual_lineage_template.csv`,
`mv2_14_lineage_risk_matrix.csv`, `mv2_14_lineage_summary.json`,
`MV2_14_GEE_LINEAGE_RECOVERY_REPORT.md`, `MV2_14_EXECUTIVE_SUMMARY.md`, `commands.txt`.
Schemas em `datasets/schemas/schema_mv2_14_*.json`.
