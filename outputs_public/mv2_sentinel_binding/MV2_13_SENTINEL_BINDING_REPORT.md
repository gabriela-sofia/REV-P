# MV2-13 — Sentinel Binding & Spectral Eligibility Gate

**Data:** 2026-06-23
**Branch executada:** `marco/validacao-label-free-evidencia-estrutural-mv1`
**Branch esperada:** `marco/mv2-12-reconstrucao-espectral-sentinel-baseline` (divergente — ver nota final)

Marco **review-only, fail-closed**. NÃO baixa raster, NÃO executa STAC, NÃO gera crop,
NÃO calcula feature espectral, NÃO cria label/silver/gold/negativo, NÃO treina, NÃO inicia sandbox.
Transforma o bloqueio "10 âncoras fortes mas sem STAC permitido" numa matriz auditável de binding.

---

## 1. Contexto MV2-10 / MV2-11 / MV2-12

- **MV2-10:** o gargalo é dado/evidência, não pipeline.
- **MV2-11:** viés regional alto (Curitiba coberta; Recife/Petrópolis subcobertas); subset balanceado mínimo é só review-only.
- **MV2-12 Data Readiness:** mapeou dados faltantes, recuperação local, download readiness, geometria de evento e raster Sentinel nativo.
- **MV2-12 Spectral Reconstruction:** catalogou **281 âncoras Sentinel**; 10 fortes, 0 médias; 0 consultas STAC; 0 itens resolvidos; 0 crops; 0 features espectrais.

## 2. Por que o gargalo atual é binding

As 281 âncoras dividem-se em **dois conjuntos disjuntos** que nunca se sobrepõem:

- **Lado espacial (128 âncoras lineage):** têm `asset_id + patch_id + bbox + crs` (zonas UTM 32722/23/25, geometria BOUNDS_ONLY) — mas **0 scene_id, 0 datetime, 0 tile, 0 cloud_cover**.
- **Lado cena (10 âncoras strong-scene):** têm `scene_id` (2 cenas Sentinel-2, tile T23KPR; 5 com datetime) — mas **0 asset_id, 0 patch_id, geometria ABSENT, 0 crs**.

Nenhuma âncora tem a cadeia completa `anchor_id + asset_id + patch_id + geometria/AOI + CRS + sensor + temporalidade + fonte`. Por isso **STAC real é estruturalmente impossível** sem recuperação: o lado espacial e o lado temporal/cena estão em âncoras diferentes. O problema não é "achar Sentinel" — é **ligar** o que já existe.

## 3. Totais

- **Âncoras processadas:** 281 | **Assets:** 128 | **Patches:** 128

### Por força de binding
| binding_strength | total |
|------------------|-------|
| BINDING_STRONG | 0 |
| BINDING_MEDIUM | 128 |
| BINDING_WEAK | 10 |
| BINDING_NONE | 141 |
| BINDING_CONFLICT | 0 |
| BINDING_INVALID | 2 |

### Por status STAC
| stac_status | total |
|-------------|-------|
| STAC_BLOCKED_NO_SCENE | 128 |
| STAC_NOT_APPLICABLE | 141 |
| STAC_BLOCKED_NO_PATCH | 10 |
| STAC_BLOCKED_INVALID | 2 |
| **STAC_DRY_RUN_ELIGIBLE** | **0** |
| **STAC_REAL_FUTURE_ELIGIBLE** | **0** |

### Por status Dia 10
| day10_status | total |
|--------------|-------|
| DAY10_BLOCKED_NO_BINDING | 143 |
| DAY10_BLOCKED_NO_SCENE_ID | 128 |
| DAY10_BLOCKED_NO_GEOMETRY | 10 |
| (desbloqueado) | 0 |

## 4. Campos faltantes mais frequentes (de 281)

| campo | ausências |
|-------|-----------|
| cloud_cover | 281 |
| tile_id | 277 |
| scene_id | 271 |
| acquisition_datetime | 243 |
| asset_id / patch_id / bbox / crs | 153 cada |

## 5. As 10 âncoras fortes reavaliadas

As 10 âncoras `STRONG_SCENE_ID` do MV2-12 carregam **scene_id forte** mas foram reclassificadas como **BINDING_WEAK** no MV2-13 porque não têm asset/patch/geometria/CRS:

- Cena `20220118T130239_20220118T130322_T23KPR` → 6 âncoras
- Cena `20220202T130251_20220202T130247_T23KPR` → 4 âncoras

Ambas são Sentinel-2 (sensor S2_MSI, confiança alta pelo formato do granule), **tile T23KPR (zona UTM 23)**.

> **Hipótese para verificação manual (não aplicada):** a zona UTM 23 do tile T23KPR coincide com a zona dos 48 patches de Petrópolis (EPSG:32723). Isso **sugere**, mas **não prova**, que essas cenas pertencem ao cohort Petrópolis. Unir cena↔patch só por tile/zona é **inferência fraca proibida**; entra na fila de recuperação como tarefa de validação humana, nunca como auto-junção.

## 6. Por que STAC real continua bloqueado

Nenhuma âncora alcança `STAC_REAL_FUTURE_ELIGIBLE` porque nenhuma é `BINDING_STRONG`. Os dois conjuntos mais próximos:

- **128 MEDIUM (lado espacial pronto):** Curitiba=43, Petrópolis=48, Recife=37. Têm patch+asset+bbox+CRS; faltam **scene_id + acquisition_datetime**. → `STAC_BLOCKED_NO_SCENE`. São a frente mais próxima de um dry-run formal.
- **10 WEAK (lado cena pronto):** têm scene_id+sensor; faltam **patch+asset+geometria+CRS**. → `STAC_BLOCKED_NO_PATCH`.

O `mv2_13_stac_dry_run_plan.csv` contém **128 rascunhos espaciais** para os MEDIUM — todos `would_download=false`, `would_create_crop=false`, `execution_status=NOT_EXECUTED_DRY_RUN_ONLY`, e explicitamente **não elegíveis** (falta referência temporal). É a demonstração da maquinaria de query sem executar nada.

## 7. O que precisa ser recuperado manualmente/local (sem download)

1. **Histórico de tasks GEE** → destrava scene_id + datetime + cloud_cover dos 128 patches espacialmente prontos (maior alavanca).
2. **Script de export GEE** e **scene_id no export** → fecha o vínculo asset→cena de forma auditável.
3. **Validar** se as 2 cenas T23KPR pertencem aos patches Petrópolis (revisão humana; sem auto-junção por tile).
4. **Resolver** os 2 datetimes futuros inválidos (`2026-*`).

Detalhe em `mv2_13_manual_recovery_queue.csv`.

## 8. O que precisa de dado externo

- **Raster Sentinel-2 L2A nativo** (bandas B02/B03/B04/B08/B11/B12 + SCL) via Copernicus/GEE — necessário para o Dia 10 e só obtenível após recuperar scene_id. Vai para diretório local-only/quarentena, nunca para `outputs_public`.

## 9. Impacto no cronograma

| Dia | Status | Efeito do MV2-13 |
|-----|--------|------------------|
| 8 | parcial | inalterado; binding não cria corpus balanceado |
| 10 | **bloqueado** | formalizado como gate testável; 0 raster nativo, 0 binding forte → `can_unlock_day10_now=false` |
| 18 | bloqueado | inalterado; evidência observacional segue ausente |
| 19 | bloqueado | inalterado; silver formal zero |
| 21 | bloqueado | inalterado; sem split treinável |
| 22 | bloqueado | inalterado; sandbox não inicia |

## 10. Guardrails confirmados

`stac_real_executed=0`, `downloads_executed=0`, `crops_created=0`, `native_rasters_created=0`,
`spectral_features_created=0`, `labels_created=0`, `silver_created=0`, `gold_created=0`,
`negatives_created=0`, `can_train=false`, `sandbox_status=bloqueado`,
`ground_truth_operational_status=ausente`, `heavy_public_outputs=0`. Binding fraco/none/conflito/inválido
nunca autoriza STAC real; PNG/NPZ/DINOv2 nunca é raster nativo; ausência nunca vira negativo;
cidade/região nunca vira label nem binding forte.

## 11. Nota de divergência de branch

A branch esperada `marco/mv2-12-reconstrucao-espectral-sentinel-baseline` está em outro worktree.
Conforme as regras, **não houve checkout/troca de branch**. As entradas do MV2-12 Spectral
Reconstruction e do lineage foram lidas como `READ_ONLY_FROM_OTHER_BRANCH` (resolução dinâmica via
`git worktree list`, sem hardcode de caminho privado). O MV2-12 Data Readiness está presente local.
Não houve colisão: o MV2-13 escreve apenas em `outputs_public/mv2_sentinel_binding/`.

## Artefatos

`mv2_13_input_discovery.csv`, `mv2_13_normalized_anchors.csv`, `mv2_13_patch_inventory_normalized.csv`,
`mv2_13_asset_inventory_normalized.csv`, `mv2_13_binding_matrix.csv`, `mv2_13_geometry_aoi_crs_matrix.csv`,
`mv2_13_temporal_scene_lineage_matrix.csv`, `mv2_13_stac_gate_matrix.csv`, `mv2_13_stac_dry_run_plan.csv`,
`mv2_13_day10_spectral_gate.csv`, `mv2_13_manual_recovery_queue.csv`, `mv2_13_binding_risk_matrix.csv`,
`mv2_13_binding_summary.json`, `MV2_13_SENTINEL_BINDING_REPORT.md`, `MV2_13_EXECUTIVE_SUMMARY.md`,
`commands.txt`. Schemas em `datasets/schemas/schema_mv2_13_*.json`.
