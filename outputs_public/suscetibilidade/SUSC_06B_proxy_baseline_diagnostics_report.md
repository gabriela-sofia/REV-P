# SUSC-06B — Diagnóstico Científico do Baseline Proxy

> O SUSC-06B não valida ocorrência real de enchente. Ele diagnostica a consistência interna de um baseline ajustado contra proxy heurístico, registra riscos de circularidade e prepara a transição para validação documental/espacial em SUSC-07.

Tags do resultado: `review_only`, `proxy_based`, `not_event_validation`.

---

## 1. Objetivo do SUSC-06B

Criar uma camada formal de diagnóstico sobre o baseline SUSC-06A para impedir leitura equivocada dos resultados: documentar a circularidade do target, reenquadrar o R² alto como recuperabilidade do heurístico, auditar as divergências de direção, checar estabilidade regional, classificar a elegibilidade de cada feature e registrar a prontidão para o overlay documental/evento do SUSC-07.

## 2. O que o SUSC-06A fez

Ajustou um baseline interpretável review-only (`StandardScaler → SplineTransformer → Ridge`) contra o proxy interno `score_evento_enchente_potencial_v5_core`. R²_CV global ≈ 0.922; R²_CV por região ≈ 0.97; efeitos parciais por feature.

## 3. Por que o R² alto NÃO é predição real

O target é um **score heurístico composto v5**, derivado dos próprios condicionantes físicos usados como features. O modelo apenas reaprende a regra que gerou o score. Logo, R² alto mede **reprodutibilidade interna**, não capacidade de prever enchente observada. `predictive_power_claim = false`.

## 4. O que significa recuperabilidade do heurístico

`r2_interpretation = heuristic_recoverability`: o baseline consegue reconstruir o score v5 a partir das features porque o score é função delas. É um teste de **consistência/auditabilidade do heurístico**, útil para checar coerência de sinais — não evidência preditiva.

## 5. Diagnóstico de circularidade

`SUSC_06B_baseline_circularity_report.csv` — `circularity_risk = high`. O target pertence ao grupo `score`, é derivado dos condicionantes físicos, e é proxy (não ground truth). Qualquer uso do R² como "acurácia preditiva" é **inválido**.

## 6. Diagnóstico das divergências (4 features para investigação)

`SUSC_06B_partial_effect_direction_audit.csv`:

| feature | esperado | observado | classe | ação |
|---------|----------|-----------|--------|------|
| `curvature_laplacian_mean` | lower_increases | efeito positivo | direction_diverges | investigate_before_score_v6 |
| `rain_3d_7d_ratio` | higher_increases | efeito negativo | direction_diverges | investigate_before_score_v6 |
| `flow_acc_log_p75` | higher_increases | ≈ 0 | near_zero_effect | investigate_before_score_v6 |
| `water_occurrence_patch` | higher_increases | +0.0002 (≈ 0) | near_zero_effect | investigate_before_score_v6 |

- **curvature_laplacian_mean:** provável convenção de sinal da curvatura (côncavo/convexo) — auditar a fórmula antes de pesar.
- **rain_3d_7d_ratio:** razão de chuva curta interage de forma não trivial com o score composto (possível redundância com os acumulados CHIRPS).
- **flow_acc_log_p75 / water_occurrence_patch:** efeito parcial ~zero no contexto do proxy — o score v5 quase não responde a essas features; investigar peso/colinearidade.

As outras 19 features concordam em direção e magnitude (`direction_agrees`).

## 7. Diagnóstico de estabilidade regional

`SUSC_06B_region_stability_diagnostics.csv` — `regional_stability_status = stable`. R²_CV: Curitiba 0.979, Petrópolis 0.971, Recife 0.973; spread = 0.008 (< 0.05). O heurístico é reconstruído com consistência semelhante nas três regiões — o que reforça (não contradiz) a circularidade.

## 8. Features que avançam para score v6 (19 candidatas)

`SUSC_06B_feature_action_matrix.csv` → `advance_to_score_v6_candidate` (19): as `ready_for_methodology` cujo efeito parcial concorda em direção e não é near-zero (elevação, declividade, HAND, TWI, TPI, distância à drenagem, flow_acc_log_mean, CHIRPS×3, rain_7d_30d_ratio, rain_persistence_index, runoff×2, NDVI, MNDWI, NDBI, urban_prop, vegetation_prop). **Candidatas, não confirmadas** — fonte ainda `requires_manual_review` (SUSC-04).

## 9. Features que avançam para baseline SPGAM/GAM

As mesmas 19 (numéricas, interpretáveis, direção coerente). As 4 sob revisão entram no SPGAM apenas após auditoria.

## 10. Features que avançam para comparação DINO

Todas as 23 `ready` + 5 `usable_with_caution` (SAR/std) → `advance_to_dino_comparison=true`, como representação/contraste. DINO nunca como detector/ground truth.

## 11. Features bloqueadas ou em revisão

- **manual_review_required (4):** curvature, rain_3d_7d_ratio, flow_acc_log_p75, water_occurrence_patch.
- **advance_to_dino_comparison apenas (5 caution):** elevation_std, slope_std, s1_vv/vh/vv_minus_vh.
- **hold_until_recomputed / proxy_only_do_not_score:** scores/labels/proxies v5 e nomes ausentes (excluídos do scoring).

## 12. Como isso prepara o SUSC-07

`SUSC_06B_event_overlay_readiness_matrix.csv` lista os campos necessários ao overlay documental/evento:

| campo | situação |
|-------|----------|
| patch_id | disponível |
| regiao | disponível |
| geometry_or_bbox | existe na matriz SUSC-03 (xmin/ymin/xmax/ymax) — juntar por patch_id |
| score_or_proxy | disponível (proxy, nunca GT) |
| main_physical_features | disponíveis (19–23) |
| future_documentary_evidence | **lacuna** — aquisição externa (objetivo do SUSC-07) |
| date_or_period | **fraco** — reference_date constante 2022-12-31; período de evento real ausente |

Lacunas para SUSC-07: `future_documentary_evidence`, `date_or_period`.

## 13. Limitações científicas

- Circularidade alta: R² mede recuperabilidade, não predição.
- Métricas sempre contra proxy, nunca contra evento real.
- 4 features divergentes/near-zero exigem auditoria antes do score v6.
- Componente espacial é apenas estratificação por região.
- Sem evidência documental/temporal de evento — o overlay (SUSC-07) é externo e ainda não existe.

## 14. Próximo marco recomendado

**SUSC-07 — Overlay com evidências documentais/eventos:** adquirir evidência documental/hidrológica externa, juntar geometria (bbox SUSC-03) por patch e confrontar os proxies com evidência observada — **sem** transformar proxy em ground truth automaticamente. Antes: investigar as 4 features sinalizadas e confirmar as fontes `requires_manual_review`.

---

## Disclaimer obrigatório

> O SUSC-06B não valida ocorrência real de enchente. Ele diagnostica a consistência interna de um baseline ajustado contra proxy heurístico, registra riscos de circularidade e prepara a transição para validação documental/espacial em SUSC-07.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
