# SUSC-06A — Baseline Proxy GAM/SPGAM-style (Review-Only)

> O SUSC-06A é um baseline interpretável review-only ajustado contra proxies internos de suscetibilidade. Ele não valida ocorrência real de enchente, não cria ground truth, não desbloqueia treinamento supervisionado e não autoriza afirmações de evento observado por patch.

---

## 1. Objetivo do SUSC-06A

Ajustar um baseline interpretável inspirado em GAM/SPGAM para examinar como as features físicas, hidrológicas, urbanas e espectrais se relacionam com os **proxies internos de suscetibilidade** já existentes na matriz (scores/labels heurísticos v5). É um diagnóstico metodológico, não um preditor de enchente.

## 2. Por que GAM/SPGAM entra no projeto

GAM/SPGAM oferece um modelo aditivo **interpretável**: cada condicionante contribui com um efeito parcial visível, permitindo checar se a direção observada bate com a direção esperada pela literatura. Isso prepara um score multimodal v6 rastreável e comparável.

## 3. Relação com SPGAM/INPE

O SPGAM modela suscetibilidade espacial com elevação, declividade, distância à drenagem, NDBI/BU/NDVI e componente espacial suavizada. Aqui usamos os mesmos condicionantes (23 features `ready_for_methodology`) num baseline aditivo com splines, e estratificamos por região como aproximação simples do componente espacial.

## 4. Relação com Baixo Jaguaribe/UFC

O estudo usa Sentinel-1/SAR, Sentinel-2, GEE, limiar de retroespalhamento, cheia/seca e pluviometria/cota. SAR (`s1_*`) é `usable_with_caution` (ambíguo) e ficou **fora** do ajuste principal; índices Sentinel-2 e contexto CHIRPS entram como condicionantes/gatilho.

## 5. Relação com DINO

DINOv2 não participa deste baseline. Ele é reservado como camada latente complementar para generalização espacial em etapa futura (comparação, não classificação). Aqui o baseline é puramente tabular/interpretável.

## 6. Dataset usado

`datasets/suscetibilidade/susc_06_proxy_baseline_dataset_v1.csv` — 300 patches × 59 colunas (3 identidade + 23 features `ready` + 5 `usable_with_caution` opcionais + 28 colunas de proxy-target). Derivado da matriz SUSC-03 **sem alterá-la**.

## 7. Features usadas (23, ajuste principal)

`elevation_mean`, `slope_mean`, `hand_mean`, `twi_mean`, `tpi_250m_mean`, `curvature_laplacian_mean`, `distance_to_water_mean`, `water_occurrence_patch`, `flow_acc_log_mean`, `flow_acc_log_p75`, `chirps_3d_mm`, `chirps_7d_mm`, `chirps_30d_mm`, `rain_3d_7d_ratio`, `rain_7d_30d_ratio`, `rain_persistence_index`, `runoff_context_7d`, `runoff_context_30d`, `ndvi_mean`, `mndwi_mean`, `ndbi_mean`, `urban_prop`, `vegetation_prop`.

## 8. Features excluídas e motivo

- **`usable_with_caution` (5, opcionais, fora do ajuste):** `elevation_std`, `slope_std` (sem direção monotônica), `s1_vv_mean_clean`, `s1_vh_mean_clean`, `s1_vv_minus_vh_mean_clean` (SAR ambíguo).
- **`proxy_only` / `blocked_until_recomputed` / `unresolved` (33):** scores/labels/proxies v5 (são target candidato, nunca feature) e nomes ausentes da matriz.

## 9. Target proxy escolhido

`score_evento_enchente_potencial_v5_core` (contínuo 0–1), via prioridade `final_susceptibility_score`. Política completa em `SUSC_06_proxy_baseline_target_policy.json`.

## 10. Por que o target não é ground truth

É um **score heurístico composto v5**, derivado dos próprios condicionantes por regra/limiar — não de evento observado. `target_is_ground_truth=false`. As métricas são **contra esse proxy**, não contra ocorrência real.

## 11. Método usado

Backend disponível: **`sklearn_spline_fallback`** (pygam e statsmodels ausentes; nada foi instalado). Pipeline reproduzível: `StandardScaler` → `SplineTransformer(n_knots=4, degree=3)` → `Ridge(alpha=1.0)` (target contínuo). Validação cruzada 5-fold (`random_state=42`). Nenhum modelo persistido em disco.

## 12. Resultados globais (contra proxy)

| Métrica | Valor |
|---------|-------|
| n | 300 |
| R² in-sample | 0.941 |
| R² CV (5-fold) média | 0.922 |
| R² CV desvio | 0.018 |
| MAE | 0.029 |
| RMSE | 0.036 |

## 13. Resultados por região (contra proxy)

| Região | n | R² CV | MAE | RMSE |
|--------|---|-------|-----|------|
| curitiba | 100 | 0.979 | 0.011 | 0.013 |
| petropolis | 100 | 0.971 | 0.012 | 0.017 |
| recife | 100 | 0.973 | 0.012 | 0.015 |

## 14. Efeitos parciais / interpretabilidade

`SUSC_06_proxy_baseline_partial_effects.csv` e `..._feature_table.csv` reportam, por feature, o efeito parcial (p10→p90) e o coeficiente linear padronizado, comparados à direção esperada. **20 de 23 features concordam** com a direção esperada da literatura. Divergências a investigar:

- `curvature_laplacian_mean` (esperado `lower_increases`, observado efeito positivo) — possível convenção de sinal da curvatura.
- `flow_acc_log_p75` (efeito ~zero no contexto do proxy).
- `rain_3d_7d_ratio` (esperado `higher_increases`, observado negativo) — razão de chuva interage de forma não trivial com o score composto.

## 15. Limitações

- **Circularidade:** o target v5 é **derivado** dessas mesmas features; o R² alto mede **recuperabilidade/consistência interna do heurístico**, não poder preditivo contra eventos reais.
- Métricas são contra proxy, nunca contra ocorrência confirmada.
- SAR e termos compostos ficaram fora do ajuste principal.
- Componente espacial é apenas estratificação por região (sem suavização espacial por coordenada).
- Fonte definitiva de cada feature ainda é `requires_manual_review` (SUSC-04).

## 16. Como isso prepara o score v6

Fornece um baseline interpretável e os efeitos parciais por condicionante, permitindo que o score multimodal v6 seja construído com pesos/transformações defensáveis e comparáveis a um modelo aditivo — não como caixa-preta.

## 17. Como isso prepara a comparação futura com DINO

Estabelece a referência tabular interpretável (efeitos físicos) contra a qual as features latentes DINOv2 poderão ser **comparadas** (concordância/complementaridade), sem que DINO atue como detector ou ground truth.

## 18. Próximo marco recomendado

**SUSC-06B / SUSC-07** — consolidar o baseline interpretável (confirmar fontes `requires_manual_review`, investigar as 3 divergências de direção) e avançar para validação por evidência documental/hidrológica (overlay), antes de qualquer noção de referência observada.

---

## Disclaimer obrigatório

> O SUSC-06A é um baseline interpretável review-only ajustado contra proxies internos de suscetibilidade. Ele não valida ocorrência real de enchente, não cria ground truth, não desbloqueia treinamento supervisionado e não autoriza afirmações de evento observado por patch.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
