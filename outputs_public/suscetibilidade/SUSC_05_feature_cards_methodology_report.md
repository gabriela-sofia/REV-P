# SUSC-05 — Feature Cards Científicos e Registro Formal da Cadeia SUSC-01→04

> O SUSC-05 não calcula score, não treina modelo e não cria ground truth. Ele transforma as features auditadas em unidades metodológicas explicáveis, permitindo que o score v6 e o baseline GAM/SPGAM sejam construídos posteriormente com rastreabilidade científica.

---

## 1. Objetivo do SUSC-05

Registrar formalmente a cadeia SUSC-01→04 no repositório e produzir **feature cards científicos** — uma unidade metodológica por métrica, consolidando fórmula/derivação, interpretação, direção esperada, fonte, limitação e papel futuro. Prepara a base metodológica para o SUSC-06 (baseline GAM/SPGAM interpretável).

## 2. SUSC-01→04 já está commitado localmente

Três commits locais (branch `marco/pre-unificacao-gates-mv1`, `[ahead 3]`, **sem push**):

| Stage | Hash | Mensagem |
|-------|------|----------|
| SUSC-01/02 | `4b49eb5` | feat: formaliza schema de suscetibilidade SUSC-01-02 |
| SUSC-03 | `7986da4` | feat: migra matriz auditavel de suscetibilidade SUSC-03 |
| SUSC-04 | `3c49608` | feat: audita proveniencia cientifica das features SUSC-04 |

Registro completo em `SUSC_01_04_chain_closure_register.md` e `manifests/suscetibilidade/susc_chain_commits_manifest_v1.csv`.

## 3. Como os feature cards foram derivados

O gerador `build_susc_feature_cards_v1.py` (read-only) cruza, por feature:
- **schema SUSC-01** (`susc_features_schema_v1.json`) — grupo, dtype, interpretação, unidade;
- **direção científica SUSC-04** (`susc_feature_scientific_direction_v1.json`) — direção esperada, racional, âncora na literatura;
- **manifesto de proveniência** (`susc_features_provenance_manifest_v1.csv`) — `public_source_known`;
- **audit de proveniência SUSC-04** (`susc_feature_provenance_audit_v1.csv`) — status de proveniência, fonte candidata, flags `safe_for_*`, `requires_manual_review`;
- **matriz de decisão v6** (`SUSC_04_score_v6_feature_decision_matrix.csv`) — papel e classe de peso.

Cada card recebe um `status` do vocabulário: `ready_for_methodology`, `usable_with_caution`, `proxy_only`, `blocked_until_recomputed`, `unresolved`. Nada é inventado; o que não está comprovado fica `requires_manual_review=true`.

## 4. Relação com SPGAM/INPE

O SPGAM modela suscetibilidade com declividade, elevação, orientação de vertentes, distância à drenagem, NDBI, BU e NDVI. Os cards de `slope_mean`, `elevation_mean`, `distance_to_water_mean`, `hand_mean`, `tpi_250m_mean`, `curvature_laplacian_mean`, `ndbi_mean`, `ndvi_mean`, `urban_prop` e `vegetation_prop` documentam esse alinhamento como condicionantes diretos.

## 5. Relação com Baixo Jaguaribe/UFC

O estudo mapeia áreas inundáveis com Sentinel-1/SAR (limiar de retroespalhamento), Sentinel-2, GEE, regime cheia/seca, precipitação/cota e uso/cobertura do solo. Os cards de `s1_vv_mean_clean`, `s1_vh_mean_clean`, `s1_vv_minus_vh_mean_clean` (núcleo SAR), índices Sentinel-2 (`ndvi`/`mndwi`/`ndbi`), `water_occurrence_patch`, `chirps_*` e proporções de uso do solo documentam esse alinhamento.

## 6. Relação com a matriz multimodal REV-P

A SUSC-03 consolidou 300 patches × 72 colunas em três regiões. Os cards organizam essas métricas em unidades comparáveis, separando feature física, índice espectral, evidência SAR, gatilho pluviométrico e heurística v5 — base para um score multimodal v6 auditável.

## 7. Relação com DINO

DINOv2 (com registers) entra como camada latente complementar de generalização espacial. Todos os cards declaram explicitamente: **DINO compara padrões visuais/espaciais latentes, não substitui a métrica física, não é detector e não é ground truth.**

## 8. Tabela de features por status

| status | nº |
|--------|----|
| ready_for_methodology | 23 |
| usable_with_caution | 5 |
| proxy_only | 30 |
| blocked_until_recomputed | 3 |
| unresolved | 0 |
| **Total de cards** | **61** |

(Detalhe completo em `feature_cards/SUSC_05_feature_cards_catalog.md` e `SUSC_05_feature_cards_summary.csv`.)

## 9. Features prontas para metodologia (23 · `ready_for_methodology`)

`elevation_mean`, `slope_mean`, `hand_mean`, `twi_mean`, `tpi_250m_mean`, `curvature_laplacian_mean`, `distance_to_water_mean`, `water_occurrence_patch`, `flow_acc_log_mean`, `flow_acc_log_p75`, `chirps_3d_mm`, `chirps_7d_mm`, `chirps_30d_mm`, `rain_3d_7d_ratio`, `rain_7d_30d_ratio`, `rain_persistence_index`, `runoff_context_7d`, `runoff_context_30d`, `ndvi_mean`, `mndwi_mean`, `ndbi_mean`, `urban_prop`, `vegetation_prop`.

## 10. Features utilizáveis com cautela (5 · `usable_with_caution`)

`elevation_std`, `slope_std` (sem direção monotônica isolada) e `s1_vv_mean_clean`, `s1_vh_mean_clean`, `s1_vv_minus_vh_mean_clean` (SAR ambíguo — evidência complementar, não verdade).

## 11. Features proxy-only (30 · `proxy_only`)

8 scores v5, 2 labels heurísticos e 18 proxies binários v5 + 2 termos de interação compostos (`urban_water_interaction`, `urban_drainage_interaction`). **Todos heurísticos — nunca ground truth, nunca treino.**

## 12. Features bloqueadas ou unresolved (3 · `blocked_until_recomputed`)

`chirps_3d_to_30d_ratio`, `chirps_7d_to_30d_ratio`, `runoff_score` — nomes solicitados ausentes da matriz SUSC-03. Reais mais próximos: `rain_3d_7d_ratio`/`rain_7d_30d_ratio`/`runoff_context_*`. Exigem recomputação explícita; nada inventado.

## 13. Features elegíveis para score v6 (23)

As 23 `ready_for_methodology` (físicas/hidro/topográficas monotônicas verificadas + precipitação-gatilho + índices espectrais + uso do solo). **Elegibilidade ≠ cálculo:** o score v6 não foi computado.

## 14. Features elegíveis para GAM/SPGAM baseline (23)

As mesmas 23 (numéricas, interpretáveis, direção definida). Pré-requisito antes do SUSC-06: confirmar manualmente a fonte definitiva (todas `requires_manual_review=true`).

## 15. Features elegíveis para comparação DINO (28)

As 23 acima + 5 features físicas/orbitais adicionais numéricas (incluindo SAR ambíguo, útil como representação/contraste). DINO entra como camada latente complementar, nunca como classificador.

## 16. Limites científicos

- Suscetibilidade ≠ ocorrência confirmada de enchente.
- Scores e labels v5 são heurísticos; carded como `proxy_only`, nunca verdade.
- Atribuição de fonte por feature ainda exige confirmação manual.
- HAND/TWI são `design_only` nos specs; computação real a confirmar.
- Termos compostos v5 e contexto de runoff/persistência: fórmula a auditar.
- Nenhum score, modelo, ground truth ou treino foi criado neste marco.

## 17. Próximo marco recomendado

**SUSC-06 — Baseline GAM/SPGAM interpretável por região**, usando as 23 features `ready_for_methodology` como condicionantes, após confirmação manual das fontes. Sem usar labels heurísticos como verdade.

---

## Disclaimer obrigatório

> O SUSC-05 não calcula score, não treina modelo e não cria ground truth. Ele transforma as features auditadas em unidades metodológicas explicáveis, permitindo que o score v6 e o baseline GAM/SPGAM sejam construídos posteriormente com rastreabilidade científica.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
