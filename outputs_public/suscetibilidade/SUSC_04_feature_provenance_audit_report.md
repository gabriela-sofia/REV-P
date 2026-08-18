# SUSC-04 — Auditoria de Proveniência Científica das Features de Suscetibilidade

> O SUSC-04 não cria ground truth e não desbloqueia treinamento supervisionado. Ele audita a proveniência e a direção científica das features de suscetibilidade para permitir, em etapa posterior, um score multimodal v6 e um baseline interpretável do tipo GAM/SPGAM.

---

## 1. Objetivo do SUSC-04

Para cada feature forte da matriz SUSC-03, auditar de forma rastreável: de onde veio, como foi calculada, qual fonte pública ou script a sustenta, qual unidade possui, qual a direção esperada em relação à suscetibilidade e qual status de confiança deve receber. Nenhum modelo é treinado; nenhum ground truth é criado; o score v6 **não** é calculado.

## 2. Estado de entrada (matriz SUSC-03)

- Matriz: `datasets/suscetibilidade/susc_features_by_patch_v1.csv` (300 patches × 72 colunas)
- SHA256 validado; `allowed_for_training=false`, `can_be_used_as_ground_truth=false`, `review_only=true`
- Review-only: atributos associados à suscetibilidade, **não** ocorrência confirmada.

## 3. Base científica usada

- **SPGAM/INPE** — declividade, elevação, orientação de vertentes, distância à drenagem, NDBI/BU/NDVI e pontos de ocorrência/não-ocorrência como condicionantes de suscetibilidade espacial.
- **Baixo Jaguaribe/UFC** — Sentinel-1/SAR (limiar de retroespalhamento), Sentinel-2 para validação espectral, GEE, cubo multitemporal cheia/seca, precipitação/cota e uso/cobertura do solo.
- **Matriz multimodal REV-P** — três regiões, features por patch, DINOv2 como representação latente complementar (não detector, não ground truth) e evidência documental pública.

## 4. Método de auditoria

O scanner `audit_susc_feature_provenance_v1.py` é somente-leitura e **exclui a própria camada SUSC** da varredura (a proveniência deve vir do pipeline a montante). Varreu **5.534 arquivos** de texto/código/config em REV-P e PROJETO (podando figuras, archives, worktrees, caches e binários). O casamento é por **limite de palavra**; a evidência de script exige o **nome exato da coluna** (ou função-spec conhecida) num `.py` não-SUSC; a fonte pública só é considerada **amarrada** quando aparece a até ±5 linhas do uso da feature no script de computação — caso contrário é registrada como `colocated_unverified`. Fonte não encontrada vira `unresolved` (nunca inventada).

Resultado global (33 features auditadas):

| provenance_status | nº |
|-------------------|----|
| verified_script_and_source | 20 |
| verified_script_only | 10 |
| verified_dataset_only | 0 |
| documented_only | 0 |
| conflicting_sources | 0 |
| unresolved | 3 |

> **Todas as 33 features estão marcadas `requires_manual_review=true`**: o scanner encontra fontes *candidatas* por proximidade, mas a fonte definitiva de cada feature precisa de confirmação humana antes de qualquer uso em score. As listas de fonte são pistas auditáveis, não atribuição final.

## 5. Features verificadas por script e fonte (20)

`elevation_mean`, `slope_mean`, `hand_mean`, `twi_mean`, `distance_to_water_mean`, `water_occurrence_patch`, `flow_acc_log_mean`, `flow_acc_log_p75`, `chirps_3d_mm`, `chirps_7d_mm`, `chirps_30d_mm`, `rain_3d_7d_ratio`, `rain_7d_30d_ratio`, `rain_persistence_index`, `runoff_context_7d`, `runoff_context_30d`, `urban_prop`, `vegetation_prop`, `urban_water_interaction`, `urban_drainage_interaction`.

Evidência de relevo é region-specific (PE3D em Recife, MDT GeoCuritiba em Curitiba, MDE em Petrópolis); contexto hidroclimático aparece junto a CHIRPS/JRC/GlobalSurfaceWater nos scripts de feature. **A atribuição exata de fonte por feature permanece para confirmação manual.**

## 6. Features verificadas apenas por script (10)

`elevation_std`, `slope_std`, `tpi_250m_mean`, `curvature_laplacian_mean`, `s1_vv_mean_clean`, `s1_vh_mean_clean`, `s1_vv_minus_vh_mean_clean`, `ndvi_mean`, `mndwi_mean`, `ndbi_mean`.

O nome da coluna (ou função-spec, p.ex. `ndvi_spec`/`mndwi_spec`/`ndbi_spec` em `features_optical.py`) aparece em scripts reais, mas nenhuma fonte pública foi amarrada por proximidade — a fonte está apenas documentada (Sentinel/Copernicus). TPI e curvatura derivam de DEM, ainda a confirmar no script de derivação.

## 7. Features verificadas apenas por dataset (0)

Nenhuma. Toda feature presente também tem evidência de script.

## 8. Features documentadas, mas sem script encontrado (0)

Nenhuma nesta passada.

## 9. Features unresolved (3)

`chirps_3d_to_30d_ratio`, `chirps_7d_to_30d_ratio`, `runoff_score` — **nomes solicitados que não existem na matriz SUSC-03**. As colunas reais mais próximas conceitualmente são `rain_3d_7d_ratio`, `rain_7d_30d_ratio` e `runoff_context_7d`/`runoff_context_30d` (definições diferentes). Nada foi inventado; ficam bloqueadas até recomputação explícita.

## 10. Direção esperada de cada métrica

| Métrica | Direção | Leitura |
|---------|---------|---------|
| HAND (`hand_mean`) | `lower_increases` | HAND baixo aumenta suscetibilidade |
| Declividade (`slope_mean`) | `lower_increases` | declividade baixa aumenta alagamento |
| Distância à drenagem (`distance_to_water_mean`) | `lower_increases` | distância menor aumenta suscetibilidade |
| TPI (`tpi_250m_mean`) | `lower_increases` | vales/depressões (negativo) acumulam |
| Curvatura (`curvature_laplacian_mean`) | `lower_increases` | concavidade converge fluxo |
| Elevação (`elevation_mean`) | `lower_increases` | cotas baixas acumulam |
| NDBI / urbanização (`ndbi_mean`, `urban_prop`) | `higher_increases` | impermeabilização alta aumenta |
| NDVI / vegetação (`ndvi_mean`, `vegetation_prop`) | `higher_decreases` | vegetação reduz escoamento |
| MNDWI (`mndwi_mean`) | `higher_increases` | água/umidade alta |
| Ocorrência de água (`water_occurrence_patch`) | `higher_increases` | presença recorrente |
| Flow accumulation (`flow_acc_log_mean`) | `higher_increases` | concentração de fluxo |
| TWI (`twi_mean`) | `higher_increases` | potencial de saturação |
| Chuva acumulada/persistente (`chirps_*`, `rain_persistence_index`, `runoff_context_*`) | `higher_increases` | gatilho hidrológico |
| Sentinel-1 VV/VH (`s1_*`) | `ambiguous` | evidência radar complementar, **não verdade** |
| Desvios (`elevation_std`, `slope_std`) | `ambiguous` | sem direção monotônica isolada |

## 11. Matriz de decisão (resumo)

Detalhe em `SUSC_04_score_v6_feature_decision_matrix.csv`.

- **Entram no score v6 agora (23):** núcleo físico/hidro/topográfico monotônico verificado + precipitação-gatilho + índices espectrais (`ndvi`/`mndwi`/`ndbi`) + `urban_prop`/`vegetation_prop`.
- **Entram no baseline SPGAM/GAM (23):** mesmas, todas numéricas, interpretáveis e com direção definida.
- **Entram só como proxy / precisam de auditoria de fórmula:** `urban_water_interaction`, `urban_drainage_interaction` (termos compostos v5).
- **Entram só na comparação DINO (28):** todas as físicas/orbitais numéricas, incluindo SAR ambíguo.
- **Precisa recalcular / excluir temporariamente (3):** `chirps_3d_to_30d_ratio`, `chirps_7d_to_30d_ratio`, `runoff_score` (ausentes da matriz).
- **Excluídas do v6 por ambiguidade:** `elevation_std`, `slope_std`, SAR (`s1_*`).

Pesos: 15 `high`, 10 `medium`, 5 `low`, 3 `blocked`. Nenhuma feature `ambiguous` recebe peso `high`.

## 12. Riscos metodológicos

- **Atribuição de fonte ambígua:** o scan lista fontes candidatas por proximidade; a fonte definitiva por feature ainda não está confirmada (todas `requires_manual_review=true`).
- **HAND/TWI como design_only:** em `features_topo_hydro.py` HAND e TWI são especificações `design_only`; a computação real que produziu os valores precisa ser localizada/confirmada.
- **Termos compostos v5** (interações, runoff_context, rain_persistence): fórmula a auditar antes de pesar.
- **SAR não-monotônico:** útil como representação/contraste, nunca como verdade de evento.
- **Uso do solo (`urban_prop`/`vegetation_prop`):** fonte pública (MapBiomas vs classificação GEE) ainda a confirmar.

## 13. O que ainda NÃO pode ser afirmado

- Que qualquer feature representa ocorrência confirmada de enchente.
- Que a fonte pública de cada feature está definitivamente estabelecida.
- Que o score v6 ou um baseline SPGAM já existam (não foram calculados).
- Que labels heurísticos v5 ou scores v5 sejam verdade.

## 14. O que JÁ pode ser afirmado

- 30/33 features têm evidência de script real no pipeline (REV-P/PROJETO); 3 nomes solicitados não existem na matriz e foram marcados `unresolved`.
- A direção científica esperada está formalizada e ancorada (SPGAM, Baixo Jaguaribe).
- Existe uma matriz de decisão auditável que separa o que pode entrar no score v6 / baseline SPGAM / comparação DINO do que está bloqueado.
- Governança preservada: todas as features `can_be_ground_truth=false`, `allowed_for_training=false`, `review_only=true`.

## 15. Próximo marco recomendado

**SUSC-05 — Formalização científica das direções esperadas das features** (consolidar `expected_direction` com citação metodológica por feature), seguida de **SUSC-06 — baseline SPGAM/GAM interpretável por região**. Antes de pesar features no v6, confirmar manualmente a fonte definitiva das 33 (todas `requires_manual_review=true`) e recomputar/excluir as 3 `unresolved`.

---

## Disclaimer obrigatório

> O SUSC-04 não cria ground truth e não desbloqueia treinamento supervisionado. Ele audita a proveniência e a direção científica das features de suscetibilidade para permitir, em etapa posterior, um score multimodal v6 e um baseline interpretável do tipo GAM/SPGAM.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
