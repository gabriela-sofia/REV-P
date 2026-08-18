# SUSC-01 — Schema Formal da Matriz de Suscetibilidade por Patch

**Data:** 2026-06-26
**Status:** Review-only. Nenhum treinamento desbloqueado. Nenhum ground truth criado.

---

## 1. Objetivo

Formalizar a representacao das 211 colunas do `dataset_final.csv` (PROJETO) como um schema auditavel de suscetibilidade urbana a enchentes no REV-P, preservando governanca, rastreabilidade e bloqueio anti-leakage.

Esta etapa NAO cria ground truth e NAO desbloqueia treinamento supervisionado. Ela formaliza uma matriz multimodal review-only de atributos associados a suscetibilidade urbana a enchentes.

---

## 2. Resumo do dataset_final.csv

| Atributo | Valor |
|----------|-------|
| Linhas | 300 |
| Colunas | 211 |
| Regioes | curitiba (100), petropolis (100), recife (100) |
| reference_date | 2022-12-31 (todas) |
| Features core completas | 46/46 (0 missing) |
| Colunas com missing | 50 (todas CBERS linkage legacy, NAO afeta susceptibilidade) |
| Colunas numericas | 111 |
| Colunas categoricas | 97 |

---

## 3. Grupos de colunas

| Grupo | Colunas | Descricao |
|-------|---------|-----------|
| patch_identity | 3 | patch_id, regiao, reference_date |
| geometry | 4 | xmin, ymin, xmax, ymax (WGS84) |
| sentinel2_bands | 6 | B2, B3, B4, B8, B11, B12 (mean) |
| spectral_index | 3 | ndvi_mean, mndwi_mean, ndbi_mean |
| topography | 6 | elevation (mean/std), slope (mean/std), tpi_250m, curvature_laplacian |
| hydrology | 6 | hand_mean, twi_mean, distance_to_water, water_occurrence, flow_acc (mean/p75) |
| precipitation | 8 | chirps (3d/7d/30d), rain ratios, persistence, runoff (7d/30d) |
| sar | 3 | s1_vv, s1_vh, vv-vh (cleaned) |
| land_use | 2 | urban_prop, vegetation_prop |
| interaction | 2 | urban_water, urban_drainage |
| proxy_v5 | 18 | Indicadores binarios do sistema de scores v5 |
| score | 8 | Scores compostos heuristicos v5 |
| heuristic_label | 3 | Label heuristico + confianca + peso |
| qa | 8 | study_value, flags, ranking, selection |
| cbers_linkage | ~82 | Legacy CBERS linkage (nao relevante para suscetibilidade) |
| cbers_refresh | ~43 | CBERS structural quality metrics |

---

## 4. Features ja operacionais (100% completas, 300 patches)

### Topografia (6 features)
- `elevation_mean` / `elevation_std` — altitude media e variabilidade
- `slope_mean` / `slope_std` — declividade media e variabilidade
- `tpi_250m_mean` — posicao topografica (vales/cristas)
- `curvature_laplacian_mean` — convergencia/divergencia de fluxo

### Hidrologia (6 features)
- `hand_mean` — altura acima da drenagem mais proxima (CRITICA)
- `twi_mean` — indice topografico de umidade
- `distance_to_water_mean` — distancia media a corpos d'agua
- `water_occurrence_patch` — ocorrencia historica de agua (JRC)
- `flow_acc_log_mean` / `flow_acc_log_p75` — acumulacao de fluxo

### Espectral (3 features)
- `ndvi_mean` — vegetacao (permeabilidade)
- `mndwi_mean` — umidade/agua superficial
- `ndbi_mean` — area construida (impermeabilizacao)

### SAR (3 features)
- `s1_vv_mean_clean` / `s1_vh_mean_clean` / `s1_vv_minus_vh_mean_clean`

### Precipitacao (8 features)
- CHIRPS 3d/7d/30d + ratios + persistence + runoff

### Uso do solo (2 features proxy)
- `urban_prop` / `vegetation_prop`

### Interacoes (2 features)
- `urban_water_interaction` / `urban_drainage_interaction`

**Total: 30 features operacionais para suscetibilidade**

---

## 5. Features proxy

| Feature | Proxy de | Limitacao |
|---------|----------|-----------|
| urban_prop | Impermeabilizacao | Fonte desconhecida. Possivelmente MapBiomas via GEE. |
| vegetation_prop | Permeabilidade | Mesma fonte desconhecida que urban_prop. |
| 18 proxy_v5_* | Indicadores binarios | Derivados dos scores v5 com limiares internos. |

---

## 6. Features heuristicas (NAO sao ground truth)

| Feature | Interpretacao | Risco |
|---------|---------------|-------|
| score_predisposicao_hidrotopografica_v5 | Predisposicao fisica ao acumulo de agua | Composicao heuristica de features |
| score_gatilho_hidroclimatico_v5 | Gatilho pluviometrico | Idem |
| score_amplificacao_urbana_v5 | Amplificacao por urbanizacao | Idem |
| score_superficie_optica_v5_diagnostic | Diagnostico espectral | Idem |
| score_umidade_antecedente_v5 | Umidade acumulada | Idem |
| score_impulso_chuva_v5 | Impulso pluviometrico | Idem |
| score_evento_enchente_potencial_v5_core | Score combinado final | HEURISTICO. NAO validado contra eventos reais. |
| label_evento_enchente_potencial_v5_core_regional_p75 | Label binario p75 | CRITICO: NAO e ground truth. Nunca usar para treino supervisionado. |

---

## 7. Features com origem incerta

| Feature | Questao de proveniencia |
|---------|------------------------|
| hand_mean | Fonte GEE HAND ou derivada localmente? Script nao identificado. |
| twi_mean | GEE ou computacao local? Script nao encontrado. |
| tpi_250m_mean | Janela 250m documentada, mas script nao encontrado. |
| curvature_laplacian_mean | Derivado de DEM, script nao encontrado. |
| flow_acc_log_mean/p75 | DEM routing, script nao encontrado. |
| water_occurrence_patch | Provavelmente JRC via GEE. Nao confirmado. |
| urban_prop / vegetation_prop | Fonte DESCONHECIDA. Auditoria critica. |
| runoff_context_7d/30d | Formula desconhecida. |
| rain_persistence_index | Formula desconhecida. |
| urban_water/drainage_interaction | Formula desconhecida. |
| s1_vv/vh_mean_clean | Metodo de limpeza desconhecido. |

**11 features requerem auditoria de proveniencia antes de publicacao.**

---

## 8. O que pode ser usado para suscetibilidade

Todas as 30 features core podem ser usadas como atributos de suscetibilidade em modo review-only, com as seguintes ressalvas:
- Proveniencia de 11 features precisa ser auditada
- Nenhuma feature pode ser usada para treino supervisionado
- Nenhuma feature e ground truth

---

## 9. O que NAO pode ser usado como ground truth

- label_evento_enchente_potencial_v5_core_regional_p75
- Todos os scores v5
- Todos os proxies v5
- Qualquer derivacao desses valores

---

## 10. Como esse schema prepara a integracao multimodal

O schema define o vocabulario formal, tipos, unidades, interpretacoes e restricoes de governanca para cada feature. Ele permite:

1. **Migracao auditavel** do dataset_final.csv para o REV-P (SUSC-02)
2. **Expansao sistematica** com novas features (MapBiomas, overlay SGB/CPRM, DINOv2)
3. **Validacao automatica** via script de integridade
4. **Rastreabilidade** de proveniencia de cada coluna
5. **Bloqueio formal** de treino e ground truth ate que governanca permita

---

## Artefatos criados

| Arquivo | Descricao |
|---------|-----------|
| `schemas/suscetibilidade/susc_features_schema_v1.json` | Schema formal com 72 features catalogadas |
| `manifests/suscetibilidade/susc_features_provenance_manifest_v1.csv` | Manifesto de proveniencia com 72 linhas |
| `scripts/suscetibilidade/validate_susc_features_schema_v1.py` | Script de validacao de integridade |
| `outputs_public/suscetibilidade/SUSC_01_feature_schema_report.md` | Este relatorio |
