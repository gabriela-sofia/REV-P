# SUSC-02 — Plano de Migracao Auditavel do dataset_final.csv para o REV-P

**Data:** 2026-06-26
**Status:** Plano aprovado para execucao. Migracao NAO executada neste estagio.

---

## 1. Objetivo

Definir como transferir a matriz de suscetibilidade de `PROJETO/data/dataset_final.csv` para o repositorio oficial `REV-P`, preservando governanca, sem copiar dados pesados e sem desbloquear treinamento.

---

## 2. O que pode migrar para REV-P

### Pode migrar (CSV tabular leve)

| Artefato | Origem | Destino REV-P | Tamanho estimado |
|----------|--------|---------------|------------------|
| Matriz de features core (30 features + identidade + geometria) | dataset_final.csv (colunas 1-43) | `datasets/suscetibilidade/susc_features_by_patch_v1.csv` | ~50 KB |
| Proxies v5 (18 colunas booleanas) | dataset_final.csv (colunas 83-100) | Incluso no CSV acima | +5 KB |
| Scores v5 (8 colunas) | dataset_final.csv (colunas 101-108) | Incluso | +3 KB |
| Labels heuristicos (4 colunas) | dataset_final.csv (colunas 109-114) | Incluso | +2 KB |
| QA/ranking (6 colunas) | dataset_final.csv (colunas 115-126) | Incluso | +3 KB |
| Schema JSON | Criado neste estagio | `schemas/suscetibilidade/susc_features_schema_v1.json` | 30 KB |
| Manifesto de proveniencia | Criado neste estagio | `manifests/suscetibilidade/susc_features_provenance_manifest_v1.csv` | 8 KB |
| Script de validacao | Criado neste estagio | `scripts/suscetibilidade/validate_susc_features_schema_v1.py` | 5 KB |
| Hash SHA256 do dataset_final.csv | A computar | `manifests/suscetibilidade/dataset_final_sha256.txt` | <1 KB |

### NAO deve migrar

| Artefato | Razao |
|----------|-------|
| Colunas CBERS linkage (44-82, 127-211) | Legacy, nao relevante para suscetibilidade, contem paths privados |
| 128 GeoTIFFs Sentinel-2 | Dados pesados (~GB). Git-ignored. |
| 83 stacks NPY | Dados pesados. Git-ignored. |
| 9 tiles PE3D MDT (TIF) | Rasters. Git-ignored. |
| Shapefiles SGB/CPRM | Dados brutos vetoriais. Git-ignored. |
| GeoJSONs hidrograficos brutos | Ficam em PROJETO/data/raw. |
| Figuras PNG | Outputs locais. |
| Embeddings NPZ | Ja em local_runs/ do REV-P (12 reais). |

---

## 3. Estrutura proposta no REV-P

```
datasets/suscetibilidade/
    susc_features_by_patch_v1.csv         <- Matriz core (67 colunas, 300 linhas)

schemas/suscetibilidade/
    susc_features_schema_v1.json          <- Schema formal [JA CRIADO]

manifests/suscetibilidade/
    susc_features_provenance_manifest_v1.csv  <- Proveniencia [JA CRIADO]
    dataset_final_sha256.txt              <- Hash do original

scripts/suscetibilidade/
    validate_susc_features_schema_v1.py   <- Validacao [JA CRIADO]
    migrate_dataset_final_to_revp.py      <- Script de migracao (a criar)

outputs_public/suscetibilidade/
    SUSC_01_feature_schema_report.md      <- Relatorio SUSC-01 [JA CRIADO]
    SUSC_02_dataset_final_migration_plan.md  <- Este plano [JA CRIADO]
```

---

## 4. Colunas a incluir na migracao

As seguintes 67 colunas devem ser extraidas do dataset_final.csv:

### Identidade e geometria (7)
patch_id, regiao, reference_date, xmin, ymin, xmax, ymax

### Sentinel-2 bandas (6)
B2_mean, B3_mean, B4_mean, B8_mean, B11_mean, B12_mean

### Indices espectrais (3)
ndvi_mean, mndwi_mean, ndbi_mean

### Topografia (6)
elevation_mean, elevation_std, slope_mean, slope_std, tpi_250m_mean, curvature_laplacian_mean

### Hidrologia (6)
hand_mean, twi_mean, distance_to_water_mean, water_occurrence_patch, flow_acc_log_mean, flow_acc_log_p75

### Precipitacao (8)
runoff_context_7d, runoff_context_30d, chirps_3d_mm, chirps_7d_mm, chirps_30d_mm, rain_3d_7d_ratio, rain_7d_30d_ratio, rain_persistence_index

### SAR (3)
s1_vv_mean_clean, s1_vh_mean_clean, s1_vv_minus_vh_mean_clean

### Uso do solo (2)
urban_prop, vegetation_prop

### Interacoes (2)
urban_water_interaction, urban_drainage_interaction

### Proxies v5 (18)
proxy_v5_hand_low, proxy_v5_distance_water_low, proxy_v5_flow_accumulation, proxy_v5_twi_wetness, proxy_v5_flat_terrain, proxy_v5_low_elevation, proxy_v5_water_history, proxy_v5_rainfall_context, proxy_v5_runoff_context, proxy_v5_rain_concentration, proxy_v5_rain_persistence, proxy_v5_urban_exposure, proxy_v5_vegetation_low, proxy_v5_urban_water_interaction, proxy_v5_urban_drainage_interaction, proxy_v5_ndbi_built, proxy_v5_ndvi_low, proxy_v5_mndwi_wet

### Scores (2 representativos)
score_predisposicao_hidrotopografica_v5, score_evento_enchente_potencial_v5_core

### Labels heuristicos (2)
label_evento_enchente_potencial_v5_core_regional_p75, label_confidence_v5

### QA (1)
study_value_score_v1

**Total: 67 colunas**

---

## 5. Procedimento de migracao

### Passo 1: Computar hash
```bash
sha256sum PROJETO/data/dataset_final.csv > REV-P/manifests/suscetibilidade/dataset_final_sha256.txt
```

### Passo 2: Extrair colunas
Script `migrate_dataset_final_to_revp.py` deve:
- Ler PROJETO/data/dataset_final.csv
- Selecionar as 67 colunas listadas
- Validar 0 missing nas colunas core
- Escrever datasets/suscetibilidade/susc_features_by_patch_v1.csv
- Computar SHA256 do CSV gerado
- Gerar relatorio de migracao

### Passo 3: Validar
```bash
python scripts/suscetibilidade/validate_susc_features_schema_v1.py
```

### Passo 4: Adicionar cabecalho de governanca ao CSV
A primeira linha do CSV deve ter um comentario (ou o script de leitura deve impor):
- allowed_for_training = false
- review_only = true
- not_ground_truth = true

---

## 6. O que NAO fazer durante a migracao

- NAO copiar colunas CBERS (contem paths privados absolutos)
- NAO copiar rasters, embeddings ou outputs pesados
- NAO alterar o dataset_final.csv original
- NAO desbloquear allowed_for_training
- NAO tratar label_v5 como ground truth
- NAO fazer git add/commit/push sem autorizacao explicita

---

## 7. Proximas acoes manuais

Apos aprovacao deste plano:

1. **Autorizar execucao da migracao** (criar script + CSV migrado)
2. **Auditar proveniencia** das 11 features com origem incerta
3. **Revisar artefatos criados** em SUSC-01
4. **Stagear e commitar** quando satisfeita:
   ```bash
   git add schemas/suscetibilidade/
   git add manifests/suscetibilidade/
   git add scripts/suscetibilidade/
   git add outputs_public/suscetibilidade/
   git commit -m "SUSC-01 schema formal da matriz de suscetibilidade por patch"
   ```

---

## 8. Decisao de governanca

| Politica | Valor | Justificativa |
|----------|-------|---------------|
| allowed_for_training | false | Nenhum ground truth disponivel |
| review_only | true | Matriz para analise e revisao |
| can_create_ground_truth | false | Labels sao heuristicos |
| anti_leakage_enforced | true | Herda politica REV-P |

Esta politica sera mantida ate que:
- Suscetibilidade SGB/CPRM seja overlaid nos patches
- Proveniencia das 11 features seja auditada
- Revisao humana aprove mudanca de politica
