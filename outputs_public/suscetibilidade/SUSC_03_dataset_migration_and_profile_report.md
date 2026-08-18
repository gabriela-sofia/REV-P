# SUSC-03 — Migração Auditável e Validação Exploratória Inicial da Matriz de Suscetibilidade

> A matriz SUSC-03 é um artefato tabular review-only de atributos associados à suscetibilidade urbana a enchentes. Ela não constitui ground truth de ocorrência, não desbloqueia treinamento supervisionado e não autoriza afirmações de evento observado por patch.

---

## 1. Objetivo do marco

Trazer para dentro do REV-P uma versão **leve, governada, validada e reproduzível** da matriz de suscetibilidade derivada de `PROJETO/data/dataset_final.csv`, selecionando apenas colunas catalogadas no schema/manifesto SUSC-01, preservando a política review-only e sem desbloquear treinamento supervisionado nem promover labels heurísticos a ground truth.

## 2. Origem da matriz

| Item | Valor |
|------|-------|
| Arquivo de origem | `PROJETO/data/dataset_final.csv` (read-only, não alterado) |
| Linhas na origem | 300 patches |
| Colunas na origem | 211 |
| Regiões | curitiba, petropolis, recife (100 patches cada) |
| Schema | `schemas/suscetibilidade/susc_features_schema_v1.json` (SUSC-01) |
| Manifesto de proveniência | `manifests/suscetibilidade/susc_features_provenance_manifest_v1.csv` (SUSC-01) |

A migração preserva os **tokens originais de string** (sem reformatar floats), mantendo fidelidade e auditabilidade.

## 3. Destino da matriz

| Artefato | Caminho |
|----------|---------|
| Matriz migrada | `datasets/suscetibilidade/susc_features_by_patch_v1.csv` |
| Manifesto do artefato | `manifests/suscetibilidade/susc_features_by_patch_v1_artifact_manifest.json` |

Governança gravada no manifesto do artefato: `allowed_for_training=false`, `review_only=true`, `can_be_used_as_ground_truth=false`, `scientific_status=susceptibility_features_not_event_ground_truth`.

## 4. Linhas/colunas migradas

- **Linhas migradas:** 300 (todos os patches)
- **Colunas migradas:** 72 (somente as catalogadas no schema **e** no manifesto)
- 139 colunas da origem **não** foram migradas (incluindo colunas CBERS legadas com paths privados, colunas não catalogadas e duplicatas).

## 5. SHA256

```
554f7deb57b6c4389809283ebb2270225521d3c6f0aed3fd62f31edbd060bd0c
```

O SHA256 é computado sobre o CSV migrado e validado contra o manifesto do artefato pelo script `validate_susc_features_by_patch_v1.py`.

## 6. Grupos de features

A matriz migrada usa o vocabulário do schema (14 grupos populados). O profile reagrupa no vocabulário de perfil de 14 buckets; `proxy_v5` (18 flags binárias heurísticas) é dobrado em `heuristic_label`, documentado abaixo.

| profile_group | nº features | schema_groups de origem |
|---------------|-------------|--------------------------|
| patch_identity | 3 | patch_identity |
| geometry | 4 | geometry |
| sentinel2_bands | 6 | sentinel2_bands |
| spectral_index | 3 | spectral_index |
| topography | 6 | topography |
| hydrology | 6 | hydrology |
| precipitation | 8 | precipitation |
| sar | 3 | sar |
| land_use | 2 | land_use |
| interaction | 2 | interaction |
| score | 8 | score |
| heuristic_label | 20 | heuristic_label + proxy_v5 |
| qa | 1 | qa |
| unknown | 0 | — |

Detalhamento completo em `SUSC_03_feature_profile_by_group.csv`.

## 7. Features completas

**Todas as 72 colunas migradas têm 0 valores ausentes** (missingness 0%). Em particular, os grupos core (`topography`, `hydrology`, `precipitation`, `spectral_index`) estão integralmente completos nos 300 patches. Detalhe em `SUSC_03_missingness_report.csv`.

## 8. Features com missing

**Nenhuma.** A varredura de missingness não encontrou colunas com valores ausentes. Nenhuma coluna totalmente vazia foi migrada.

## 9. Features de origem incerta

A definição precisa será produzida no SUSC-04. Estado atual conforme o manifesto de proveniência:

- **62 features** carregam `requires_provenance_audit=true`.
- **68 features** não tiveram o script de computação localizado dentro do REV-P (`computation_script_found=false`) — esperado, pois a computação ocorreu no pipeline operacional em `PROJETO`.
- **4 features measured** têm fonte pública genuinamente indeterminada (`public_source_known=false`): `urban_prop`, `vegetation_prop`, `urban_water_interaction`, `urban_drainage_interaction`.
- **19 features** dependem de raster/insumo bruto não disponível no REV-P (`raw_source_available=false`) — sobretudo hidrologia, SAR e contexto pluviométrico.

> O plano SUSC-02 estimou "~11 features de origem incerta". O manifesto formal SUSC-01 amplia e estrutura essa contagem; a reconciliação e a lista definitiva são tarefa do **SUSC-04**. Nada aqui é tratado como resolvido.

## 10. Relação com SPGAM

O SPGAM emprega declividade, elevação, orientação de vertentes, distância à drenagem e índices de impermeabilização como condicionantes de suscetibilidade. A matriz SUSC-03 já carrega os correspondentes diretos: `slope_mean`/`slope_std`, `elevation_mean`/`elevation_std`, `distance_to_water_mean`, `hand_mean`, `twi_mean`, `tpi_250m_mean`, `curvature_laplacian_mean`, e proxies de impermeabilização (`urban_prop`, `ndbi_mean`). Isso prepara um baseline interpretável tipo GAM por região (SUSC-06), **sem** treinar nada agora.

## 11. Relação com estudo Baixo Jaguaribe

O estudo do Baixo Jaguaribe usa Sentinel-1/SAR (retroespalhamento), Sentinel-2, regime cheia/seca, uso/cobertura do solo e validação por evidência hidrológica. A matriz traz `s1_vv_mean_clean`, `s1_vh_mean_clean`, `s1_vv_minus_vh_mean_clean` (SAR), bandas Sentinel-2 e índices (`ndvi_mean`, `mndwi_mean`, `ndbi_mean`), `water_occurrence_patch` e proporções de uso do solo. A validação por evidência hidrológica/documental permanece **pendente** e é tratada como marco separado (SUSC-07), sem substituir ocorrência observada.

## 12. Relação com DINOv2

DINOv2 (com registers, rota validada) entra como **representação latente complementar**, não como detector nem ground truth. A matriz SUSC-03 é tabular/física-orbital; os embeddings DINO serão acoplados como features latentes adicionais por patch em marco posterior (SUSC-08/SUSC-09), preservando read-only e sem promover clustering a classe.

## 13. Por que isso ainda não é ground truth

- Os labels (`label_evento_enchente_potencial_v5_core_regional_p75`, `label_confidence_v5`) são **heurísticos**, derivados de scores compostos por limiar regional (p75) — não de eventos observados.
- Os scores v5 são composições heurísticas de condicionantes, **não validadas** contra ocorrência documental.
- Nenhuma coluna tem `can_be_used_as_ground_truth=true`; o validador falha-fechado caso isso mude.
- Suscetibilidade descreve **predisposição**, não a confirmação de que uma enchente ocorreu no patch.

## 14. Por que isso ainda não desbloqueia treino

- Todas as 72 features carregam `allowed_for_training=false`; a migração rejeita (fail-closed) qualquer coluna marcada para treino.
- Sem ground truth de ocorrência, treinar supervisionado introduziria vazamento (heurística → "verdade" circular).
- A política anti-leakage do REV-P é herdada e mantida até overlay de suscetibilidade oficial (SGB/CPRM) + revisão.

## 15. Como essa matriz prepara o score multimodal v6

A SUSC-03 consolida, em um único artefato auditável e versionável, as features físicas/orbitais já completas e os scores v5 de referência. Isso fornece a base tabular sobre a qual o score multimodal v6 (SUSC-09) combinará: (a) condicionantes físico-hidro-topográficos, (b) evidência orbital óptica/SAR, e (c) representação latente DINOv2 — todos como **features**, comparados contra um baseline SPGAM e contra o v5, sem que nenhum deles seja tratado como rótulo verdadeiro.

## 16. Próximos marcos recomendados

Ver `SUSC_roadmap_after_03.md`. Em resumo: SUSC-04 (auditoria de proveniência), SUSC-05 (direções esperadas), SUSC-06 (baseline SPGAM/GAM por região), SUSC-07 (validação evento-real/documental), SUSC-08 (expansão DINO), SUSC-09 (score multimodal v6), SUSC-10 (comparação SPGAM vs v6 vs DINO).

---

## Status de testes globais

A validação específica do marco SUSC-03 foi concluída com sucesso: schema, matriz migrada, profiling e testes específicos passaram (11/11).

A suíte global `python -m pytest tests -q` não foi usada como critério bloqueante porque apresentou 23 erros de coleção pré-existentes relacionados a módulos ausentes fora do escopo SUSC-03 (ex.: `ModuleNotFoundError: No module named 'revp_v1il_deep_local_vector_asset_recovery'`). Esses erros não foram introduzidos por este marco e não afetam os artefatos de migração, validação e profiling da matriz de suscetibilidade.

---

## Validações executadas

- `validate_susc_features_schema_v1.py` → PASSED (72 features, governança OK)
- `migrate_dataset_final_to_revp.py` → 300×72, SHA256 gravado
- `validate_susc_features_by_patch_v1.py` → PASSED (12 checagens fail-closed)
- `profile_susc_features_by_patch_v1.py` → 5 saídas exploratórias, 0 alertas de colinearidade ≥ 0.90, nenhum modelo treinado

## Disclaimer científico (obrigatório)

> A matriz SUSC-03 é um artefato tabular review-only de atributos associados à suscetibilidade urbana a enchentes. Ela não constitui ground truth de ocorrência, não desbloqueia treinamento supervisionado e não autoriza afirmações de evento observado por patch.
