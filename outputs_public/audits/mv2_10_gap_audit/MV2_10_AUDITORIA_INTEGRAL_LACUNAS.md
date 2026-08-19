# MV2-10 — Auditoria Integral de Lacunas e Estado Científico do REV-P

**Data da auditoria:** 2026-06-23  
**Branch auditada:** `marco/validacao-label-free-evidencia-estrutural-mv1` (branch atual com untracked)  
**Branches MV2 auditadas:** `marco/mv2-10-dinov2-offline-hardening-visual` (commits MV2-01 a MV2-10)  
**Auditor:** Claude (auditoria técnico-científica; não implementa modelo, não cria label, não promove candidato)

---

## 1. Visão geral e posição no cronograma

O projeto REV-P opera **entre os Dias 8 e 14** do cronograma técnico, com marcos MV2-01 a MV2-10 **registrados em commit** na branch `marco/mv2-10-dinov2-offline-hardening-visual` mas **não na branch ativa** (`marco/validacao-label-free-evidencia-estrutural-mv1`).

A branch ativa contém dezenas de arquivos untracked de execuções recentes (execution_reports, metrics, tables e docs metodológicos). Esses arquivos documentam estado real mas **não estão commited**.

**Posição exata no cronograma:**
- Dias 1–7: CONCLUÍDOS (MV2-01 a MV2-03 + contexto MV1)
- Dia 8 (embeddings n≥59): **PARCIAL** — DINOv2 offline funcional; 60 embeddings totais gerados, mas cobertura Curitiba=100%, Recife=5.4%, Petrópolis=6.25% → corpus inválido para análise intercidade
- Dias 9–17: CONCLUÍDOS (via MV2-04 a MV2-09 + estrutura de readiness)
- Dia 10 (baseline espectral): **BLOQUEADO** — nenhum raster Sentinel espectral legível localmente
- Dias 18–21: **BLOQUEADOS** — dependem de evidência observacional, positivos/negativos formais, silver set e splits
- Dia 22 (sandbox supervisionado): **BLOQUEADO** por cascata de todos os anteriores

---

## 2. Auditoria de corpus, patches e inventários

### 2.1 Definição canônica do corpus

O corpus territorial/contextual original é de **59 patches** (definido antes do MV2). O manifesto Sentinel para DINO (`v1fu`) lista **128 assets** (43 Curitiba + 48 Petrópolis + 37 Recife).

**Diferença explicada:** 128 ≠ 59 porque o manifesto de entrada DINO inclui múltiplos assets candidatos por patch; os 59 são o conjunto territorial original; os 128 são candidatos de entrada para extração de embeddings.

**Problemas identificados:**
- Vinculação patch→asset é por **ordem determinística** (`DESIGNATED_BY_DETERMINISTIC_ORDER_NEEDS_REVIEW`), não por confirmação espacial (overlay CRS)
- 20 patches têm designação candidata; 32 sem designação; 7 placeholder sem geometria de origem
- `patch_bound_validated=0/59` — zero patches com boundary vetorial confirmado espacialmente
- `can_create_training_label=false` em todos os 128

### 2.2 Distribuição por região

| Região | Patches (manifesto) | Assets canônicos PNG | Embeddings DINOv2 |
|--------|--------------------|--------------------|-------------------|
| Curitiba | 43 | 43 (100%) | 48 visuais + 4 originais = ~52 |
| Petrópolis | 48 | 3 (6.25%) | 0 novos (visuais) + 4 originais = 4 |
| Recife | 37 | 2 (5.4%) | 0 novos (visuais) + 4 originais = 4 |

O viés extremo de Curitiba é **o bloqueio principal do Dia 8**.

---

## 3. Auditoria de dados brutos e arquivos externos faltantes

### 3.1 Manifesto público

O arquivo `revp_manifesto_publico_arquivos_externos_url_hash_mv1.csv` registra **8 fontes externas baixadas** com SHA256 verificado, todas em quarentena local (não versionadas):

| Fonte | Papel | Status |
|-------|-------|--------|
| IBGE malhas municipais (3 cidades) | Contexto territorial | Baixado, SHA256 OK |
| GeoCuritiba bacias hidrográficas | Contexto GIS | Baixado, SHA256 OK |
| ANA inventário de estações (PDF) | Catálogo, não série | Baixado, SHA256 OK |
| MapBiomas COL.10 (XLSX nacional) | Contexto, não por patch | Baixado, SHA256 OK |
| SGB/CPRM carta suscetibilidade Petrópolis (PDF) | Suscetibilidade ≠ evento | Baixado, SHA256 OK |
| NHESS artigo Petrópolis 2022 | Contexto acadêmico | Baixado, SHA256 OK |
| Copernicus Charter758 (PNG+metadata) | Evidência candidata Recife | Parcial — vetor ausente |
| ANA série Capibaribe estação 39187800 | Dado hidrométrico Recife 2022 | VAZIO — sem dados 2022 |

### 3.2 Fontes críticas ausentes

**Crítico para desbloquear gates G2/G3/DIA_18:**
- GeoTIFFs Sentinel espectrais (128 recortes) → baseline espectral e DINOv2 espectral
- Vetor de inundação Recife 2022 (Charter758 ou Defesa Civil pós-revisão)
- Scene IDs dos 128 assets exportados via GEE

**Importante para gates G4/G5:**
- CEMADEN histórico de eventos (exige LAI formal)
- APAC série pluviométrica Recife
- Defesa Civil Curitiba — ocorrências jan/2022

Ver detalhes completos em `mv2_10_external_data_backlog.csv`.

---

## 4. Auditoria temporal

**Estado:** CRÍTICO

| Métrica | Valor |
|---------|-------|
| Assets com datetime válido | 0/128 |
| Assets com cloud cover | 0/128 |
| Sentinel-2 confirmados | 37/128 |
| Sentinel-1/SAR | 0/128 |
| Sensor UNKNOWN | 91/128 |
| Patches elegíveis Trilha A | 0 |
| Patches com ≥3 datas úteis | 0 |

A **Trilha A está completamente bloqueada** e permanecerá até que scene IDs e cloud cover sejam recuperados via histórico GEE ou re-exportação.

A tabela temporal unificada (datetime + cloud_cover + sensor + tile por patch) **não existe** — cada MV2 redescobre o bloqueio independentemente.

---

## 5. Auditoria espacial, geometria e CRS

### 5.1 Geometrias existentes

| Arquivo | CRS | Geometria real | Pode ser GT | Obs |
|---------|-----|---------------|-------------|-----|
| patch_boundary_REC_00019_from_lineage.geojson | EPSG:4326 | Sim (Polygon) | Não | Reproj de EPSG:32725; requires_human_review=true |
| v2aq_event_geometry_rec-2022-05-24-30.geojson | N/A | Null | Não | TEXTUAL_ANCHOR_ONLY; manual_digitization_required |
| v2aq_event_geometry_ctb-*.geojson (3 eventos) | N/A | Null | Não | Sem geometria vetorial |
| v2aq_event_geometry_pet-*.geojson (3 eventos) | N/A | Null | Não | Sem geometria vetorial |
| recife_defesa_civil_risk_locations.geojson | 4326 | Pontos (400 pontos) | Não | CONFLITANTE com Charter758; is_reviewed=false |

**Bloqueio principal:** Todos os GeoJSONs de evento têm `geometry: null`. O overlay patch-evento é impossível sem geometria não-nula.

### 5.2 CRS dos assets Sentinel

MV2-03 confirmou: EPSG:32722 (43 assets CUR), EPSG:32723 (48 PET), EPSG:32725 (37 REC). CRS está documentado. O bloqueio não é de CRS, é de ausência de geometria de evento.

---

## 6. Auditoria do Protocolo C e evidência observacional

### 6.1 Status dos eventos candidatos

| Evento | Região | Fenômeno | Geometria | Compatibilidade Sentinel | Evidência | Bloqueio |
|--------|--------|---------|-----------|------------------------|-----------|---------|
| REC_2022_05_24_30 | Recife | Inundação | null (textual) | Compatível (37 assets REC) | Charter758 PNG + 400 pontos DC conflitantes | geometry=null; overlay=impossible |
| REC_2023_02_05_06 | Recife | Inundação | null | Compatível | Contexto | geometry=null |
| REC_2024_06_14_16 | Recife | Inundação | null | Compatível | Contexto | geometry=null |
| PET_2022_02_15 | Petrópolis | Mass movement | null | Compatível (48 assets PET) | NHESS + DRM-RJ | MASS_MOVEMENT ≠ FLOOD; cohort separado |
| PET_2022_03_20_21 | Petrópolis | Mass movement | null | Compatível | Contexto | cohort separado |
| PET_2024_03_21_28 | Petrópolis | Mass movement | null | Compatível | Contexto | cohort separado |
| CUR_2022_01_05 | Curitiba | Inundação urbana | null | Compatível (43 assets CUR) | Defesa Civil CTB (html CONTEXT_ONLY) | geometry=null; deep crawl sem resultado |
| CUR_2022_01_15 | Curitiba | Inundação urbana | null | Compatível | Defesa Civil CTB | geometry=null |
| CTB_2023_10_28_30 | Curitiba | Inundação | null | Compatível | Contexto | geometry=null |

**Separação de evidência:**
- Evidência contextual: IBGE, bacias hidrográficas, SGB, MapBiomas, NHESS
- Evidência territorial: Charter758 PNG (Recife, não vetorial, não revisado)
- Evidência documental: PDFs APAC, ANA catálogo, notícia Defesa Civil PR
- Evidência observacional: **ZERO** — nenhum polígono de inundação revisado por humano
- Candidato IA: 8 itens de MV2-06 (exploratórios, não formais)
- Silver formal: **ZERO**
- Gold: **ZERO**

---

## 7. Auditoria de embeddings DINOv2 e representações visuais

### 7.1 Estado atual dos embeddings

| Tipo | Quantidade | Origem | Espectral | Pode usar em treino |
|------|----------|--------|-----------|-------------------|
| DINOv2 originais (v1fz/v1ge) | 12 | Raster Sentinel (local_runs) | Sim (indireto) | Não |
| DINOv2 visuais MV2-10 | 48 | PNG renderizado (CUR dominante) | **Não** | Não |
| Features visuais MV2-09 | 48 | PNG canonical (CUR) | **Não** | Não |
| Features espectrais MV2-07 | 0 | — | — | — |

**ATENÇÃO CRÍTICA:** Os 48 novos embeddings DINOv2 (MV2-10) foram gerados a partir de **PNG renderizados** (representações visuais de cena), **não** de rasters Sentinel espectrais. Isso significa que não capturam informação de banda B08/B11, NDWI, ou qualquer feature multiespectral. São representações visuais, não dados Sentinel.

### 7.2 Bloqueio DINOv2 offline — diagnóstico completo

| Componente | Status |
|-----------|--------|
| Modelo facebook/dinov2-with-registers-base | PRESENTE localmente (HF cache) |
| Executor HF transformers | FUNCIONAL (MV2-10: READY_OFFLINE_HF_TRANSFORMERS_MANUAL_PREPROCESS) |
| Imagem de entrada espectral | AUSENTE (apenas PNG visual disponível) |
| Contrato de input espectral | AUSENTE (sem campo de bandas/normalização no manifesto) |
| GPU/CPU fallback | DISPONÍVEL |
| Dependência PyTorch/transformers | DISPONÍVEL |

O bloqueio DINOv2 **não é mais de executor** (resolvido em MV2-10). O bloqueio atual é de **dados de entrada**: apenas PNGs visuais estão disponíveis, não rasters espectrais das 128 cenas.

### 7.3 Viés de cobertura — detalhe crítico

```
Tipo representação        CUR    REC    PET
PNG canônico              100%   5.4%   6.25%
DINOv2 visual novo (MV2-10) 100%   5.4%   6.25%
DINOv2 original           9.3%   10.8%  8.3%
```

O viés extremo de Curitiba no corpus visual torna inválida qualquer comparação intercidade sobre esses embeddings.

---

## 8. Auditoria de viés de cidade e confounders

**Status: ALTO RISCO**

| Análise | Resultado | Status |
|---------|----------|--------|
| same_city_rate DINOv2 original n=12 | 0.389 (acaso ≈ 0.27) | indeterminado (n pequeno) |
| same_city_rate features visuais MV2-10 | 0.896 | CONFOUNDER — Curitiba domina |
| Intra-CUR vs inter-cidades | 0.998 vs 0.962 | alta similaridade intra mas n pequeno |
| Permutation test | AUSENTE | não realizado |
| Bootstrap IC95 | AUSENTE | não realizado |
| Leave-one-city-out | AUSENTE | não realizado |

A análise label-free atual sustenta apenas **hipótese exploratória muito inicial**. Qualquer claim além de "ilustração de infraestrutura" é frágil.

---

## 9. Auditoria de baseline espectral

**Status: BLOQUEADO**

Os 128 assets físicos foram descobertos (MV2-07: 111 encontrados, 48 casam com contrato). No entanto:

- **0 rasters legíveis como GeoTIFF espectral** — PIL lê como PNG; rasterio retorna erro
- **0 features NDWI/NDVI/NDBI/NIR/SWIR extraídas**
- `recovered_spectral_features.csv` contém 0 linhas válidas
- principal_root_cause=`NO_PHYSICAL_FILE_MATCH` (os arquivos encontrados fisicamente não correspondem aos esperados)

Arquivos necessários para baseline espectral: GeoTIFFs com bandas B02/B03/B04/B08/B11, CRS explícito, cloud mask, resolução 10m/20m. Esses arquivos existem no workspace PROJETO (privado) mas não estão acessíveis no workspace REV-P.

---

## 10. Auditoria de negativos, unknowns e silver set

| Item | Valor |
|------|-------|
| Negativos formais | 0 |
| Silver formais | 0 |
| Candidatos IA (exploratórios) | 8 silver / 128 negative candidates |
| Unknowns preservados como unknown | Sim (não promovidos) |
| Conversão ausência→negativo | Proibido e confirmado como não ocorrido |

**Política de evidência negativa** (`revp_politica_evidencia_negativa_mv1.md`):  
- `unknown` nunca vira negativo ✓  
- Ausência de evento nunca vira negativo formal ✓  
- Curitiba nunca vira negativo formal por default ✓  
- Evidência contextual nunca vira label ✓  

A política existe em prosa e CSV mas **não tem checagem programática** (risco metodológico: crítica C08 da banca).

**Bloqueios para Dia 19 (silver set):**
1. Positivos formais = 0 (G2/G3 bloqueados)
2. Revisão humana = 0 itens executados (G5 bloqueado)
3. Anti-leakage = policy existe mas nenhuma amostra aprovada (G7 bloqueado)
4. Formal protocol = não existe ainda

**Bloqueios para Dia 21 (matched/hard negatives):**
1. Fonte de não-ocorrência independente = ausente (G6 bloqueado)
2. Janela temporal fechada = ausente (G2 bloqueado)
3. Evidência formal = ausente

---

## 11. Auditoria de splits, anti-leakage e trainability gates

| Gate | Status | Motivo |
|------|--------|--------|
| G0 (patch ID válido) | PARCIAL | Rastreável mas não liberado para treino |
| G1 (asset ID válido) | PARCIAL | Rastreável; metadados temporais insuficientes |
| G2 (janela temporal fechada) | BLOQUEADO | 0 datas; 0 cloud cover |
| G3 (geometria espacial fechada) | BLOQUEADO | geometry=null em todos os eventos |
| G4 (fonte label independente) | BLOQUEADO | Nenhum label operacional; sem fonte independente |
| G5 (revisão humana completa) | BLOQUEADO | Decisões humanas vazias |
| G6 (política negativo satisfeita) | BLOQUEADO | 0 negativos formais; sem evidência de não-ocorrência |
| G7 (anti-leakage aprovado) | BLOQUEADO | Policy formalizada; 0 amostras aprovadas |
| G8 (liberado para treino) | BLOQUEADO | Todos os gates anteriores necessários |

**Riscos de leakage identificados:**
- Patches do mesmo evento/cidade vizinhos: risco moderado (aguarda splits)
- REC_00019 aparece como amostra estrutural E candidato positivo histórico: risco de circularidade futura (crítica C19)
- PNG renderizado → DINOv2 → evidência espectral: risco de confusão visual/espectral

---

## 12. Auditoria de scripts e pipeline

### 12.1 Scripts existentes — MV2

Os commits MV2-01 a MV2-10 criaram um total de ~80 scripts Python distribuídos em `scripts/`. Eles cobrem:
- Contrato observacional (`mv2_build_patch_asset_event_contract.py`)
- Manifesto temporal (`mv2_build_sentinel_asset_temporal_manifest.py`)
- Lineage (`mv2_build_asset_scene_lineage_manifest.py`)
- Inventário de representações (`mv2_build_embedding_inventory.py`)
- Readiness (`mv2_build_review_only_readiness_report.py`)
- Adjudicação IA (`mv2_ai_adjudication_common.py`)
- Forense raster (`mv2_08_raster_forensics_common.py`)
- DINOv2 visual (`mv2_10_dinov2_hardening_common.py`)

### 12.2 Scripts faltantes críticos

| Script | Propósito | Pode criar agora |
|--------|-----------|-----------------|
| `consolidar_tabela_temporal_por_patch.py` | Tabela mestra datetime+cloud+sensor+scene_id | Não (depende scene IDs) |
| `probe_gee_task_history.py` | Recuperar scene IDs dos exports GEE | Não (depende credencial GEE) |
| `exportacao_assets_visuais_recife_petropolis.py` | Balancear corpus visual | Não (depende workspace PROJETO) |
| `extrator_baseline_espectral.py` | NDWI/NDVI/NDBI dos rasters | Não (depende GeoTIFFs) |
| `vincular_patch_evento_spatial.py` | Overlay patch-evento | Não (depende geometria evento) |
| `bootstrap_intra_inter_cidade.py` | IC95 nas distâncias topológicas | Não (depende corpus balanceado) |

### 12.3 Scripts não relacionados a MV2

O diretório `scripts/` contém scripts de Protocolo C (v2at-v2cg) com runners (`run_v2at*.py`) e engines. Esses são scripts da cadeia Protocolo C, não da cadeia MV2, e têm funcionalidade independente.

---

## 13. Auditoria de schemas

### 13.1 Schemas existentes (MV2)

MV2-01 a MV2-10 criaram 13+ schemas JSON em `datasets/schemas/`:
- `mv2_patch_asset_event.schema.json` (MV2-01)
- `mv2_sentinel_asset_temporal_manifest.schema.json` (MV2-02)
- `mv2_asset_scene_lineage.schema.json` (MV2-03)
- `mv2_embedding_inventory.schema.json` (MV2-04)
- `mv2_silver_readiness.schema.json`, `mv2_negative_candidate_review.schema.json` (MV2-05)
- `mv2_ai_adjudication_result.schema.json` (MV2-06)
- `mv2_spectral_baseline_features.schema.json` (MV2-07)
- `mv2_raster_forensics.schema.json` (MV2-08)
- `mv2_canonical_visual_feature.schema.json`, `mv2_dinov2_offline_expansion.schema.json` (MV2-09)
- `mv2_dinov2_visual_embedding_manifest.schema.json`, `mv2_visual_confounder_hardening.schema.json` (MV2-10)

### 13.2 Schemas ausentes

| Schema | Urgência |
|--------|---------|
| `schema_temporal_manifest_unificado.json` — cloud_cover explícito | ALTA |
| `schema_event_geometry_fechado.json` — evento aprovado com geometria não-null | ALTA |
| `schema_negative_formal.json` — negativo com fonte de não-ocorrência | MÉDIA |
| `schema_silver_set.json` — item silver com protocolo formal | MÉDIA |
| `schema_split_trainability.json` — split com anti-leakage | MÉDIA |

---

## 14. Auditoria de outputs públicos e reprodutibilidade

### 14.1 Arquivos em outputs_public

O diretório `outputs_public/` contém documentação leve reproduzível:
- `README.md` — índice geral
- `execution_reports/` — 60+ relatórios MD
- `metrics/` — JSONs de métricas de execução
- `tables/` — CSVs de inventários, manifests, gates
- `mv2/`, `mv2_*` — artefatos MV2-01 a MV2-10 (nos commits MV2)
- `audits/mv2_10_gap_audit/` — **este pacote de auditoria**

**Brutos pesados em outputs_public:** 0 (confirmado em todos os summaries).

### 14.2 Reprodutibilidade por terceiro

Um terceiro com acesso apenas ao Git **não pode reproduzir os embeddings DINOv2**:
- Os 12 embeddings originais derivam de `local_runs/dino_embeddings/v1ge` (git-ignored)
- Os 48 visuais MV2-10 derivam de PNGs canônicos privados (workspace PROJETO)
- Nenhum arquivo `.npz` versionado

O que um terceiro **pode** fazer:
- Re-download das 8 fontes externas via URLs no manifesto (com verificação SHA256)
- Executar scripts read-only que geram relatórios de bloqueio
- Verificar que gates G0-G8 estão bloqueados

**Diferença:**
- Repetibilidade computacional: possível para metadados e manifests
- Replicabilidade científica (re-criar embeddings): **impossível sem workspace privado**

---

## 15. Auditoria de claims e comunicação científica

### 15.1 Claims verificadas como seguras

Os relatórios MV2-01 a MV2-10 mantêm linguagem defensiva adequada:
- `labels_created=0` em todos os summaries ✓
- `can_train=false` em todos ✓
- `ground_truth_operacional_status=ausente` em todos ✓
- `sandbox_status=bloqueado` em todos ✓

### 15.2 Claims que precisam de atenção

| Claim | Arquivo | Risco | Ação |
|-------|---------|-------|------|
| "DINOv2 offline executor status: READY" sem mencionar PNG ≠ espectral | MV2-10 relatório | Leitores inferem extração espectral | Adicionar advertência explícita: embeddings visuais ≠ embeddings espectrais |
| "ai_silver_candidates=8" sem destaque de que não são silver formais | MV2-06 summary | Confundido com silver set | Renomear para ai_exploratory_candidates |
| Distâncias topológicas n=12 apresentadas sem IC ou teste de permutação | MV2-04 output | Lida como resultado estatístico | Adicionar nota obrigatória: "illustrativo; sem significância estatística" |
| "Cronograma Dias 1–17 recuperados" | MV2-06 summary | Lida como progresso científico quando são apenas estrutura de artefatos | Diferenciar: dias de estrutura vs dias de evidência cientificamente fechada |

---

## 16. Auditoria de testes

### 16.1 Suíte atual

- **Testes MV1/v1fx-v1gz**: ~40 arquivos de teste cobrindo pipeline DINO e Protocolo C
- **Testes MV2-01 a MV2-10**: criados via commits MV2 (~25 arquivos)
- **Testes Protocolo C v2ca-v2ch**: ~70 arquivos
- **Testes v2bb-v2bg**: ~60 arquivos (geometria Recife)

**Falhas pré-existentes não relacionadas:**
- `v1lj_v1lq_common` ausente → testes que importam esse módulo travam
- Testes DINO antigos travam por dependência local

### 16.2 Testes críticos faltantes

| Teste | Descrição | Urgência |
|-------|-----------|---------|
| `test_guardrail_unknown_nao_vira_negativo.py` | Assert que falha se unknown→label | CRÍTICA |
| `test_city_as_label_blocked.py` | Assert que falha se cidade==label | CRÍTICA |
| `test_spectral_features_not_visual_rendered.py` | Assert is_spectral_baseline=true antes de uso espectral | CRÍTICA |
| `test_dinov2_coverage_balanced.py` | Assert cobertura REC>50% e PET>50% | ALTA |
| `test_temporal_metadata_completeness.py` | Assert cloud_cover e datetime presentes | ALTA |
| `test_event_geometry_not_null.py` | Assert geometry != null para evento aprovado | ALTA |

---

## 17. Diagnóstico final

Ver arquivo `mv2_10_gap_matrix.csv` para a matriz completa (28 lacunas classificadas).

**Lacunas CRÍTICAS (resolução requerida antes de qualquer avanço):**

| ID | Lacuna | Depende de externo |
|----|--------|-------------------|
| GAP_001 | 0/128 assets com datetime de aquisição | Sim (GEE/Copernicus) |
| GAP_003 | geometry=null em todos os eventos | Sim (Charter758 vetor/digitalização) |
| GAP_004 | Cobertura REC=5.4% PET=6.25% no corpus DINOv2 | Sim (workspace PROJETO) |
| GAP_005 | DINOv2 sobre PNG renderizado, não raster espectral | Sim (GeoTIFFs espectrais) |
| GAP_006 | 0 features espectrais reais | Sim (GeoTIFFs espectrais) |
| GAP_007 | 0 positivos formais | Sim (geometria + revisão humana) |
| GAP_008 | 0 negativos formais | Sim (fonte de não-ocorrência) |
| GAP_014 | Guardrail unknown≠negativo sem teste programático | Não (pode criar agora) |
| GAP_015 | Guardrail cidade≠label sem teste programático | Não (pode criar agora) |
| GAP_016 | Risco PNG/raster sem assert automático | Não (pode criar agora) |
| GAP_020 | Claim "DINOv2 READY" sem mencionar PNG≠espectral | Não (pode corrigir agora) |
| GAP_022 | Silver set=0 com protocolo formal ausente | Sim (cascata de anteriores) |

---

## 18. Plano de ação priorizado

### Ações imediatas (não dependem de dado externo)

1. **Criar testes programáticos de guardrail** (GAP_014, GAP_015, GAP_016): `test_guardrail_unknown_nao_vira_negativo.py`, `test_city_as_label_blocked.py`, `test_spectral_features_not_visual_rendered.py`
2. **Corrigir claim MV2-10** (GAP_020): adicionar advertência explícita que embeddings são visuais (PNG), não espectrais
3. **Documentar pipeline Sentinel→encoder** (GAP_012): registrar bandas, normalização, composição RGB e CRS de entrada para os 12 embeddings originais
4. **Criar schemas ausentes** (GAP_018): `schema_event_geometry_fechado.json` e `schema_temporal_manifest_unificado.json`
5. **Criar test de cobertura balanceada** (GAP_026): `test_dinov2_coverage_balanced.py` — deve **falhar agora** como gate de bloqueio

### Ações dependentes de dado externo — por prioridade

1. **Recuperar scene IDs via histórico GEE** (EXT_002): desbloquearia GAP_001, GAP_002, GAP_009, GAP_013 e a Trilha A inteira
2. **Exportar assets canônicos PNG de Recife e Petrópolis** (EXT_006): desbloquearia GAP_004, GAP_011 e tornaria o corpus DINOv2 válido para análise intercidade
3. **Digitalizar footprint evento REC_2022_05_24_30** (EXT_004/EXT_005): desbloquearia GAP_003, GAP_007 — primeiro gate para positivo formal
4. **Obter GeoTIFFs Sentinel espectrais** (EXT_001): desbloquearia GAP_005, GAP_006, Dia 10 inteiro
5. **Solicitação formal CEMADEN** (EXT_006): desbloquearia G4 para Recife e Curitiba
6. **Geometria Defesa Civil Curitiba** (EXT_011): desbloquearia coorte Curitiba

---

*Esta auditoria é read-only. Nenhum artefato científico foi modificado. Nenhum label, negativo, silver ou ground truth foi criado ou promovido.*
