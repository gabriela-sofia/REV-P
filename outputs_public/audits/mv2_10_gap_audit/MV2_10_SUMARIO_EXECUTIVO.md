# MV2-10 — Sumário Executivo: Estado Científico do REV-P

**Data:** 2026-06-23 | **Branch ativa:** `marco/validacao-label-free-evidencia-estrutural-mv1`  
**Marcos MV2:** commits MV2-01 a MV2-10 em `marco/mv2-10-dinov2-offline-hardening-visual`

---

## Onde o projeto está exatamente no cronograma

O REV-P está no **Dia 8 parcial** de um cronograma de 22 dias. O executor DINOv2 offline foi resolvido (MV2-10), mas o corpus de entrada tem viés extremo de Curitiba (100% vs 5.4% Recife vs 6.25% Petrópolis), que bloqueia qualquer análise intercidade válida. Os Dias 10, 18, 19, 21 e 22 estão bloqueados por ausência de dados, não por falta de infraestrutura.

---

## O que está cientificamente fechado

- **Infraestrutura de auditoria e bloqueio** (MV2-01 a MV2-10): contratos, schemas, manifests, scripts, testes de formato
- **Protocolo fail-closed** para ground truth: gates G0-G8 definidos, todos os bloqueios documentados
- **Piloto label-free n=12** (v1fz/v1ge): 12 embeddings DINOv2 768D, 4/4/4 Curitiba/Petrópolis/Recife, análise de vizinhança e topologia — explicitamente rotulado como piloto exploratório
- **Ontologia de estados de label**, política de evidência negativa, política anti-leakage
- **8 fontes externas** baixadas com SHA256 verificado (IBGE, GeoCuritiba, ANA catálogo, MapBiomas, SGB Petrópolis, NHESS, Charter758 PNG, ANA série vazia)
- **Executor DINOv2 offline** (HF transformers, local_files_only) funcionando localmente

---

## O que está apenas organizado, mas não fechado cientificamente

- **Manifesto temporal** (MV2-02/03): estrutura criada, 128 assets registrados, mas `datetime=null` e `cloud_cover=null` em todos — temporal/lineage são estrutura vazia
- **Adjudicação IA** (MV2-06): 8 silver candidates e 128 negative candidates adjudicados por IA — são exploratórios, não formais
- **Readiness de negativos/silver/splits** (MV2-05): frameworks prontos, valores todos zero
- **Corpus visual** (MV2-08/09/10): 48 features visuais de PNG Curitiba-dominante — são representações de aparência visual, não dados Sentinel espectrais
- **60 embeddings DINOv2 total** pós-MV2-10 — todos sobre PNG visual; não substituem DINOv2 espectral; cobertura Recife e Petrópolis insuficiente

---

## O que está bloqueado

| Bloqueio | Causa raiz | Depende de externo |
|---------|-----------|-------------------|
| **Dia 8 completo** (corpus n≥59 balanceado) | Recife=5.4% Petrópolis=6.25% no corpus PNG/DINOv2 | Sim — PNG dos assets PROJETO |
| **Dia 10** (baseline espectral) | 0 GeoTIFFs espectrais legíveis; 0 features NDWI/NDVI | Sim — GeoTIFFs Sentinel |
| **Dia 18** (evidência observacional) | geometry=null em todos os 9 eventos candidatos | Sim — vetor Charter758/Defesa Civil |
| **Dia 19** (silver set) | formal_silver_items=0; formal_protocol_exists=false | Sim — cascata dos anteriores |
| **Dia 21** (matched/hard negatives) | 0 negativos formais; sem fonte de não-ocorrência | Sim — CEMADEN/ANA/fonte formal |
| **Dia 22** (sandbox supervisionado) | G0-G8 não satisfeitos; trainable_items=0 | Sim — cascata de todos |
| **Trilha A** (série temporal) | 0/128 assets com datetime; 0 com cloud cover | Sim — scene IDs GEE |

---

## Dados externos que faltam

**Bloqueadores críticos (destravam múltiplos gates):**
1. **Scene IDs dos 128 exports GEE** — desbloquearia datetime, cloud cover, lineage, Trilha A inteira
2. **PNG canônicos de Recife (35) e Petrópolis (45)** — desbloquearia corpus DINOv2 balanceado para análise intercidade
3. **Footprint vetorial inundação Recife 2022** — Charter758 vetor (Copernicus EMS) OU digitalização manual do PNG baixado
4. **GeoTIFFs Sentinel espectrais** — necessários para baseline espectral e DINOv2 espectral real

**Importantes:**
5. CEMADEN histórico de alertas (exige LAI formal — cartas criadas)
6. ANA série hidrométrica estação 39187800 (baixada mas vazia em 2022)
7. Geometria Defesa Civil Curitiba jan/2022 (deep crawl realizado; 0 resultado; dossiê de solicitação criado)

---

## Scripts que faltam

| Script | Para que serve | Pode criar agora |
|--------|---------------|-----------------|
| `consolidar_tabela_temporal_por_patch.py` | Tabela mestra datetime+cloud+scene_id | Não (depende scene IDs) |
| `probe_gee_task_history.py` | Recuperar scene IDs do histórico GEE | Não (depende credencial GEE) |
| `exportacao_assets_visuais_recife_petropolis.py` | Balancear corpus PNG | Não (depende workspace PROJETO) |
| `extrator_baseline_espectral.py` | NDWI/NDVI/NDBI de rasters | Não (depende GeoTIFFs) |
| `bootstrap_intra_inter_cidade.py` | IC95 e permutação nas distâncias | Não (depende corpus balanceado) |
| `vincular_patch_evento_spatial.py` | Overlay patch-evento | Não (depende geometria evento) |

---

## Schemas que faltam

| Schema | Urgência |
|--------|---------|
| `schema_temporal_manifest_unificado.json` (com cloud_cover explícito) | ALTA |
| `schema_event_geometry_fechado.json` (evento aprovado geometry≠null) | ALTA |
| `schema_negative_formal.json` | MÉDIA |
| `schema_silver_set.json` | MÉDIA |
| `schema_split_trainability.json` | MÉDIA |

---

## Testes que faltam

| Teste | Urgência | Pode criar agora |
|-------|---------|-----------------|
| `test_guardrail_unknown_nao_vira_negativo.py` | CRÍTICA | **Sim** |
| `test_city_as_label_blocked.py` | CRÍTICA | **Sim** |
| `test_spectral_features_not_visual_rendered.py` | CRÍTICA | **Sim** |
| `test_dinov2_coverage_balanced.py` | ALTA | **Sim** (falhará até corpus balancear) |
| `test_temporal_metadata_completeness.py` | ALTA | **Sim** (falhará até scene IDs chegarem) |
| `test_event_geometry_not_null.py` | ALTA | **Sim** (falhará até geometria chegar) |

---

## Claims que precisam de correção

| Claim arriscado | Arquivo | Correção |
|----------------|---------|---------|
| "READY_OFFLINE_HF_TRANSFORMERS" sem mencionar PNG≠espectral | MV2-10 relatório | Adicionar: "embeddings gerados sobre PNG renderizado (visual), não sobre raster Sentinel espectral" |
| `ai_silver_candidates=8` pode ser lido como silver formal | MV2-06 summary | Adicionar nota: "candidatos exploratórios de IA ≠ silver formal; formal_silver_items=0" |
| Distâncias topológicas n=12 sem IC ou permutação | MV2-04 output | Adicionar: "ilustrativo; sem teste de significância; n=12 insuficiente para inferência" |
| "Dias 1-17 recuperados" no cronograma | MV2-06 summary | Diferenciar: dias de estrutura organizacional vs dias de evidência científica fechada |

---

## Próxima ação recomendada — por prioridade

1. **[Agora, sem dado externo]** Criar 3 testes de guardrail programáticos: `unknown≠negativo`, `cidade≠label`, `PNG≠espectral`. São barreiras que impedem regressão metodológica e podem ser feitas imediatamente.

2. **[Agora, sem dado externo]** Corrigir texto do relatório MV2-10 para deixar explícito que os 48 novos embeddings DINOv2 foram gerados sobre PNG visual, não sobre raster Sentinel espectral.

3. **[Depende de ação humana — GEE]** Recuperar histórico de tasks de export GEE para obter scene IDs dos 128 assets. Isso desbloquearia datetime, cloud cover, lineage e a Trilha A inteira com uma única ação.

4. **[Depende de ação humana — PROJETO local]** Exportar PNG canônicos das cenas Recife e Petrópolis do workspace PROJETO para balancear o corpus DINOv2 visual. Isso tornaria válida qualquer análise intercidade.

5. **[Depende de ação humana — digitalização]** Digitalizar polígono de inundação Recife mai/2022 a partir do raster Charter758 PNG baixado. Esse é o único caminho imediato para geometry≠null no evento mais forte do corpus.

6. **[Depende de infraestrutura]** Obter GeoTIFFs Sentinel espectrais com bandas para os 128 patches — via re-exportação GEE ou download Copernicus com scene IDs recuperados na ação 3.

---

**Decisão metodológica final:** O projeto tem infraestrutura sólida e guardrails corretos. O que falta não é código — é dado. As ações 1-2 podem ser feitas agora; as ações 3-6 dependem de você.
