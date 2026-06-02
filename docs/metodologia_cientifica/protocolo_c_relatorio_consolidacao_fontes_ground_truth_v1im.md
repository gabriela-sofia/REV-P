# Relatório Científico: Consolidação Mestre de Fontes e Auditoria de Precisão (v1im)

## Resumo Executivo

Esta é uma **consolidação crítica do Protocolo C** — após 6 etapas de busca, consolidação, auditoria temporal e varredura local, nenhum candidato vetorial oferece base suficiente para ground truth operacional com a evidência pública e localmente auditada até agora.

**Ground truth operacional:** BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE (lacuna de evidência temporal)  
**ML label:** BLOCKED (sem validação temporal)  
**Próximas ações:** v1in (extração de evidência temporal de documentos existentes) + v1io (síntese final)

---

## 1. Consolidação de Fontes (v1im Outputs)

### 1.1 Fontes Consolidadas (4 total)

| Fonte | Instituição | Tipo | Autoridade | Status | Bloqueador |
|-------|-------------|------|-----------|--------|-----------|
| **SGB_CPRM_PETRÓPOLIS_2022** | SGB/CPRM | Documental (PDF) | HIGH | NOT_READY | SEM GEOMETRIA, SEM DATA EXPLÍCITA |
| **OFFICIAL_REPOSITORIES_V1II** | Múltiplos | Vetorial | HIGH | NOT_READY | SEM DATA EXPLÍCITA |
| **CONSOLIDATION_V1IJ_V1IK** | Protocolo C | Consolidação | HIGH | NOT_READY | SEM DATA EXPLÍCITA |
| **LOCAL_RECOVERY_V1IL** | Local | Varredura | LOW | NOT_READY | NÃO OFICIAL, NÃO RECUPERADO |

### 1.2 Consolidação de Candidatos (20 analisados)

**De onde vieram:**
- **v1if:** 1 (SGB/CPRM checkout → sem vetor)
- **v1ii:** 12 (oficiais → falta data)
- **v1ij:** 18 consolidados (falta data)
- **v1ik:** 18 auditados (confirmado: bloqueio temporal)
- **v1il:** 0 recuperados (camada de pontos de feições de deslizamento fotointerpretadas não em PROJETO)

**Total líquido:** 20 candidatos únicos

### 1.3 Status de Cada Candidato

Todos os 20 candidatos têm **MESMO BLOQUEADOR: TEMPORAL**

```
blocking_gate: "temporal"
blocking_reason: "No explicit event or survey date from official source"
minimum_evidence_needed: "Explicit event date or survey date from official source"
```

**Nenhum candidato tem data documentada de evento ou levantamento.**

---

## 2. Ranking de Fontes por Utilidade

### Por Autoridade

1. **HIGH:** SGB_CPRM, OFFICIAL_REPOSITORIES, CONSOLIDATION
2. **LOW:** LOCAL_RECOVERY

### Por Geometria Oferecida

1. **SIM:** OFFICIAL_REPOSITORIES, CONSOLIDATION (12 + 18 candidates)
2. **NÃO:** SGB_CPRM (PDF only), LOCAL_RECOVERY (nada)

### Por Data Oferecida

1. **EXPLÍCITA:** NENHUMA
2. **INDIRETA:** SGB_CPRM (evento 2022-02-15 do PDF, mas sem data explícita no vetor)

### Por Fenômeno Separável

1. **SIM:** CONSOLIDATION, OFFICIAL_REPOSITORIES (movimento de massa)
2. **NÃO:** SGB_CPRM (genérico), LOCAL_RECOVERY (N/A)

---

## 3. Candidatos Mais Próximos de Ground Reference

### camada original de feições poligonais de deslizamento fotointerpretadas (MAIS PRÓXIMO)

| Propriedade | Status | Detalhes |
|-------------|--------|----------|
| **Geometria** | ✓ PASS | Shapefile .shp+.dbf+.shx completo |
| **CRS** | ✓ PASS | EPSG:31983 documentado |
| **Fenômeno** | ✓ PASS | Movimento de massa explícito |
| **Observação** | ✓ PASS | feições de deslizamento observadas (não modelagem) |
| **Data de evento** | ✗ **FAIL** | SEM DATA DOCUMENTADA |
| **Separação fenomenológica** | ✓ PASS | Apenas movimento de massa |
| **Autoridade de fonte** | ✓ PASS | Oficial (v1ii) |
| **Precisão espacial** | ✓ PASS | Feature-level |

**Lacuna mínima:** UMA data (evento ou levantamento)

**Por que não pode ser ground truth:** Sem data documentada = impossível ligar a Sentinel-1/2, impossível validar fenômeno no tempo correto.

### camada original de pontos de feições de deslizamento fotointerpretadas (NÃO RECUPERADO)

| Propriedade | Status | Detalhes |
|-------------|--------|----------|
| **Encontrado em PROJETO** | ✗ NÃO | v1il não recuperou |
| **Encontrado em oficial** | ✗ NÃO | v1ii não encontrou |
| **Status** | MISSING | Bloqueado duplo: não localizado + sem data |

**Por que não pode ser ground truth:** Não foi sequer recuperado.

### Demais Candidatos (18)

Todos têm MESMO bloqueio: SEM DATA. Nenhuma diferenciação adicional por falta de outro discriminador.

---

## 4. Matriz de Prontidão Completa

### Todos os Gates (Amostra de camada de feições poligonais de deslizamento fotointerpretadas)

| Gate | Status | Observação |
|------|--------|-----------|
| source_authority_gate | PASS | SGB/CPRM é HIGH authority |
| geometry_gate | PASS | Shapefile .shp+.dbf+.shx |
| crs_gate | PASS | EPSG:31983 |
| temporal_gate | **FAIL** | Nenhuma data documentada |
| phenomenon_gate | PASS | Movimento de massa |
| observed_not_modelled_gate | PASS | feições de deslizamento, não risco |
| phenomenon_separation_gate | PASS | Apenas movimento de massa |
| spatial_precision_gate | PASS | Feature-level |
| **overall_ground_truth_readiness** | **NOT_READY** | Bloqueado por temporal |
| **can_be_ground_reference_candidate** | **NO** | Requer data explícita |

### Candidatos por Bloqueador

- **Temporal:** 20/20 (100%)
- **Geometry:** 0/20
- **Phenomenon:** 0/20
- **Observed status:** 0/20
- **Spatial precision:** 0/20

**Conclusão:** O único bloqueador é temporal. Se data for fornecida para camada de feições poligonais de deslizamento fotointerpretadas, todos os outros gates passariam.

---

## 5. O Que Exatamente Falta (Evidência Mínima)

### Para camada de feições poligonais de deslizamento fotointerpretadas

**Evidência mínima necessária:** 
- Data de evento EXPLÍCITA (ex: "2022-02-15") OU
- Data de levantamento EXPLÍCITA (ex: "fotografado em 2022-02-18") 

**De onde poderia vir:**
1. Metadados do arquivo shapefile (XML sidecar)
2. Documentação oficial da fonte (SGB/CPRM)
3. Validação de campo (Protocolo B)
4. Publicação de artigo/tese citando geometria com data

**De onde NÃO pode vir:**
- Data de modificação do arquivo (`mtime`)
- Data de publicação do repositório
- Ano mencionado em documentação genérica
- Inferência de data histórica

### Para Demais Candidatos

Mesmo requisito: data documentada explícita.

---

## 6. Decision: Ground Truth Operacional

### Status Final

**GROUND TRUTH OPERACIONAL:** BLOQUEADO PERMANENTEMENTE NESTA ETAPA

**Razão:** Protocolo C buscou ocorrência observada real, vetorial, datada, institucional e auditável. Encontrou:
- ✓ Vetorial (21 candidatos com geometria)
- ✓ Institucional (fonte SGB/CPRM)
- ✓ Auditável (cadeia de consolidação documentada)
- ✗ **Datada** (nenhum candidato com data documentada)
- ✗ Observação (nenhuma validação de campo)

### Bloqueio não é Genérico

**Não é:** "sem dados"
**É:** "sem data documentada para geometrias existentes"

A geometria existe. O fenômeno é claro. Falta **uma coisa: data.**

---

## 7. Decision: ML Label

### Status Final

**ML LABEL:** BLOQUEADO ATÉ PROTOCOLO B + SPLIT_AND_LEAKAGE_PROTOCOL

**Por quê não pode ser label agora:**
1. Sem data → sem possibilidade de validar contra observações de satélite
2. Sem campo → sem validação de "era realmente assim quando foi observado?"
3. Sem training/test split → leakage garantido
4. Não há supervisão científica → sem contato com especialistas de campo

**Por quê quando será liberado:**
- Protocolo B (validação de campo) fornece data de levantamento + confirmação visual
- SPLIT_AND_LEAKAGE_PROTOCOL implementa validação cruzada e temporal
- Aí sim: label com confiança científica

---

## 8. Observações Qualitativas Finais

### Por que Não Há Evidência Temporal Suficiente

Este não é um fracasso de busca. Esta é uma **lacuna de evidência temporal em fontes públicas e localmente auditadas**.

**feições de deslizamento observáveis existem:**
- Geometria vetorial foi consolidada
- Fenômeno é claro (movimento de massa observado, não risco)
- Instituição oficial confirma evento histórico (2022-02-15)

**Mas falta a ponte temporal:**
- Nenhuma fonte auditada documenta "essa geometria é DE 2022-02-15"
- Nenhuma fonte auditada documenta "essa geometria é da resposta aos deslizamentos de 2022"
- Data e geometria não estão linkadas em nenhuma fonte pública/local consolidada

Sem essa linkagem documentada, não há base suficiente para ground truth operacional com a evidência atual.

### A Conclusão Técnica Correta

"Protocolo C auditou buscas oficiais, consolidação, auditoria temporal e varredura local com fontes públicas e localmente acessíveis. A evidência temporal está ausente ou indireta em todas as fontes consolidadas. Com a evidência pública e localmente auditada até agora, não há base suficiente para ground truth operacional."

**Não é:** "não há dados"
**Não é:** "não há geometria"
**Não é:** "não há instituição confiável"

**É:** "lacuna de evidência temporal documentada entre geometria e data de evento/levantamento em fontes públicas/locais auditadas"

---

## 9. Próximas Etapas Auditáveis do Protocolo C

### v1in — Temporal Evidence Extraction from Existing Documentary Sources

**Objetivo:** Auditar fontes documentárias já consolidadas (PDFs SGB/CPRM, relatórios, metadata) para extrair evidência temporal estruturada.

**Escopo:**
- Ler PDFs e documentação já consolidada (não solicitar)
- Extrair menções de datas, períodos, eventos
- Auditar força de evidência (EXPLICIT vs INDIRECT)
- Gerar registry de linkagem candidato-documento-data
- Sem OCR pesado, sem solicitação externa

**Resultado esperado:** Evidência temporal adicional que poderia passar gates

**Próximo:** Se v1in encontrar data documentada → reexecutar v1im com nova evidência

### v1io — Ground Truth Readiness Final Synthesis

**Objetivo:** Síntese final depois de v1in, consolidando a prontidão completa com toda evidência pública/local auditada.

**Escopo:**
- Agregar evidência de v1if através v1in
- Computar prontidão final com todos os gates
- Declarar status: READY_FOR_REFERENCE ou BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE
- Sem liberar training label
- Sem validação manual
- Sem claim operacional prematuro

**Resultado esperado:** Declaração final da lacuna exata e próximo passo viável

---

## 10. Validações Executadas

- [x] Script v1im roda sem erros
- [x] Outputs locais criados em local_runs/protocolo_c/v1im/
- [x] 3 registries públicos criados
- [x] 3 schemas públicos criados
- [x] can_be_operational_ground_truth = "NO" sempre
- [x] can_create_training_label = "NO" sempre
- [x] Todos os candidatos têm bloqueador identificado (temporal)
- [x] Minimum_evidence_needed é específico
- [x] Sem path privado em arquivos públicos
- [x] local_runs/ não é versionado
- [x] Todos os testes passam (19 v1im + 70 anteriores = 89 total)

---

## Referências

- **v1if:** Official Observed Event Vector Acquisition Audit
- **v1ii:** Targeted Official Repository Event Vector Mining
- **v1ij:** Consolidated Observed Event Vector Evidence
- **v1ik:** Temporal Provenance Recovery
- **v1il:** Deep Local Vector Asset Recovery and Bundle Audit
- **v1im:** Master Source Consolidation and Ground Truth Precision Audit (este)

---

**Data de Conclusão:** 2026-05-23  
**Etapa:** v1im — Master Source Consolidation  
**Status de Ground Truth:** BLOQUEADO POR DEFICIÊNCIA TEMPORAL DOCUMENTADA  
**Status de ML:** BLOQUEADO ATÉ PROTOCOLO B + VALIDAÇÃO CRUZADA  
**Markdown público:** Português  
**Conclusão científica:** Ground truth observacional, vetorial, datado e auditável não está disponível neste dataset com as fontes oficiais consolidadas.**
