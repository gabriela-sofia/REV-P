# Protocolo C — Consolidação Mestre de Fontes e Auditoria de Precisão de Ground Truth (v1im)

## Contexto e Justificativa

### Por que v1im existe

Após 5 estágios de busca, consolidação e auditoria temporal (v1if a v1il), nenhum candidato passou em todos os gates para ground truth. v1im existe para:

1. **Consolidar TODAS as fontes** em uma matriz mestre única
2. **Auditoria de precisão**: não bloqueios genéricos, mas **motivos exatos**
3. **Máxima clareza científica**: exatamente o que cada fonte oferece e o que falta

v1im não é mais um estágio que procura. É o estágio que **consolida e explica com precisão** a lacuna de evidência temporal em fontes públicas e localmente auditadas.

### Diferenças Fundamentais Entre Tipos de Fonte

| Tipo | O que é | Oferece | Não oferece | Pode ser GT? |
|------|---------|---------|------------|------------|
| **Fonte oficial documentária** | Anexos institucionais (PDFs, relatórios) | Confirmação de evento, fonte de autoridade | Geometria vetorial, data de evento | NÃO |
| **Fonte vetorial oficial** | Shapefiles, GeoJSON em repositórios públicos | Geometria, CRS | Data documentada, fenômeno específico | SEM DATA |
| **Fonte consolidada** | v1ij/v1ik: agregação de múltiplas fontes | Geometria, CRS, fenômeno | Data documentada, observação clara | SEM DATA |
| **Fonte local** | Varredura local (v1il) | Possível ativo não encontrado oficialmente | Autoridade institucional | NUNCA AUTOMATICAMENTE |
| **Risco/suscetibilidade** | Mapas de risco modelados | Fenômeno, escala | Observação real, data de evento | **NUNCA** (é modelagem) |

### Por que tipo de fonte não basta

**Fonte forte sem geometria:**
- SGB/CPRM confirma evento de 2022-02-15 em Petrópolis (PDF)
- Mas sem mapa: não é ground truth, é contexto histórico apenas

**Geometria sem data:**
- Shapefile com feições de deslizamento de movimento de massa
- Mas sem data: não sabemos se é 2022, 2015, ou especulação histórica
- Não é ground truth observado

**Data sem geometria:**
- "Evento de 2022-02-15 em Petrópolis confirmado"
- Mas sem mapa: não delimita "onde" ocorreu
- Não é ground truth geolocalizado

**Risco em vez de observação:**
- Mapa de suscetibilidade a movimento de massa
- Não é observação: é *possibilidade*
- Nunca vira ground truth de ocorrência observada

## Definições Precisas do v1im

### Gate de Autoridade de Fonte

**PASS:** Fonte é oficial, de instituição governamental (SGB, CPRM) ou rastreável em repositório público confiável

**FAIL:** Fonte é local, privada, especulativa ou sem proveniência clara

### Gate de Geometria

**PASS:** Ativo vetorial presente (shapefile .shp + .dbf + .shx, GeoJSON, etc.)

**FAIL:** Apenas cartografia (PDF), documento (relatório), ou sem geometria

### Gate de CRS

**PASS:** Coordenadas com sistema de referência documentado (.prj ou header)

**UNKNOWN:** CRS não determinado (não cancela, mas marca como incompleto)

**FAIL:** Sem CRS e sem possibilidade de inferência

### Gate Temporal

**PASS:** Data de evento explícita ou data de levantamento documentada (não data de publicação, não data de arquivo)

**FAIL:** Nenhuma data documentada, apenas pistas indiretas ou data de sistema de arquivo

### Gate de Fenômeno

**PASS:** Fenômeno separável e específico (movimento de massa OU hidrológico, não misto)

**FAIL:** Fenômeno vago, misto (movimento+hidrologia), ou indocumentado

### Gate de Observação vs Modelagem

**PASS:** Ativo é claramente observação de evento real documentado

**FAIL:** Ativo é risco/suscetibilidade (camada modelada) ou proxy especulativo

### Gate de Separação Fenomenológica

**PASS:** Fenômeno hidrológico e movimento de massa são geometrias diferentes

**FAIL:** Fenômenos sobrepostos, mistos ou impossíveis de separar

### Gate de Precisão Espacial

**PASS:** Resolução é feature-level (geometrias individuais, não agregações)

**FAIL:** Agregação regional, rasterizada, ou de resolução baixa demais

### Gate de Usabilidade em Patch-Level

**PASS:** Geometrias podem ser ligadas a patches Sentinel-2 ou similar

**FAIL:** Escala incompatível, projeção não-processável, ou incompatível com dados de observação remota

## Prontidão de Ground Truth (Matriz de Decisão)

### GROUND_REFERENCE_CANDIDATE_READY

Um candidato entra neste status apenas se **todos os 8 gates PASSAM**:

- source_authority_gate = PASS
- geometry_gate = PASS
- crs_gate = PASS
- temporal_gate = PASS
- phenomenon_gate = PASS
- observed_not_modelled_gate = PASS
- phenomenon_separation_gate = PASS
- spatial_precision_gate = PASS

**Resultado esperado até agora:** 0 candidatos

### Bloqueadores Conhecidos

**Bloqueador primário: TEMPORAL**
- Nenhuma fonte oferece data de evento explícita
- Data de publicação ou inferência não conta
- camada de feições poligonais de deslizamento fotointerpretadas: bloqueado porque não há data documental

**Bloqueador secundário: GEOMETRY**
- Fontes documentárias (SGB PDFs) não têm mapa
- Fontes locais não foram recuperadas (v1il)

**Bloqueador terciário: OBSERVED_NOT_MODELLED**
- Risco/suscetibilidade não é ocorrência
- Modelagem não é observação

## Matriz de Consolidação (v1im Output)

### master_ground_truth_source_registry.csv

**Propósito:** Inventário único de todas as fontes consolidadas

**Fontes consolidadas:**
1. `SGB_CPRM_PETRÓPOLIS_2022` — Anexos oficiais do evento
   - Autoridade: HIGH (oficial)
   - Oferece: Confirmação evento, data, região
   - Bloqueia: SEM GEOMETRIA, SEM OBSERVAÇÃO CLARA (é documento)
   - Status: NOT_READY

2. `OFFICIAL_REPOSITORIES_V1II` — Repositórios oficiais (v1ii)
   - Autoridade: HIGH (oficial)
   - Oferece: Candidatos vetoriais
   - Bloqueia: SEM DATA EXPLÍCITA
   - Status: NOT_READY

3. `CONSOLIDATION_V1IJ_V1IK` — Consolidação + auditoria temporal
   - Autoridade: HIGH (Protocolo C)
   - Oferece: Geometria, CRS, fenômeno separável
   - Bloqueia: SEM DATA EXPLÍCITA, MIXED OBSERVED STATUS
   - Status: NOT_READY

4. `LOCAL_RECOVERY_V1IL` — Varredura local
   - Autoridade: LOW (privado, não recuperado)
   - Oferece: Nada
   - Bloqueia: NÃO OFICIAL, NÃO RECUPERADO
   - Status: NOT_READY

### candidate_source_linkage_matrix.csv

**Propósito:** Matriz que liga cada candidato a cada fonte e diz **o que cada fonte suporta**

**Exemplo:** camada original de feições poligonais de deslizamento fotointerpretadas
- Suportado por: CONSOLIDATION_V1IJ_V1IK
- Oferecido: geometria=YES, crs=YES, phenomenon=YES, event_date=NO
- Bloqueio: SEM DATA

### ground_truth_precision_readiness_matrix.csv

**Propósito:** Matriz de prontidão com **bloqueador específico e evidência mínima necessária**

**Exemplo:** camada original de feições poligonais de deslizamento fotointerpretadas
```
blocking_gate: temporal
blocking_reason: No explicit event or survey date
minimum_evidence_needed: Explicit event date or survey date from official source
```

## Invariantes de v1im

```
can_be_operational_ground_truth = "NO" sempre
can_create_training_label = "NO" sempre
pode_criar_proxy = false
pode_inferir_data = false
pode_aceitar_mtime = false
pode_aceitar_risco_como_observação = false
pode_aceitar_PDF_como_vetor = false
precision_máxima_de_bloqueadores = true
sem_ground_truth_nesta_etapa = true
sem_ML_nesta_etapa = true
```

## Próximas Etapas Auditáveis do Protocolo C

### v1in — Temporal Evidence Extraction from Existing Documentary Sources

**Objetivo:** Auditar fontes documentárias existentes (PDFs, relatórios, metadata sidecars) para extrair evidência temporal, sem OCR pesado, sem solicitação externa, sem validação manual como conclusão.

**Escopo:**
- Ler PDFs já consolidados e arquivos de metadata
- Extrair menções de datas, períodos, eventos documentados
- Documentar força de evidência temporal (EXPLICIT, INDIRECT, INSUFFICIENT)
- Gerar registry de evidência temporal extraída
- Não enviar e-mail, não solicitar dados a instituições

**Output:**
- Registry de evidência temporal em documentos
- Matriz candidato-documento com data encontrada
- Decisão por gate: se data extraída passa validação temporal

**Invariantes:**
- Sem inventar data
- Sem OCR pesado por padrão
- Sem solicitação externa
- Sem mudança de status sem gate explícito

### v1io — Ground Truth Readiness Final Synthesis

**Objetivo:** Síntese final da prontidão depois de v1in, consolidando todas as evidências públicas e localmente auditadas.

**Escopo:**
- Agregar evidência de v1if/v1ii/v1ij/v1ik/v1il/v1im/v1in
- Computar prontidão final com todos os gates
- Declarar status final: READY_FOR_REFERENCE ou BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE
- Sem liberar training label
- Sem validação manual externa
- Sem claim operacional prematuro

**Output:**
- Final readiness synthesis matrix
- Blocking reason and minimum evidence per candidate
- Decision: próximo passo técnico ou conclusão do Protocolo C

---

**Versão:** v1im — Master Source Consolidation and Ground Truth Precision Audit  
**Status:** BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE — lacuna de evidência temporal em fontes públicas/localmente auditadas  
**ML Status:** BLOCKED (sem base suficiente para training até validação temporal)  
**Próximo passo:** v1in (temporal evidence extraction from documentos existentes)  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão, precisão máxima de bloqueadores.**
