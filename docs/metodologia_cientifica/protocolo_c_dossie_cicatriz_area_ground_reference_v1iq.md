# Protocolo C — Dossiê de Cicatriz_Area_A: Ground Reference (v1iq)

## Contexto e Justificativa

### Por que v1iq existe

v1ip concluiu: Cicatriz_Area_A.shp é STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK.

**Pergunta:** Será que a análise focada em Cicatriz_Area_A.shp, consolidando todas as evidências disponíveis e lendo os arquivos locais reais, permite promovê-lo a GROUND_REFERENCE_CANDIDATE?

**v1iq responde:** Executando investigação focada com leitura real de DBF, XML sidecar, registries e evidências de v1in.

### Por Que Não Continuar Genericamente

- v1il buscou todos os candidatos locais → 0 encontrados
- v1im consolidou todas as fontes → 20 candidatos bloqueados
- v1in extraiu toda evidência documental → 14 fortes, 0 linkadas
- v1io sintetizou tudo → bloqueado por temporal linkage
- v1ip avaliou composto → STRONG_COMPOSITE_BUT_WEAK_TEMPORAL por falta de linkage explícito

**Decisão:** Investir esforço em Cicatriz_Area_A, o mais promissor, lendo os arquivos reais.

---

## O Que v1iq Faz

### 1. Ler Metadados Locais Reais

- DBF: cabeçalho binário → data de criação, número de registros, campos
- DBF VALORES (R2): leitura registro a registro → valores únicos por campo, busca de termos
- XML sidecar: Cicatriz_Ponto_P.shp.xml → CreaDate, origem, path de produção
- PRJ: CRS (EPSG:31983)

### 2. Ler Registries Públicos Existentes

- v1ij: targeted_official_repository_event_vector_registry.csv → blocking_reason
- v1ik: temporal_provenance_recovery_registry.csv → temporal_status
- v1ip: composite_ground_reference_candidate_registry.csv → v1ip_decision

### 3. Cruzar com Evidências de v1in

- v1in_evidence_strength_decision.csv → total STRONG, linked, PET/2022 de docs internos

### 4. Avaliar 8 Gates com Evidência Real

Gates avaliados:
1. **Geometry** — Shapefile completo
2. **CRS** — Sistema de referência documentado
3. **Observed Status** — Ocorrência, não modelagem
4. **Source Authority** — Fonte oficial (SGB/CPRM)
5. **Event Date OR Survey Date** — Data documentada ligada ao vetor
6. **Document-Vector Linkage** — Vínculo verificável entre documento e vetor
7. **Region Match** — Localidade em ambos
8. **Phenomenon Match** — Fenômeno em ambos

### 5. Tomar Decisão de Promoção

- GROUND_REFERENCE_CANDIDATE: todos 8 PASS/STRONG
- STRONG_COMPOSITE_BUT_WEAK_TEMPORAL: gates 5 ou 6 falham (temporal/linkage)
- Bloqueado: Se falta gate crítico diferente

---

## Resultado: Gates Avaliados (v1iq)

| Gate | Status | Evidência | Nota |
|------|--------|-----------|------|
| **gate_geometry** | ✓ PASS | shp+dbf+shx existem, 444 registros | Bundle completo |
| **gate_crs** | ✓ PASS | SIRGAS_2000_UTM_Zone_23S (EPSG:31983) | Compatível com Petrópolis/RJ |
| **gate_observed_status** | ✓ PASS | TIPO=Deslizamento/Cicatriz; fotointerpretação | Observado, não modelado |
| **gate_source_authority** | ✓ PASS | SGB/CPRM — RIGeo confirmado | HIGH authority |
| **gate_event_or_survey_date** | ✗ FAIL | dbf_date=2015/11/30; sem campo DATA; cicatrizes cumulativas | v1ij confirmou: cumulativas sem data específica |
| **gate_document_vector_linkage** | ✗ WEAK | SIG histórico 2013-2015 publicado como pós-desastre 2022; v1in=0 linkages | Sem vínculo explícito com 2022-02-15 |
| **gate_region_match** | ✓ STRONG | MUNICIPIO=PETRÓPOLIS, UF=RJ, CRS=Zone_23S | Petrópolis confirmado em todas as fontes |
| **gate_phenomenon_match** | ✓ STRONG | TIPO=Deslizamento; movement_of_mass | Movimento de massa confirmado |

**Gates passando: 6/8 | Gates falhando: 2/8**

---

## Evidências Lidas Localmente

### DBF de Cicatriz_Area_A.shp (Petrópolis) — Cabeçalho

- Data de criação: **2015/11/30**
- Número de registros: **444**
- Campos: `GEOMETRIA:C`, `MUNICIPIO:C`, `UF:C`, `TIPO:C`, `CONDICIONA:C`, `FONTE:C`, `AREA_KM2:F`, `OBS:C`
- Campo de data: **NÃO** (nenhum campo DATA, DATE, DT, ANO)

### DBF de Cicatriz_Area_A.shp — Valores dos Atributos (v1iq-R2)

Leitura registro a registro de todos os 444 polígonos (read-only, sem movimentar arquivo):

| Campo | Valores Únicos | Contagem |
|-------|---------------|----------|
| **GEOMETRIA** | `Sim` | 1 valor |
| **MUNICIPIO** | `PETRÓPOLIS` | 1 valor |
| **UF** | `RJ` | 1 valor |
| **TIPO** | `Deslizamento` | 1 valor |
| **CONDICIONA** | `Processo natural` | 1 valor |
| **FONTE** | `Fotointerpretação` | 1 valor |
| **AREA_KM2** | 444 valores numéricos únicos | Áreas em km² |
| **OBS** | *(vazio)* | 0 valores |

**O que os atributos confirmam:**

- ✅ **Fenômeno**: TIPO = `Deslizamento` → movimento de massa observado
- ✅ **Região**: MUNICIPIO = `PETRÓPOLIS`, UF = `RJ`
- ❌ **Fonte institucional**: FONTE = `Fotointerpretação` (método, não instituição) → SGB/CPRM não aparece nos atributos
- ❌ **Vínculo temporal**: nenhum campo com "2022", "levantamento", "vistoria" ou qualquer termo temporal
- ❌ **OBS**: completamente vazio — sem notas, sem referências, sem datas

**Resultado da busca de termos nos atributos:**

| Categoria | Termos encontrados | Campos | Resultado |
|-----------|-------------------|--------|-----------|
| Temporal | nenhum | — | `has_temporal_expression_in_field = FALSE` |
| Fonte | nenhum | — | `has_source_in_field = FALSE` |
| Fenômeno | `deslizamento` | TIPO | `has_phenomenon_in_field = TRUE` |
| Data de evento | nenhum | — | `has_event_or_survey_date_in_field = FALSE` |

**Força da evidência de atributos: `WEAK`** — apenas fenômeno confirmado nos atributos; sem fonte institucional e sem vínculo temporal.

### XML de Cicatriz_Ponto_P.shp (proxy de proveniência do SIG)

- CreaDate: **20150122**
- Origem: `D:\SUSCETIBILIDADE\Correções_Kits_2013\...`
- Nota: **SIG_SUSCETIBILIDADE_2013;FOTOINTERPRETACAO**

### Registry v1ij (blocking_reason)

- `gate_04_no_event_date; cicatrizes_cumulativas_sem_data_especifica`

### v1in (linkages a candidatos)

- 14 STRONG evidence encontradas
- **0 linkadas** a candidatos específicos
- Evidências PET/2022 existentes vêm de documentos internos de metodologia (não SGB primário)

---

## Decisão de Proveniência por Atributos (v1iq-R2)

A auditoria de valores produziu uma decisão de proveniência independente:

```
candidate_asset_name         = Cicatriz_Area_A.shp
records_count                = 444
has_source_in_field          = NO   (FONTE = "Fotointerpretação", não SGB/CPRM)
has_phenomenon_in_field      = YES  (TIPO = "Deslizamento")
has_temporal_expression      = NO   (nenhum campo temporal)
has_event_or_survey_date     = NO   (sem data de evento)
observed_status_from_attr    = OBSERVED_MASS_MOVEMENT
source_lineage_from_attr     = SOURCE_NOT_EXPLICIT_IN_ATTRIBUTE
temporal_link_from_attr      = NO_TEMPORAL_EXPRESSION_IN_ATTRIBUTE
attribute_evidence_strength  = WEAK
promotion_after_attr_audit   = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK
remaining_blocker            = Sem expressão temporal nos atributos; sem vínculo de data
```

Os atributos confirmam fenômeno e região, mas não adicionam fonte institucional nem vínculo temporal. **A decisão não muda.**

---

## Decisão: STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK

### Por Que NÃO é GROUND_REFERENCE_CANDIDATE

Cicatriz_Area_A.shp pertence a um SIG de susceptibilidade de 2013-2015, produzido por fotointerpretação e publicado no RIGeo SGB como "SIG pós-desastre Petrópolis 2022". O SIG foi **publicado** após o desastre de 2022, mas os dados (feições, datas) são de 2013-2015.

**Bloqueios específicos:**

1. **gate_event_or_survey_date = FAIL**
   - DBF date: 2015/11/30 — anterior ao evento de 2022
   - Sem campo de data nas feições
   - Cicatrizes são cumulativas: não é possível discriminar quais são de 2022-02-15

2. **gate_document_vector_linkage = WEAK**
   - Nenhum documento auditado declara explicitamente que Cicatriz_Area_A.shp representa feições mapeadas após 2022-02-15
   - O SIG foi criado em 2013-2015 (XML sidecar confirma)
   - v1in encontrou 0 linkages a candidatos específicos

### O Que Significaria Superar

Para ser GROUND_REFERENCE_CANDIDATE, seria necessário:
- Confirmação explícita em documento oficial SGB/CPRM de que Cicatriz_Area_A.shp inclui feições mapeadas especificamente após 2022-02-15, OU
- Campo de data no shapefile discriminando feições pelo evento de 2022-02-15

Sem essa evidência, o bloqueio persiste com a análise atual.

---

## Invariantes

- ❌ NÃO é ground truth operacional (requer Protocolo B — não iniciado)
- ❌ NÃO é label de treino
- ❌ NÃO libera modelo
- ❌ NÃO implica Protocolo B
- ❌ NÃO implica treino supervisionado
- ❌ NÃO houve e-mail, solicitação ou vistoria

**can_be_ground_reference_candidate = NO**  
**can_be_operational_ground_truth = NO**  
**can_create_training_label = NO**  
**can_train_model = NO**  
**can_reopen_protocol_b = NO**

---

## Hierarquia de Status

```
STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK (v1iq — resultado atual)
├─ 6/8 gates passando
├─ Bloqueio: cicatrizes cumulativas; SIG histórico 2013-2015
└─ Evidência mínima para superar: documento SGB confirmando feições pós-2022-02-15

GROUND_REFERENCE_CANDIDATE (requer evidência adicional)
├─ Todos 8 gates PASS/STRONG
├─ Vínculo explícito: documento ↔ vetor ↔ evento ↔ data
└─ NÃO foi alcançado com evidência atual

GROUND_TRUTH_OPERACIONAL (futuro, hipotético)
├─ Ground reference candidate confirmado
├─ Validação de campo (Protocolo B) realizada
└─ NÃO iniciado

ML LABEL / TRAINING (bloqueado)
├─ Requer múltiplos ground truth operacionais
└─ NÃO iniciado, não planejado
```

---

**Versão:** v1iq-R2 — Focused Ground Reference Dossier + DBF Attribute Value Audit  
**Resultado:** Cicatriz_Area_A.shp = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK  
**Gates passando:** 6/8 | **Gates falhando:** 2/8  
**Bloqueio:** gate_event_or_survey_date (FAIL) + gate_document_vector_linkage (WEAK)  
**Atributos auditados:** 444 registros × 8 campos — fenômeno confirmado, fonte e data não  
**Attribute evidence strength:** WEAK  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
