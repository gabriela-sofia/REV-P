# Relatório Científico: Dossiê de Cicatriz_Area_A — Ground Reference (v1iq)

## Resumo Executivo

**RESULTADO PRINCIPAL: Cicatriz_Area_A.shp PERMANECE COMO STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK**

v1iq-R2 executou investigação focada em Cicatriz_Area_A.shp com leitura real de arquivos locais:
DBF cabeçalho, DBF valores registro a registro (444 × 8 campos), XML sidecar, registries, v1in.

Resultado: 6/8 gates passam, 2 falham por bloqueio temporal. A auditoria de valores dos atributos confirmou fenômeno (TIPO=Deslizamento) e região (MUNICIPIO=PETRÓPOLIS, UF=RJ), mas não encontrou fonte institucional nem expressão temporal em nenhum campo.

**Status:** SÍNTESE FOCADA COMPLETA (v1iq-R2)  
**Candidato:** Cicatriz_Area_A.shp  
**Região:** Petrópolis, RJ (PET)  
**Evento de referência:** 2022-02-15  
**Decisão:** **STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK**  
**Gates Passando:** 6/8  
**Gates Falhando:** 2/8  
**Bloqueio Principal:** gate_event_or_survey_date (FAIL) + gate_document_vector_linkage (WEAK)  
**Attribute Evidence Strength:** WEAK (fenômeno confirmado; fonte e data ausentes nos atributos)

---

## 1. Análise Detalhada de Gates

### Gate 1: Geometry
**Status:** ✓ PASS

Shapefile completo verificado via leitura de metadados:
- .shp, .dbf, .shx existem (geometria, atributos, índice)
- 444 registros no DBF
- Bundle válido

### Gate 2: CRS
**Status:** ✓ PASS

CRS lido do arquivo .prj:
- SIRGAS_2000_UTM_Zone_23S (EPSG:31983)
- Compatível com Petrópolis/RJ

### Gate 3: Observed Status
**Status:** ✓ PASS

Cicatrizes são ocorrências observadas por fotointerpretação:
- TIPO=Deslizamento/Cicatriz (campo DBF)
- FONTE=Fotointerpretação (XML sidecar confirma)
- São observações de campo, não modelagem de risco/susceptibilidade

### Gate 4: Source Authority
**Status:** ✓ PASS

Fonte institucional HIGH authority confirmada:
- SGB/CPRM — Serviço Geológico do Brasil
- Dataset no RIGeo (https://rigeo.sgb.gov.br/handle/doc/22668)
- Publicado como "SIG pós-desastre Petrópolis 2022 — SGB/CPRM"

### Gate 5: Event Date OR Survey Date
**Status:** ✗ FAIL

**Evidência lida do DBF (cabeçalho):**
- Data de criação do arquivo: **2015/11/30**
- Campos presentes: GEOMETRIA, MUNICIPIO, UF, TIPO, CONDICIONA, FONTE, AREA_KM2, OBS
- **Nenhum campo de data** nas feições (sem DATA, DATE, DT, ANO)

**Evidência do registro a registro — valores reais (v1iq-R2):**
- OBS: **completamente vazio** — nenhum dos 444 registros tem observação
- FONTE: `Fotointerpretação` — método de produção, não data nem instituição
- CONDICIONA: `Processo natural` — sem expressão temporal
- Nenhum campo contém "2022", "levantamento", "vistoria", "pós-desastre" ou qualquer termo temporal
- `has_temporal_expression_in_field = FALSE` (confirmado após leitura de 444 × 8 campos)
- `has_event_or_survey_date_in_field = FALSE`

**Evidência do registry v1ij:**
- blocking_reason: `gate_04_no_event_date; cicatrizes_cumulativas_sem_data_especifica`

**Conclusão:** As cicatrizes são cumulativas e históricas. O DBF foi criado em 2015, anterior ao evento de 2022-02-15. Sem campo de data nas feições, sem notas em OBS, sem qualquer expressão temporal nos atributos. O bloqueio temporal é estrutural e confirmado em três camadas independentes (cabeçalho DBF, valores DBF, registry v1ij).

### Gate 6: Document-Vector Linkage
**Status:** ✗ WEAK

**Evidência do XML sidecar (Cicatriz_Ponto_P.shp.xml):**
- CreaDate: **20150122**
- Origin path: `D:\SUSCETIBILIDADE\Correções_Kits_2013\...`
- Nota detectada: **SIG_SUSCETIBILIDADE_2013;FOTOINTERPRETACAO**

**O SIG foi criado no contexto do mapeamento de susceptibilidade de 2013-2015** e republicado como "pós-desastre 2022". Os dados são históricos.

**Evidência de v1in:**
- 14 evidências STRONG encontradas em documentos locais
- **0 linkadas** a candidatos específicos (Cicatriz_Area_A)

**Conclusão:** Nenhum documento auditado declara explicitamente que Cicatriz_Area_A.shp representa feições mapeadas especificamente após 2022-02-15. O vínculo é fraco (WEAK).

### Gate 7: Region Match
**Status:** ✓ STRONG

Localidade confirmada em todas as fontes auditadas:
- DBF cabeçalho: estrutura presente com 444 registros em Petrópolis
- DBF valores (v1iq-R2): MUNICIPIO = `PETRÓPOLIS` (1 valor único), UF = `RJ` (1 valor único)
- CRS: Zone_23S (compatível com RJ)
- Registry v1ij: region=PET

### Gate 8: Phenomenon Match
**Status:** ✓ STRONG

Fenômeno confirmado em todas as fontes:
- DBF valores (v1iq-R2): TIPO = `Deslizamento` (1 valor único em 444 registros)
- CONDICIONA = `Processo natural` (todos os registros)
- Registry v1ij: phenomenon=movement_of_mass
- Documento SGB: cicatrizes de deslizamento em Petrópolis
- `observed_status_from_attributes = OBSERVED_MASS_MOVEMENT`

---

## 2. Auditoria de Atributos — Valores Reais do DBF (v1iq-R2)

### O Que Foi Lido

444 registros × 8 campos lidos em modo read-only (sem mover, copiar ou modificar o shapefile):

| Campo | Valores Únicos Encontrados | Termos de Fonte | Termos Temporais | Termos de Fenômeno |
|-------|--------------------------|-----------------|-----------------|-------------------|
| GEOMETRIA | `Sim` | — | — | — |
| MUNICIPIO | `PETRÓPOLIS` | — | — | — |
| UF | `RJ` | — | — | — |
| TIPO | `Deslizamento` | — | — | ✓ deslizamento |
| CONDICIONA | `Processo natural` | — | — | — |
| FONTE | `Fotointerpretação` | ✗ (método, não instituição) | — | — |
| AREA_KM2 | 444 valores numéricos únicos | — | — | — |
| OBS | *(vazio — 0 valores)* | — | — | — |

### Conclusão da Auditoria de Atributos

```
has_source_in_field          = NO
has_phenomenon_in_field      = YES
has_temporal_expression      = NO
has_event_or_survey_date     = NO
attribute_evidence_strength  = WEAK
promotion_after_attr_audit   = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK
```

**Por que `has_source_in_field = NO`:** O campo FONTE contém `Fotointerpretação` — que é o método de produção das cicatrizes, não a instituição. SGB, CPRM, RIGeo não aparecem em nenhum campo do DBF. A autoridade institucional vem do registro externo (v1ij/RIGeo), não dos atributos do shapefile.

**Por que `attribute_evidence_strength = WEAK`:** Apenas fenômeno confirmado nos atributos. Fonte e data ausentes. Isso é consistente com um SIG de susceptibilidade que documenta o fenômeno observado, mas não vincula cada feição a uma data ou instituição específica.

---

## 3. Consolidação de Evidência Composta

### Componentes Verificados

| Componente | Status | Evidência Auditada |
|-----------|--------|--------------------|
| **Vetor** | ✓ SIM | Shapefile completo (geometry + CRS) — lido via DBF + PRJ |
| **Fenômeno** | ✓ SIM | Movimento de massa / cicatrizes — TIPO no DBF |
| **Fonte Oficial** | ✓ SIM | SGB/CPRM (HIGH authority) — RIGeo URL confirmado |
| **Region Match** | ✓ SIM | Petrópolis (ambos) — DBF + registry |
| **Phenomenon Match** | ✓ SIM | MM/cicatrizes (ambos) — DBF + registry |
| **Evento Documentado (2022)** | ✗ NÃO-LINKADO | Data existe em documentos, mas SIG é de 2013-2015 |
| **Data no Vetor** | ✗ NÃO | DBF date=2015; sem campo DATA; cicatrizes cumulativas |
| **Vínculo Explícito Doc→Vetor** | ✗ FRACO | Sem documento que declare linkage pós-2022 |

### Conclusão de Composição

Evidência composta é STRONG em 6 aspectos, FRACA em 2 aspectos críticos (temporal e linkage).

O problema não é ausência de dados — é que os dados existentes contradizem o evento de referência:
- O SIG foi criado em 2013-2015 (não em resposta a 2022-02-15)
- Cicatrizes são cumulativas e históricas
- Não há campo de data que discrimine evento específico

---

## 4. Decisão: STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK

### Critério Faltante

Para ser GROUND_REFERENCE_CANDIDATE:

```
Cicatriz_Area_A.shp + documento oficial SGB + "data 2022-02-15"
+ vínculo EXPLÍCITO entre eles
= GROUND_REFERENCE_CANDIDATE
```

**Atualmente:**

```
✓ Cicatriz_Area_A.shp (completo)
✓ Fonte SGB/CPRM (HIGH authority)
✓ Região Petrópolis confirmada
✓ Fenômeno movimento de massa confirmado
✗ Data de evento: SIG criado em 2013-2015 (DBF: 2015/11/30)
✗ Vínculo explícito: XML confirma SIG_SUSCETIBILIDADE_2013, não resposta pós-2022
```

### Diferença em Relação à Análise v1ip

| Aspecto | v1ip | v1iq |
|--------|------|------|
| **Escopo** | Avaliação composta | Leitura real de arquivos locais |
| **gate_event_or_survey_date** | FAIL (raciocínio geral) | FAIL (DBF date=2015; cumulative confirmado via v1ij) |
| **gate_document_vector_linkage** | WEAK (sem linkage explícito) | WEAK (XML confirma SIG_SUSCETIBILIDADE_2013) |
| **Decisão** | STRONG_COMPOSITE_BUT_WEAK_TEMPORAL | STRONG_COMPOSITE_BUT_WEAK_TEMPORAL |

O que mudou: v1iq confirma com leitura direta de arquivos que o bloqueio não é por falta de busca — é estrutural (SIG histórico).

---

## 5. Evidência Mínima para Superar o Bloqueio

Confirmação explícita de que Cicatriz_Area_A.shp inclui feições mapeadas especificamente após 2022-02-15 em documento oficial SGB/CPRM, ou campo de data no shapefile discriminando feições por evento.

Essa evidência não está disponível com os arquivos auditados.

---

## 6. Invariantes Atendidos

- [x] Nenhum label foi criado
- [x] Nenhum modelo foi treinado
- [x] Protocolo B não foi reaberto
- [x] can_create_training_label = NO (sempre)
- [x] can_train_model = NO (sempre)
- [x] can_be_operational_ground_truth = NO (sempre)
- [x] can_reopen_protocol_b = NO (sempre)
- [x] can_be_ground_reference_candidate = NO
- [x] Sem inventar vínculo
- [x] Sem aceitar contexto genérico
- [x] Sem usar data de arquivo/pasta como evento
- [x] Sem aceitar risco como observado
- [x] Sem e-mail, solicitação ou vistoria
- [x] Sem path privado em arquivos públicos
- [x] local_runs/ não versionado

---

## 7. Contribuição de v1iq-R2

v1iq-R2 demonstrou que:

1. **Bloqueio temporal é estrutural** — não é apenas ausência de busca. O SIG foi confirmado como histórico (2013-2015) via leitura real do XML sidecar (CreaDate=20150122).

2. **Cicatrizes são cumulativas** — 444 feições sem campo de data, criadas em contexto de mapeamento de susceptibilidade, não de resposta ao evento de 2022.

3. **OBS completamente vazio** — nenhum dos 444 registros tem qualquer observação. Descarta hipótese de que notas textuais pudessem conter datas ou referências ao evento de 2022-02-15.

4. **FONTE = "Fotointerpretação"** — o campo FONTE registra o método de produção (fotointerpretação de imagens), não a instituição. SGB/CPRM não aparece nos atributos do shapefile. A autoridade institucional vem do metadado externo (RIGeo), não dos campos vetoriais.

5. **v1ip estava correto** — a análise focada com leitura de valores reais não alterou a decisão, apenas aprofundou e consolidou a evidência do bloqueio com dados concretos.

6. **Há valor metodológico** — o processo demonstra como gate avaliação com leitura real de metadados e de valores de atributos funciona na prática: três camadas independentes (cabeçalho DBF, valores DBF, registry v1ij) confirmam o mesmo bloqueio.

---

## 8. Status Final

**Commitável?** SIM, como registro de análise metodológica

**Razão:**
- v1iq confirma o resultado de v1ip via leitura real de arquivos
- A decisão STRONG_COMPOSITE_BUT_WEAK_TEMPORAL é auditável e rastreável
- Documentação explica o bloqueio com precisão operacional
- Registry de dossiê oferece artefato técnico para continuidade futura

**Se Commitar v1iq:**
- Incluir script, tests, docs, datasets (dossier + gate matrix)
- Manter local_runs/ fora do versionamento
- Mensagem de commit: "Cicatriz_Area_A confirmada como STRONG_COMPOSITE_BUT_TEMPORAL_LINK_WEAK via leitura real de metadados"

---

**Data de Execução:** 2026-05-23  
**Etapa:** v1iq-R2 — Focused Ground Reference Dossier + DBF Attribute Value Audit  
**Decisão Final:** Cicatriz_Area_A.shp = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK  
**Gates Passando:** 6/8 | **Gates Falhando:** 2/8  
**Bloqueadores:** gate_event_or_survey_date (FAIL) + gate_document_vector_linkage (WEAK)  
**Atributos lidos:** 444 registros × 8 campos — fenômeno confirmado; fonte e data ausentes  
**Attribute Evidence Strength:** WEAK  
**has_event_or_survey_date_in_field:** NO — confirmado  
**OBS nos registros:** VAZIO — descarta hipótese de notas com datas  
**Conclusão:** Bloqueio temporal confirmado em três camadas independentes  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
