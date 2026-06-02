# Protocolo C — Referência Composta de Ground Truth (v1ip)

## Contexto e Justificativa

### Por que v1ip existe

v1io declarou: BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE.

Bloqueador específico identificado: data documentada não está linkada a geometria candidato em fontes públicas auditadas.

**Problema com essa conclusão:** Exigir que data esteja DENTRO do arquivo shapefile (como atributo ou metadata) é excessivamente restritivo para ground truth observacional real. Na prática, observações de campo frequentemente têm:
- Vetor criado em data X (shapefile foi gerado em 2023)
- Evento documentado em data Y (deslizamento ocorreu em 2022-02-15)
- Linkage entre eles via: relatório oficial, nome de arquivo, metadados de proveniência

v1ip reconhece essa realidade e permite **evidência composta**: vínculo rastreável entre:
- Vetor observado (geometria + CRS)
- Fonte oficial (SGB/CPRM)
- Relatório/documento técnico (PDF anexo)
- Evento datado (2022-02-15)
- Fenômeno documentado (movimento de massa)
- Território (Petrópolis)
- Proveniência (SGB criou o vetor em resposta ao evento)

### Diferença Crítica: Por que Data Não Precisa Estar no Vetor

| Cenário | Aceitável? | Razão |
|---------|-----------|-------|
| **Data NO vetor** (atributo shapefile) | ✓ SIM (ideal) | Máxima rastreabilidade |
| **Data NO documento oficial linkado** | ✓ SIM (aceitável) | Vínculo forte, proveniência clara |
| **Data NO ambiente (pasta, arquivo nome)** | ✗ NÃO (insuficiente) | Sem proveniência oficial |
| **Data extraída do relatório sem vínculo** | ✗ NÃO (insuficiente) | Data pode ser genérica |
| **Data inferida (ex: "foi em 2022")** | ✗ NÃO (proibido) | Viola princípio "sem inventar data" |

---

## O Que v1ip Faz

### 1. Identificar Candidatos Fortes

Prioriza:
- camada original de feições poligonais de deslizamento fotointerpretadas (passou 7/8 gates em v1ik, bloqueado por temporal)
- Qualquer vetor com geometria+CRS+fenômeno+observed_not_risk provável
- Candidatos com bloqueio temporal mas fonte oficial STRONG

### 2. Construir Dossiês Compostos

Para cada candidato, agrega:
- **Vetor:** geometria, CRS, tipo
- **Fonte:** instituição, documento oficial
- **Evento:** data, fenômeno, região
- **Proveniência:** vínculo entre vetor e evento
- **Temporal:** data do evento, data de levantamento se existir

### 3. Auditar Vínculo Composto

Avalia 8 sub-gates:
1. **Geometry Gate** — Vetor existe e é válido
2. **CRS Gate** — Sistema de referência documentado
3. **Observed Status Gate** — É ocorrência, não modelagem
4. **Source Authority Gate** — Fonte oficial (HIGH authority)
5. **Event Date Or Survey Date Gate** — Data documentada (evento OU levantamento)
6. **Document-Vector Linkage Gate** — Vínculo entre documento e vetor é forte
7. **Region Match Gate** — Região mencionada em documento E vetor
8. **Phenomenon Match Gate** — Fenômeno mencionado em documento E vetor

### 4. Produzir Decisão Conservadora

Classifica como:
- **GROUND_REFERENCE_CANDIDATE** — Todos os 8 gates STRONG
- **STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK** — 7/8 STRONG, temporal linkage MODERATE/WEAK
- **VECTOR_OBSERVED_BUT_EVENT_LINK_INSUFFICIENT** — Vetor OK, mas evento não linkado
- **DOCUMENTED_EVENT_BUT_VECTOR_LINK_INSUFFICIENT** — Evento OK, mas vetor não linkado
- **CONTEXT_ONLY** — Apenas contexto histórico
- **NOT_USABLE** — Falha em múltiplos gates

---

## Status Possíveis (Definições)

### GROUND_REFERENCE_CANDIDATE

**Critérios:**
- Geometria: YES
- CRS: YES
- Fenômeno: documentado
- Observado (não risco): YES
- Data de evento OU data de levantamento: documentada
- Vínculo documento-vetor: STRONG
- Lineage de fonte: STRONG
- Matches (região, fenômeno): STRONG

**Implicação:** Candidato pode entrar em validação cruzada com Sentinel. Próximo passo seria Protocolo B.

**Não é:** Ground truth operacional. Requer validação de campo.

### STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK

**Critérios:**
- Geometria: YES
- CRS: YES
- Fenômeno: documentado
- Observado: YES
- Data: documentada
- Vínculo documento-vetor: MODERATE (existe, mas não explícito)
- Matches: STRONG

**Implicação:** camada de feições poligonais de deslizamento fotointerpretadas cai aqui. Tem evidência forte composta, mas falta link explícito tipo "camada original de feições poligonais de deslizamento fotointerpretadas foi gerado em 2022-02-15".

**Próximo passo:** OCR manual em PDF SGB ou acesso a PROJETO para metadados sidecars.

---

## Aplicação a camada original de feições poligonais de deslizamento fotointerpretadas

### Componentes Dispon íveis

| Componente | Status | Detalhes |
|-----------|--------|----------|
| **Vetor** | ✓ SIM | Shapefile completo (.shp+.dbf+.shx) |
| **CRS** | ✓ SIM | EPSG:31983 documentado |
| **Fenômeno** | ✓ SIM | Movimento de massa (feições de deslizamento observadas) |
| **Observado** | ✓ SIM | Ocorrência real, não risco |
| **Fonte** | ✓ SIM | SGB/CPRM (HIGH authority) |
| **Evento Documentado** | ✓ SIM | 2022-02-15 (v1if/SGB confirmaram) |
| **Região Match** | ✓ SIM | Petrópolis (ambos) |
| **Fenômeno Match** | ✓ SIM | Movimento de massa (ambos) |
| **Linkage Documento-Vetor** | ✗ FRACO | Não explícito em fontes auditadas |

### Decisão v1ip

**Status:** STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK

**Razão:** Evidência composta é forte (7/8 componentes), mas falta link explícito entre "2022-02-15" e "camada original de feições poligonais de deslizamento fotointerpretadas".

**Bloqueador:** `document_vector_package_link` é MODERATE, não STRONG

**Evidência Mínima para Destravar:**
```
Uma das seguintes linkagens explícitas:
1. "camada original de feições poligonais de deslizamento fotointerpretadas foi derivado de observações do evento 2022-02-15"
   em documento oficial SGB
2. Metadata XML do shapefile mencionando "2022-02-15" ou "evento pós-deslizamento"
3. OCR confiável em PDF SGB linkando "camada de feições poligonais de deslizamento fotointerpretadas" a data/período do evento
```

---

## Por Que v1ip Não Libera ML/Treino

Mesmo que camada de feições poligonais de deslizamento fotointerpretadas fosse promovido a GROUND_REFERENCE_CANDIDATE:

| Requisito ML | Status | Por Quê |
|------------|--------|--------|
| **Ground Truth** | FALTA | Ainda requer validação de campo (Protocolo B) |
| **Training Set** | FALTA | 1 candidato não é dataset |
| **Test Set** | FALTA | Sem divisão temporal/espacial |
| **Supervision** | FALTA | Sem especialista validando |
| **Cross-validation** | FALTA | Sem Sentinel-1/2 validado contra data |

**Conclusão:** v1ip produz GROUND_REFERENCE_CANDIDATE (candidato para referência), não GROUND_TRUTH_OPERATIONAL (verdade de campo validada).

---

## Diferenças Entre Status

```
GROUND_REFERENCE_CANDIDATE (v1ip)
├─ Vetor é observado
├─ Fenômeno é claro
├─ Evento é documentado
├─ Linkage entre eles é verificável
└─ Próximo passo: Protocolo B / Sentinel cross-validation

GROUND_TRUTH_OPERACIONAL (antes era impossível)
├─ Ground reference candidate confirmado
├─ Validação de campo (Protocolo B) realizada
├─ Observação de satélite (Sentinel) validada
├─ Sem conflitos ou inconsistências
└─ Pronto para training supervisionado

ML LABEL/TREINO (bloqueado, permanece bloqueado)
├─ Requer ground truth operacional
├─ Requer múltiplos candidatos validados
├─ Requer training/test split sem leakage
└─ Requer supervisão científica
```

---

## Regras Operacionais de v1ip

### Permitido

✓ Avaliar vínculo entre vetor e documento oficial  
✓ Usar data de documento externo se vínculo for forte  
✓ Auditar proveniência (SGB criou o vetor em resposta ao evento)  
✓ Aceitar "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK" como status intermédio  
✓ Documentar exatamente qual vínculo falta  

### Proibido

✗ Inventar vínculo entre vetor e data  
✗ Aceitar contexto genérico ("2022 foi um ano de chuvas")  
✗ Usar data de arquivo/pasta como prova  
✗ Aceitar risco/suscetibilidade como observado  
✗ Liberar label/target/treino  
✗ Reabrir Protocolo B como recomendação  
✗ Enviar e-mail ou criar solicitação  

---

## Próximos Passos Viáveis (Fora de v1ip)

Se v1ip encontrar GROUND_REFERENCE_CANDIDATE (improvável com dados atuais):
1. Reexecução de v1io com nova classificação
2. Preparação para Protocolo B (vistoria de campo)
3. Análise temporal Sentinel-1/2 para validação

Se v1ip confirmar STRONG_COMPOSITE_BUT_TEMPORAL_LINK_WEAK:
1. Acesso a PROJETO para metadados sidecars
2. OCR manual em PDFs SGB
3. Contato com SGB para documentação de linkage

---

**Versão:** v1ip — Composite Ground Reference Evidence Builder  
**Status:** Construtor de Dossiês Compostos  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
