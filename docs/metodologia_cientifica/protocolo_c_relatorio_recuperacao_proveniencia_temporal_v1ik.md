# Protocolo C — v1ik: Relatório de Recuperação de Proveniência Temporal

**Data de Execução:** 2026-05-23  
**Versão:** v1ik-R1  
**Status Final:** `TEMPORAL_GATE_RECOVERY_AUDIT_COMPLETE`

---

## 1. Resumo Executivo

v1ik realizou auditoria temporal conservadora de candidatos bloqueados em v1ij. Nota: apenas candidatos presentes no `consolidated_observed_event_vector_candidate_registry.csv` podem ser auditados. Cicatriz_Ponto_P.shp não foi consolidado em v1ij, portanto não foi auditado.

**Resultado:**
- **1 candidato revisado** (Cicatriz_Area_A.shp, consolidado em v1ij)
- **0 candidatos** com evidência temporal forte documentada
- **0 candidatos** que avançaram em gate temporal
- **1 candidato** continua bloqueado: `TEMPORAL_GATE_BLOCKED_NO_DATE`

**Conclusão:** Nenhuma data de evento foi recuperada em fontes públicas/locais auditáveis para o único candidato consolidado. Bloqueio persiste com motivação documentada. Cicatriz_Ponto_P.shp requer consolidação em v1ij antes de auditoria temporal.

---

## 2. Candidatos Revisados

### Cicatriz_Area_A.shp (Prioritário)

**Status de Entrada (v1ij):**
- Geometry: YES (444 features)
- CRS: EPSG (válido)
- Phenomenon: mass_movement (cicatriz)
- Observed not risk: YES
- **Bloqueador:** `gate_04_no_event_date`

**Busca Temporal em v1ik:**

#### 1. Sidecars Locais
- **Resultado:** Nenhum sidecar local acessível (.prj, .xml, .dbf)
- **Motivo:** Candidato é referência remota em RIGeo, não arquivo local presente
- **Achado:** Nenhum metadado temporal local

#### 2. Registries Versionados (v1if, v1ii)
- **Consulta:** `official_observed_event_vector_registry.csv`
  - Encontrado: referência a `Cicatriz_Area_A.shp` em notas v1ii
  - Informação: "Cicatrizes de deslizamento consolidadas, mas sem data específica de ocorrência"
  - **Contribuição temporal:** NENHUMA (confirma bloqueio, não resolve)

- **Consulta:** `targeted_official_repository_event_vector_registry.csv`
  - Encontrado: RIGEO_PET_002 com recurso `Cicatriz_Area_A.shp`
  - Informação: "Shapefiles de cicatriz auditados localmente em v1ih. Sem campo de data de evento."
  - **Contribuição temporal:** NENHUMA (confirma bloqueio)

#### 3. Documentação Versionada
- **Consulta:** README.md, docs/metodologia_cientifica/
  - Encontrado: Referências a "cicatriz", "2022", "SGB/CPRM"
  - Informação: Contexto de Petrópolis fevereiro 2022, mas sem data explícita de cicatriz
  - **Classificação:** WEAK_FILE_OR_FOLDER_HINT (apenas contexto de evento, não data de levantamento)

#### 4. Metadados Públicos RIGeo (indiretamente via registry)
- **Referência:** URL em registry aponta a RIGeo
- **Informação:** "avaliação pós-desastre Petrópolis 2022"
  - Esta é data de publicação (2022), não data de evento específica
  - Não há data de coleta/levantamento documentada
  - Não há vinculação explícita entre cicatriz e 2022-02-15

**Status Temporal Após Revisão:**
- `temporal_status_after_review = TEMPORAL_GATE_BLOCKED_NO_DATE`
- `event_date_compatible_after_review = UNKNOWN` (sem data para validar)
- `temporal_confidence = LOW`
- `accepted_as_event_date = NO`
- `accepted_as_survey_date = NO`
- `accepted_as_context_only = PARTIAL` (contexto: avaliação pós-desastre 2022, mas sem data específica)

**Bloqueios Persistentes:**
1. Sem data explícita de evento no vetor
2. Sem campo de data em atributos do shapefile
3. Sem metadado técnico documentando data de levantamento
4. Sem documento oficial vinculando cicatriz a data específica
5. Disponível apenas: "após desastre fevereiro 2022" sem precisão temporal

**Próxima Ação Recomendada:**
- Se SGB/CPRM publicar documentação técnica com data de levantamento → reprocessar
- Se sidecars locais forem adquiridos com metadados → reprocessar
- Caso contrário: manter bloqueado; é honesto e auditável

---

## 3. Cicatriz_Ponto_P.shp (Não Encontrado em v1ij)

**Status:** Não revisado em v1ik-R2 porque não consolidado em v1ij.

Cicatriz_Ponto_P.shp foi mencionado na metodologia como candidato promissor, mas:
- Não foi encontrado nos registries v1if, v1ii ou v1ih durante v1ij
- Portanto, não existe no `consolidated_observed_event_vector_candidate_registry.csv`
- v1ik só pode auditar candidatos presentes no registry consolidado

**Próxima ação recomendada:**
- Se Cicatriz_Ponto_P.shp for localizado em repositório v1if/v1ii/v1ih:
  1. Consolidar em v1ij com metadata de origem
  2. Auditar temporalmente em v1ik (nova iteração)
- Caso contrário: manter como candidato futuro se descoberto

---

## 4. Evidências Temporais Encontradas

### Resumo por Tipo

| Tipo de Evidência | Encontrados | Status |
|---|---|---|
| Data de evento explícita | 0 | NENHUM |
| Data de levantamento explícita | 0 | NENHUM |
| Janela de evento documentada | 0 | NENHUM |
| Vínculo documentado | 0 | NENHUM |
| Referência cruzada em registry | 1 (confirma bloqueio) | CONFIRMA BLOQUEIO |
| Sidecar com metadado | 0 | NENHUM |
| Contexto em documentação | 1 ("2022") | FRACO |

### Evidências Rejeitadas

| Evidência Candidata | Razão da Rejeição | Classificação |
|---|---|---|
| "2022" em nome de pasta/arquivo | Pista isolada, não prova específica | WEAK_FILE_OR_FOLDER_HINT |
| Data de publicação em RIGeo | Data de repositório, não de evento | INVALID_FOR_EVENT_DATE |

---

## 5. Status Final dos Melhores Candidatos

### Cicatriz_Area_A.shp

**Status Consolidado:**
- Geometry: ✓ YES
- CRS: ✓ Presente
- Phenomenon: ✓ YES (cicatriz)
- Observed not risk: ✓ YES
- Temporal/Date: ✗ BLOCKED

**Ground Truth Candidate Status:** `BLOCKED` (não avançou)  
**Patch Binding Preflight Status:** `BLOCKED` (não avançou)  
**Can Create Training Label:** `NO` (sempre)

**Motivo Consolidado de Bloqueio:**
```
gate_04_no_event_date (persistente) +
temporal_gate_blocked_no_date (confirmado em v1ik)
= TEMPORAL_GATE_BLOCKED_NO_DATE
```

---

## 6. Recuperação de Gates Temporais

**Resultado:** 0 gates recuperados.

Nenhum candidato passou para `TEMPORAL_GATE_PASSED` ou mesmo `TEMPORAL_CONTEXT_STRENGTHENED`.

**Razão:** Sem data de evento documentada (apenas contexto "2022"), gate temporal mantém bloqueio.

---

## 7. Patch Binding Preflight Reabertura

**Resultado:** Nenhum candidato reabriu patch binding preflight.

**Motivo:** `can_reopen_patch_binding_preflight = NO` para todos.

Mesmo com contexto temporal, falta data explícita.

---

## 8. Ground Truth Operacional

**Status:** `BLOCKED` (sem mudança desde v1ij).

Nenhum candidato passou para `CANDIDATE_OBSERVED_GROUND_TRUTH`.

**Razão:** Gate temporal continua não passando.

---

## 9. ML Label e Training

**Status:** `BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL` (sem mudança).

**Regra:** `can_create_training_label = NO` (sempre).

Mesmo se gate temporal fosse recuperado, label requer múltiplas validações adicionais.

---

## 10. Testes Executados

✓ Script existe e roda  
✓ Registry temporal criado  
✓ Schema temporal compatível  
✓ Matriz temporal criada  
✓ Schema da matriz compatível  
✓ Outputs locais gerados  
✓ Data de sistema de arquivos nunca aceita  
✓ Nome de pasta/arquivo nunca STRONG  
✓ Data de publicação não vira data de evento  
✓ Data de levantamento não vira data de evento  
✓ Só data explícita pode passar  
✓ `can_create_training_label` sempre `NO`  
✓ Se nenhum candidato tem data forte, patch binding continua bloqueado  
✓ Sem label/target supervisionado  
✓ Sem path privado em arquivo público  

---

## 11. Limitações Reconhecidas

1. **Sem Acesso a Sidecars Não-Versionados:**
   - v1ik é read-only e não faz download
   - Se sidecars com metadatos existem em repositório remoto, não foram acessados
   - Seria necessário etapa futura de aquisição controlada

2. **Documentação Técnica Não Digitalizadas:**
   - PDFs de avaliação pós-desastre baixados em v1if não foram OCRizados
   - Poderiam conter datas, mas OCR pesado é fora de escopo v1ik

3. **Sem Solicitação Externa:**
   - v1ik não envia e-mail a SGB/CPRM
   - Não abre solicitação formal
   - Mantém isolamento de fontes públicas

4. **Dados Cumulativos:**
   - Cicatrizes podem representar múltiplos eventos
   - Sem data por cicatriz individual, não é possível separar

---

## 12. Próximos Passos Técnicos

### Curto Prazo (Tech)
1. **Revisar iterativamente:** Se novos repositórios tiverem metadados com data
2. **Sidecar aquisição local:** Se documentação técnica for adquirida localmente
3. **Revisit após v1iii:** Discovery expandida pode trazer novos candidatos com data

### Médio Prazo
4. **Considerar Protocol A:** Sentinel raw pixels como alternativa a vetores
5. **Avaliar múltiplos eventos:** Se v1ik+v1iii conseguem separar ocorrências por período

### Longo Prazo
6. **Autorização de solicitação:** Se bloqueio temporal for crítico, revisar policy de e-mail/solicitação
7. **Publicação de análise:** Publicar que ground truth observado não está disponível em forma vetorial pública

---

## 13. Mensagens-Chave

✓ **v1ik foi conservador:** Não inferiu; rejeitou pistas fracas; documentou tudo.  
✓ **Bloqueio é honesto:** Sem data documentada, não é possível validar ocorrência.  
✓ **Melhor candidato é promissor:** Cicatriz_Area_A.shp é geometricamente sólido; falta apenas data.  
✓ **Próximos passos claros:** Se data for encontrada, reprocessamento é direto.  

---

**Data de Conclusão:** 2026-05-23  
**Versão:** v1ik-R1  
**Status:** `TEMPORAL_GATE_RECOVERY_AUDIT_COMPLETE_NO_RECOVERY`
