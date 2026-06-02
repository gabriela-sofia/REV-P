# Relatório Científico: Referência Composta de Ground Truth (v1ip)

## Resumo Executivo

v1ip avaliou dossiês compostos de candidatos vetoriais fortes, verificando se vínculo rastreável entre vetor, documento, evento e fenômeno permite promover algum candidato para GROUND_REFERENCE_CANDIDATE.

**Resultado Principal:** camada original de feições poligonais de deslizamento fotointerpretadas é STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK

**Status:** SÍNTESE COMPLETA  
**Candidatos Avaliados:** 1 (camada de feições poligonais de deslizamento fotointerpretadas)  
**Ground Reference Candidates:** 0  
**Strong Composite But Weak Temporal:** 1  
**Bloqueador:** Vínculo explícito documento-vetor faltando  
**Evidência Mínima:** Link entre data de evento (2022-02-15) e camada de feições poligonais de deslizamento fotointerpretadas em documento oficial

---

## 1. Metodologia de v1ip

### Gates Compostos Avaliados

| Gate | Tipo | camada de feições poligonais de deslizamento fotointerpretadas |
|------|------|-----------------|
| **Geometry** | Binário | ✓ PASS |
| **CRS** | Binário | ✓ PASS |
| **Observed Status** | Binário | ✓ PASS (observed, não risk) |
| **Source Authority** | Binário | ✓ PASS (HIGH) |
| **Event Date Or Survey Date** | Binário | ✓ PASS (2022-02-15 documentado) |
| **Document-Vector Linkage** | Força | ✗ MODERATE (falta explícito) |
| **Region Match** | Força | ✓ STRONG (Petrópolis em ambos) |
| **Phenomenon Match** | Força | ✓ STRONG (MM em ambos) |

### Pontuação Geral

| Métrica | camada de feições poligonais de deslizamento fotointerpretadas |
|---------|-----------------|
| Gates Binários Passando | 5/5 |
| Gates de Força STRONG | 2/3 |
| Composite Evidence Strength | MODERATE |
| Overall Status | STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK |

---

## 2. Análise Detalhada: camada original de feições poligonais de deslizamento fotointerpretadas

### Componentes Verificados

#### 1. Vetor Observado
```
✓ Shapefile completo (.shp + .dbf + .shx)
✓ CRS: EPSG:31983 documentado
✓ Geometria: Feature-level (feições de deslizamento individuais)
✓ Fenômeno: Movimento de massa (feições de deslizamento de deslizamento)
✓ Observado: feições de deslizamento são ocorrências reais observáveis, não susceptibilidade
```

#### 2. Fonte Oficial
```
✓ Instituição: SGB/CPRM (HIGH authority)
✓ Documento: Anexo oficial pós-evento
✓ Ano: 2022-02-15 documentado em relatório SGB
✓ Confiabilidade: Fonte oficial governamental
```

#### 3. Evento Documentado
```
✓ Data: 2022-02-15
✓ Localidade: Petrópolis, RJ
✓ Fenômeno: Deslizamentos múltiplos confirmados
✓ Contexto: Evento de chuvas intensas → movimento de massa
```

#### 4. Vínculo Documento-Vetor

**O Que Falta:**
```
✗ Nenhuma fonte auditada (v1if até v1io) declara explicitamente:
  "camada original de feições poligonais de deslizamento fotointerpretadas foi criado em 2022-02-15"
  OU
  "camada original de feições poligonais de deslizamento fotointerpretadas é parte do pacote de resposta ao evento de 2022-02-15"

✗ Nenhuma metadata ou sidecar (.xml, .txt) linkando shapefile a data
✗ Nome de arquivo não menciona data de evento
```

**O Que Existe:**
```
✓ Data 2022-02-15 está documentada em PDF SGB (v1in extraiu)
✓ Fenômeno (movimento de massa) é mencionado em ambos
✓ Localidade (Petrópolis) aparece em ambos
✓ Fonte é única (SGB), proveniência é rastreável
→ Vínculo é MODERADO: forte por proveniência, fraco por explicitude
```

### Status de Vínculo Composto

| Aspecto | Força | Evidência |
|---------|-------|-----------|
| **Geometria-Fenômeno Match** | STRONG | feições de deslizamento ≡ movimento de massa |
| **Região Match** | STRONG | Petrópolis mencionado em ambos |
| **Fenômeno Match** | STRONG | Movimento de massa em documento e vetor |
| **Source Lineage** | STRONG | SGB é fonte única e oficial |
| **Data de Evento** | EXISTE | 2022-02-15 documentado |
| **Vínculo Explícito Data→Vetor** | FRACO | Falta linkage documental direto |
| **Temporal Link Strength** | MODERATE | Data existe, mas não linkada ao vetor |

**Resultado:** Evidência composta é forte (7/8 componentes), mas temporal linkage é fraco.

---

## 3. Por Que Não É GROUND_REFERENCE_CANDIDATE (Ainda)

### Critério Faltante

Para ser GROUND_REFERENCE_CANDIDATE, precisa de:
```
camada original de feições poligonais de deslizamento fotointerpretadas + documento oficial + "data 2022-02-15"
+ vínculo EXPLÍCITO entre eles
```

**Atualmente:**
```
✓ camada original de feições poligonais de deslizamento fotointerpretadas
✓ Documento oficial SGB
✓ Data 2022-02-15 documentada
✗ Vínculo explícito FALTANDO
```

### Por Que Essa Exigência Não É Excessiva

O vínculo explícito protege contra:
1. **Confusão de eventos:** "Qual evento gerou qual feição de deslizamento?"
2. **Datação incorreta:** "Essa feição de deslizamento é de 2022 ou 2015?"
3. **Múltiplas versões:** "Qual versão do shapefile é a de 2022?"

Sem vínculo explícito, não é possível validar contra Sentinel com confiança.

---

## 4. Decisão: STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK

### Classificação

**Status:** STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK

**Significado:**
- Evidência composta é forte em todos os aspectos EXCETO temporal linkage
- camada de feições poligonais de deslizamento fotointerpretadas é candidato sério para validação
- Falta detalhe específico: vínculo data-vetor

**Próximos Passos Possíveis:**
1. Acesso a PROJETO para metadata sidecars (.xml, .prj)
2. OCR manual em PDF SGB (pode haver menção a "camada de feições poligonais de deslizamento fotointerpretadas")
3. Contato com SGB para documentação de lineage
4. Nenhum desses está aqui — fora de escopo de v1ip

---

## 5. Invariantes Atendidos

- [x] Nenhum label foi criado
- [x] Nenhum modelo foi treinado
- [x] Protocolo B não foi reabertor
- [x] can_create_training_label = NO (sempre)
- [x] can_train_model = NO (sempre)
- [x] can_be_operational_ground_truth = NO (sempre)
- [x] can_reopen_protocol_b = NO (sempre)
- [x] Sem inventar vínculo
- [x] Sem aceitar contexto genérico
- [x] Sem usar data de arquivo/pasta
- [x] Sem aceitar risco como observado
- [x] Sem e-mail ou solicitação
- [x] Sem path privado em arquivos públicos
- [x] local_runs/ não versionado

---

## 6. Comparação: v1io vs v1ip

| Aspecto | v1io | v1ip |
|--------|------|------|
| **Abordagem** | Síntese de 7 etapas | Avaliação de vínculo composto |
| **Status camada de feições poligonais de deslizamento fotointerpretadas** | BLOCKED | STRONG_COMPOSITE_BUT_WEAK_TEMPORAL |
| **Razoamento** | Data não está no vetor | Data existe mas vínculo é fraco |
| **Implicação** | Impossível validar | Possível com vínculo explícito |
| **Próximo Passo** | v1in/manual OCR | Acesso a metadados/OCR |
| **Ground Truth** | Bloqueado permanentemente* | Bloqueado, mas destravável |

*v1io usava linguagem "bloqueado com evidência pública" (não permanentemente), v1ip a explora concretamente.

---

## 7. Conclusão Científica

### Contribuição de v1ip

v1ip demonstrou que:

1. **Bloqueio temporal tem raiz específica:** Não é "sem dados", é "sem linkage explícito entre data e geometria"

2. **Evidência composta é viável:** Combinar vetor + documento + evento + fenômeno produz candidato sério

3. **camada de feições poligonais de deslizamento fotointerpretadas é valoroso:** Mesmo sem passar em todos os gates, merece status diferenciado

4. **Próximos passos são claros:** Precisa-se especificamente de vínculo data→vetor, não busca genérica

### Valor para Tese

- Mostra que ground truth observacional é **resolvível**, não **impossível**
- Identifica exatamente qual falta (vínculo de linkage)
- Abre caminho para pesquisa futura com recursos adicionais

---

## 8. Status Final (Decisão de Commit)

**Commitável?** Sim, mas com ressalva

**Razão:**
- Registry de referência composta oferece novo insight (STRONG_COMPOSITE mas WEAK_TEMPORAL)
- Classificação v1ip é mais precisa que v1io para candidatos principais
- Documentação explica a lacuna com clareza operacional

**Se Commitar:**
- Incluir v1ip (scripts, tests, docs)
- Manter v1il/v1im/v1in local (ainda não recuperaram evidência nova)
- Atualizar v1io commit message para referenciar v1ip como refinamento

**Se Não Commitar Agora:**
- Aguardar OCR manual ou acesso a PROJETO
- Completar v1ip com linkage explícito encontrado
- Aí sim, commit de "camada de feições poligonais de deslizamento fotointerpretadas promovido para GROUND_REFERENCE_CANDIDATE"

---

**Data de Execução:** 2026-05-23  
**Etapa:** v1ip — Composite Ground Reference Evidence Builder  
**Status de camada de feições poligonais de deslizamento fotointerpretadas:** STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK  
**Conclusão:** Bloqueio temporal é específico, resolvível, não genérico  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
