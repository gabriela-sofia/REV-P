# protocolo de revisão supervisora — REV-P v1hb

**Status**: Metodologia; Review-only; Interpretativa  
**Data**: 2026-05-18  
**Fase**: v1hb — Review Gate Execution Package

---

## 1. Propósito da revisão supervisora

A revisão supervisora é uma etapa de interpretação estrutural **exploratória** que permite validar
coerência visual entre:

- Estrutura de embeddings DINO (vizinhança, medoids, outliers);
- Padrões visuais observáveis em Sentinel RGB/NDVI;
- Cobertura GIS contextual (quando disponível);
- Variação regional na representação de paisagem.

**Objetivo**: Compreender a heterogeneidade estrutural documentada em v1gu e conectar
evidência matemática (embeddings) com interpretação visual e contextual.

**Não é**:
- Validação de modelo contra ground truth
- Classificação de risco ou vulnerabilidade
- Atribuição de label operacional
- Predição de enchente ou detecção de risco
- Medição de performance do DINO

---

## 2. O Que o Reviewer Pode Observar e Afirmar

✓ **Padrões visuais observáveis**
- Uso de solo (urbano, vegetação, água, solo exposto)
- Presença de corpos d'água, drenagem
- Características de infraestrutura (estradas, construções)
- Qualidade de dados (cobertura de nuvem, sombras, artefatos)

✓ **Coerência estrutural**
- DINO vizinhos mais próximos: visualmente similares ao patch?
- Heterogeneidade regional: patches de regiões diferentes parecem estruturalmente distintos?
- Padrões regionais consistentes (medoid é representativo; outlier é exceção)?

✓ **Consistência contextual**
- Cobertura GIS faz sentido visualmente?
- Indicadores ausentes são compensados por padrões visuais?
- Contexto geográfico (proximidade a água, topografia) alinha com estrutura visual?

✓ **Incerteza e limitações**
- Qualidade de dados limita o que pode ser observado?
- Reviewer tem confiança baixa/média/alta na interpretação?
- Corpus pequeno (12/128) implica que generalizações requerem cautela?

---

## 3. O Que o Reviewer NÃO Pode Afirmar

✗ **Nenhuma predição**
- "Este patch está em risco de enchente"
- "Será vulnerável a inundação"
- "Probabilidade de evento X"

✗ **Nenhuma detecção**
- "Este patch é uma zona de enchente"
- "Há evidência de risco aqui"
- "Detecção de hazard"

✗ **Nenhuma classificação**
- "Este patch é classe X" (risco baixo/médio/alto, zona de risco, etc)
- "Classificação de vulnerabilidade"
- "Categorização operacional"

✗ **Nenhuma validação supervisionada**
- "DINO predictions estão corretas" (não há predição)
- "Embeddings validam modelo"
- "Performance do DINO é Y%"

✗ **Nenhuma afirmação de ground truth**
- "Este é o padrão correto"
- "GIS é ground truth"
- "Este é o padrão ideal/esperado"

✗ **Nenhuma causalidade**
- "Isto causa enchente"
- "Isto determina risco"
- "Relação causal entre estrutura e evento"

✗ **Nenhuma generalização além da amostra**
- "Os 12 patches representam os 128"
- "Padrão é universal para a região"
- "Isso se aplica a todos os patches"

---

## 4. Categorias de Revisão

### 4.1 Medoid Regional

**Definição**: Patch mais central na distribuição de embeddings de sua região.

**Origem**: v1gu — embedding_regional_summary

**O que é**:
- Patch representativo estruturalmente
- Ponto de referência para coerência regional
- Centro da distribuição de similitude

**O que não é**:
- Patch "melhor", "ideal", ou "operacional"
- Rótulo de qualidade
- Classificação
- Validação

**Foco de revisão**:
1. Este patch é visualmente representativo para a região?
2. Que características típicas aparecem aqui?
3. Há consistência estrutural com outros patches regionais?
4. Data quality é adequada para interpretação?

### 4.2 Outlier Estrutural

**Definição**: Patch periférico na distribuição de embeddings regionais.

**Origem**: v1gu — embedding_regional_summary

**O que é**:
- Patch estruturalmente distinto da média regional
- Maior distância ao centroide
- Possível indicador de variação

**O que não é**:
- Patch "anômalo", "problemático", ou "errado"
- Falha de método ou erro
- Risco ou vulnerabilidade
- Qualidade inferior

**Foco de revisão**:
1. O que torna este patch estruturalmente diferente?
2. Há padrão visual correspondente à diferença estrutural?
3. Variação é explicável por contexto geográfico/uso de solo?
4. Outlier é válido ou artefato?

### 4.3 Cobertura GIS Baixa

**Definição**: Patch com poucos indicadores GIS disponíveis (AVAILABLE/PARTIAL).

**Origem**: v1gv — evidence_coverage_matrix

**O que é**:
- Patch com dados contextuais limitados
- Baixa cobertura em indicadores externos
- Necessidade de interpretação sem GIS

**O que não é**:
- Patch inútil ou de qualidade inferior
- Patch sem contexto real
- Patch não-representativo
- Falha de coleta

**Foco de revisão**:
1. Há padrões visuais mesmo sem GIS?
2. Contexto local (visual) compensa dados faltantes?
3. Ausência de GIS limita interpretação a que ponto?
4. Que evidência visual substitui indicadores ausentes?

### 4.4 Coherência Embedding-GIS (se presente)

**Definição**: Patch com sinais concordantes entre DINO e cobertura GIS.

**O que é**:
- Estrutura de embedding alinha com indicadores GIS
- Padrão visual consistente com contexto
- Multi-fonte de concordância

**O que não é**:
- Validação de DINO
- GIS como ground truth
- Confirmação de modelo

**Foco de revisão**:
1. Indicadores GIS fazem sentido visualmente?
2. Estrutura DINO reflete padrões visíveis?
3. Que aspectos concordam entre fontes?

### 4.5 Conflito Embedding-GIS (se presente)

**Definição**: Patch com sinais divergentes entre DINO e cobertura GIS.

**O que é**:
- Estrutura de embedding diverge de indicadores GIS
- Possível artefato de dados
- Heterogeneidade real
- Oportunidade de compreender discordância

**O que não é**:
- Erro de método DINO
- Falha de GIS
- Problema a resolver
- Indicador de invalidade

**Foco de revisão**:
1. Por que divergem?
2. GIS incompleto explica a divergência?
3. Heterogeneidade real (visual) existe?
4. Que fonte é mais confiável aqui?

---

## 5. Protocolo de Anotação

### 5.1 Preparação

1. Abrir manifest de execução: `review_gate_execution_manifest_v1hb.csv`
2. Para cada item (HRE001–HRE047):
   - Identificar canonical_patch_id
   - Localizar Sentinel RGB + NDVI correspondentes
   - Revisar embedding evidence (v1gu summary)
   - Revisar GIS coverage (v1gv matrix)

### 5.2 Inspeção Visual

1. **RGB + NDVI**:
   - Observar cor e textura (RGB)
   - Índice de vegetação (NDVI)
   - Anomalias ou padrões

2. **Interpretação de uso de solo**:
   - Urbano: estruturas, ruas, densidade
   - Vegetação: cobertura, tipo (densa/esparsa)
   - Água: corpos d'água, drenagem
   - Solo exposto: claridade, padrão

3. **Qualidade de dados**:
   - Cobertura de nuvem (% aproximado)
   - Sombras
   - Artefatos de aquisição (striping, distorção)

### 5.3 Contexto Geográfico

1. **Proximidade hidrográfica**:
   - Distância visual a rios/lagoas
   - Padrão de drenagem (se visível)
   - Presença de canais artificiais

2. **Topografia** (se observável):
   - Encostas (relevo)
   - Vales ou depressões
   - Altitude relativa

3. **Contexto regional**:
   - Posição na região (margem, centro, etc)
   - Padrão predominante (urbano/rural)
   - Proximidade a boundaries administrativas

### 5.4 Sinais Estruturais

1. **DINO vizinhos**:
   - Revisar top-5 vizinhos mais próximos (se v1gu disponível)
   - Visualmente similares?
   - Mesma categoria de uso de solo?
   - Mesma região? Inter-região?

2. **Coherência GIS**:
   - Indicadores presentes fazem sentido visual?
   - Indicadores ausentes: por quê? (limite de coleta? realidade?)

3. **Heterogeneidade regional**:
   - Patch é típico ou atípico para a região?
   - Como se compara com medoid/outlier já revistos?

### 5.5 Campos de Anotação

Usar template: `review_gate_annotation_template_v1hb.csv`

- **reviewer_name_or_initials**: Identificação (REV-01, etc)
- **review_date**: YYYY-MM-DD
- **visual_pattern_notes**: O que reviewer observa (cor, textura, padrão)
- **surrounding_context_notes**: Contexto geográfico, topografia, proximidade
- **external_evidence_notes**: Consistência/discordância com GIS
- **uncertainty_level**: low | medium | high (confiança do reviewer)
- **usable_in_discussion**: yes | no | conditional
- **discussion_note**: Como este patch informa a Discussão do TCC?
- **no_label_created_confirmed**: Confirmação: reviewer não criou label
- **no_prediction_claim_confirmed**: Confirmação: reviewer não fez claim preditivo

---

## 6. Como Usar revisão supervisora na Discussão do TCC

### 6.1 Insumos Diretos

Tabela: `review_gate_discussion_inputs_v1hb.csv`

Cada linha oferece:
- **finding_id**: Identificador para referência
- **evidence_type**: Tipo de evidência (medoid regional, outlier, etc)
- **finding_summary**: O que foi observado (ex: "3 patches em medoid category")
- **supporting_artifact**: Referência (ex: "v1gu summary + v1gy figures")
- **interpretation_for_discussion**: Como conectar ao TCC (ex: "estrutura revela padrões X")
- **limitation**: O que limita a generalização
- **claim_allowed**: yes (pode ser afirmado)
- **claim_blocked**: Qual tipo de claim é bloqueado

### 6.2 Estrutura de Discussão

1. **Seção 5.1: Análise Estrutural Revisada**
   - Resumo de observações por categoria
   - Concordância entre DINO e padrões visuais
   - Heterogeneidade regional documentada
   - Representatividade do corpus (12/128)

2. **Seção 5.2: Sinergias Embedding-Contextual**
   - Onde DINO e GIS concordam
   - Onde divergem (e por quê)
   - Interpretação de covariância

3. **Seção 5.3: Limitações e Escopo**
   - Corpus reduzido não generaliza
   - Qualidade de dados (GIS parcial)
   - Incertezas documentadas
   - Exploratório, não preditivo

4. **Seção 5.4: Implicações para Pesquisa Futura**
   - Que perguntas revisão supervisora levanta?
   - Que dados faltam?
   - Como expandir corpus validando estrutura?

---

## 7. Garantias Metodológicas

### 7.1 Revisão Permanece Interpretativa

- Não é validação (sim-não, certo-errado)
- É compreensão (como, por quê, que padrões)
- Não é operacional (sem decisão ou ação)

### 7.2 Nenhum Label ou Classe é Criado

- Reviewer observa padrões, não rotula
- Observação é descritiva ("vejo água"), não classificatória ("isso é risco")
- Template explicitamente pede confirmação: "no_label_created_confirmed"

### 7.3 Nenhuma Ground Truth é Estabelecida

- Reviewer não confirma ou valida
- Observação é um dado adicional, não verdade canônica
- GIS permanece contextual, não validante

### 7.4 Corpus Permanece Pequeno

- 12 embeddings, 47 candidatos revistos (máximo)
- Amostra inicial, exploratória
- Não estima população de 128
- Reviewer confirma escopo: "usable_in_discussion: conditional"

### 7.5 Toda Incerteza é Documentada

- Campo "uncertainty_level": low | medium | high
- Campo "limitation": O que impede interpretação?
- Discussão reflete cautela

---

## 8. Checklist para Reviewer

Antes de submeter anotações:

- [ ] Todos os 47 itens foram revistos?
- [ ] Cada item tem reviewer_name_or_initials e review_date?
- [ ] Nenhum item tem claims preditivas/classificatórias?
- [ ] Campo "no_label_created_confirmed": todos "yes"?
- [ ] Campo "no_prediction_claim_confirmed": todos "yes"?
- [ ] Incerteza (low/medium/high) é plausível?
- [ ] Usable_in_discussion (yes/no/conditional) é justificado?
- [ ] Discussion_note conecta patch específico a argumentos de Discussão?

---

## 9. Próximos Passos Pós-Revisão

1. **Consolidar anotações**
   - Mesclar múltiplos reviewers (se houver)
   - Resolver discordâncias por consenso
   - Documentar discordâncias (se persistirem)

2. **Extrair findings**
   - Quais padrões emergiram?
   - Qual consistência intra-regional?
   - Qual variação inter-regional?
   - Qual nível de incerteza?

3. **Escrever Discussão**
   - Usar template de "discussion_inputs"
   - Referenciar artifacts (v1gu, v1gy)
   - Manter escopo (exploratório, revisado)
   - Documentar limitações

4. **Finalizar TCC**
   - Seção 5 com discussão baseada em revisão
   - Conclusões conservadoras
   - Trabalho futuro claro

---

**Versão**: v1hb  
**Última atualização**: 2026-05-18  
**Status**: Review-only; Metodologia; Audit-ready
