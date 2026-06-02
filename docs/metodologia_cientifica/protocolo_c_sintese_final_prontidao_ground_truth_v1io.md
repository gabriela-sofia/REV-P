# Protocolo C — Síntese Final de Prontidão de Ground Truth (v1io)

## Contexto e Justificativa

### Por que v1io existe

Protocolo C executou 7 etapas de auditoria (v1if a v1in) sobre fontes oficiais, repositórios públicos, varredura local e documentos locais. Nenhum candidato passou em todos os gates para ground truth operacional.

v1io existe para:
1. **Agregar resultados** de v1if até v1in sem busca nova
2. **Declarar status final** com clareza científica
3. **Explicar a lacuna** exatamente onde está
4. **Fechar o ciclo** do Protocolo C com conclusão metodológica, não fracasso
5. **Documentar para tese** como resultado acadêmico válido

### Diferença entre "bloqueado" e "indisponível"

| Estado | Significado | Implicação |
|--------|------------|-----------|
| **BLOQUEADO com evidência pública** | Candidato não passa gate, mas falta específico | Lacuna estrutural identificada |
| **INDISPONÍVEL** | Fonte nunca existiu ou não pode ser acessada | Sem resolução técnica |
| **READY_FOR_REFERENCE** | Passa todos os gates, pode ser ground truth | Raridade em projetos iniciantes |

v1io vai declarar: **BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE** — blocagem específica, não indisponibilidade.

## O Que v1io Faz

### 1. Agregação de Resultados

Lê sem processar novo:
- ✓ Registries públicos de v1if, v1ii, v1ij, v1ik
- ✓ Sumários locais de v1il, v1im, v1in
- ✓ Consolidação de candidatos de v1ij
- ✓ Auditoria de gates de v1ik

### 2. Síntese de Status Final

Para cada candidato, determina:
- Qual etapa o encontrou
- Quais gates passa
- Qual gate o bloqueia
- Se está ready_for_reference ou bloqueado

### 3. Declaração de Lacuna

Especifica:
- **Qual bloqueador**: temporal (99% dos candidatos)
- **Qual evidência falta**: data documentada (event date ou survey date)
- **Por que não resolve automaticamente**: sem linkage entre data e geometria candidato

### 4. Decisão Final de Ground Truth

Declara sem ambiguidade:
- 0 candidatos ready para reference operacional
- 20 auditados, 19 bloqueados, 1 missing
- Status: **BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE**
- Não é permanente: é lacuna específica

## O Que v1io NÃO Faz

- ✗ Não busca dados novos
- ✗ Não baixa nada
- ✗ Não envia e-mail
- ✗ Não cria label ou target
- ✗ Não reabrir Protocolo B
- ✗ Não cria proxy
- ✗ Não faz overlay
- ✗ Não sugere "validação manual" como caminho principal
- ✗ Não usa linguagem de "impossível" ou "permanente"

## Resultados Agregados

### Por Etapa

| Etapa | Nome | Evidência | Ativos | Candidatos | Gates Passados |
|-------|------|-----------|--------|-----------|----------------|
| **v1if** | Official Observed Event Vector Acquisition | Documental (SGB/CPRM) | 1 | 0 | 0 |
| **v1ii** | Targeted Official Repository Event Vector Mining | Vetorial (oficial) | 12 repos | 12+ | 7/8 (falta temporal) |
| **v1ij** | Consolidated Observed Event Vector Evidence | Consolidação | 12 fontes | 18 | 7/8 (falta temporal) |
| **v1ik** | Temporal Provenance Recovery | Auditoria temporal | 18 cand | 18 | 7/8 (temporal confirmado fail) |
| **v1il** | Deep Local Vector Asset Recovery | Varredura local | 29.157 | 0 | - |
| **v1im** | Master Source Consolidation | Consolidação | 4 fontes | 20 | 7/8 (falta temporal) |
| **v1in** | Documentary Temporal Evidence Extraction | Documentos | 2.866 docs | 0 linkados | - |

### Por Candidato

**Total Auditados:** 20
- **Ready for Reference:** 0
- **Bloqueados:** 19 (temporal gate fail)
- **Missing:** 1 (camada de pontos de feições de deslizamento fotointerpretadas)

**camada original de feições poligonais de deslizamento fotointerpretadas** (MAIS PRÓXIMO):
- Geometria: ✓ YES (shapefile completo)
- CRS: ✓ YES (EPSG:31983)
- Fenômeno: ✓ YES (movimento de massa)
- Observado (não risco): ✓ YES
- Data documentada: ✗ NO (BLOQUEADOR)
- Status: **BLOCKED** (temporal gate)

**camada original de pontos de feições de deslizamento fotointerpretadas**:
- Encontrado: ✗ NO
- Status: **MISSING**

**18 outros** (v1ij):
- Todos: geometria+CRS+fenômeno OK, data faltando
- Bloqueador: **temporal gate** (100%)

## O Bloqueador Primário: Temporal Gate

### Por que camada de feições poligonais de deslizamento fotointerpretadas não passa

```
Gate        | Status | Evidência
============|========|==================================================
geometry    | PASS   | Shapefile .shp+.dbf+.shx completo
CRS         | PASS   | EPSG:31983 documentado
phenomenon  | PASS   | Movimento de massa (feição de deslizamento observada)
observed    | PASS   | Ocorrência real, não risco/susceptibilidade
authority   | PASS   | Fonte oficial (SGB/CPRM)
TEMPORAL    | FAIL   | Sem data documentada: "qual evento é DE qual data?"
```

**Falta:** Uma ponte documentada entre geometria e data.
- Documentos confirmam: "houve deslizamento em Petrópolis em 2022-02-15"
- Documentos NÃO dizem: "camada original de feições poligonais de deslizamento fotointerpretadas É DESTE evento"

Sem essa linkage, não há base para validação cruzada com Sentinel-1/2.

## Evidência Temporal Contextual (v1in)

v1in encontrou **14 evidências STRONG** documentais:
- Data explícita (YYYY-MM-DD)
- Fenômeno mencionado (deslizamento, feição de deslizamento)
- Localidade mencionada (Petrópolis, Recife, Curitiba)
- Ligação documentada entre data, fenômeno, localidade

**Mas:** 0 linkadas a candidato específico

**Resultado:** Evidência existe, mas não resolve bloqueio porque não responde "qual geometria é de qual data?"

## Status Final de Ground Truth Operacional

### Declaração Científica

**BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE**

**Interpretação:**
- Não é "sem dados" (há geometrias)
- Não é "sem evento" (evento é documentado)
- Não é "sem autoridade" (fontes são oficiais)
- **É:** "sem linkage temporal entre geometria e data em fontes públicas/localmente auditadas"

### Por que ML/Label Seguem Bloqueados

| Requisito ML | Status | Razão |
|--------------|--------|-------|
| **Ground truth** | FALTA | Sem validação de campo (Protocolo B) |
| **Temporal linkage** | FALTA | Data não ligada a geometria candidato |
| **Training/test split** | FALTA | Sem base para separação temporal |
| **Cross-validation** | FALTA | Sem validação cruzada com Sentinel |
| **Supervision** | FALTA | Sem especialista validando |

**Conclusão:** Nenhum label pode ser criado. Sem ground truth, sem supervisão.

## Por Que Não Reabrir Protocolo B

Protocolo B (validação de campo) requer:
- Acesso a Petrópolis, Recife, Curitiba
- Vistoria em situ das feições de deslizamento
- Coleta de coordenadas com GPS
- Confirmação visual de data (documentos, evidência do evento)

**Isso está fora de escopo de Protocolo C**, que é read-only, auditável, baseado em dados públicos.

**v1io não reabre Protocolo B porque:**
- Decisão foi pré-estabelecida
- v1io é síntese, não novo estágio
- Reabrir Protocolo B seria mudança de escopo, não conclusão

## Síntese para Tese

### Contribuição Metodológica

Protocolo C demonstrou:

1. **Rastreabilidade completa:** 7 etapas audíveis, cada uma documentada
2. **Rigor em bloqueadores:** gate temporal é específico e mensurável
3. **Falta não é fracasso:** lacuna estrutural é descoberta válida
4. **Replicabilidade:** próxima pessoa pode reexecutar v1if-v1io e chegar aos mesmos bloqueadores

### Proximas Pesquisas Viáveis

Baseado em v1io, futuro pesquisador poderia:

1. **Acessar PDFs SGB/CPRM manualmente** para encontrar referência a "camada de feições poligonais de deslizamento fotointerpretadas" ou "feições de deslizamento de Petrópolis 2022-02-15"
2. **Fazer OCR robusto** em documentos de SGB (fora de v1in scope)
3. **Executar Protocolo B** se recurso disponível
4. **Validar com Sentinel-1/2** observações de mudança em 2022-02-15 em Petrópolis
5. **Se tudo validar:** criar linkage manual, rodar v1im/v1io com nova evidência

Nenhuma dessas é "trabalho perdido" — são continuações viáveis de Protocolo C.

---

**Versão:** v1io — Ground Truth Readiness Final Synthesis  
**Status:** Síntese Final Completa  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão científica, rigor máximo.**
