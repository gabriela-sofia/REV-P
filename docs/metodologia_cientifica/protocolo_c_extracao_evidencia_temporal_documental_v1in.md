# Protocolo C — Extração de Evidência Temporal de Documentos Locais (v1in)

## Contexto e Justificativa

### Por que v1in existe

Após consolidação em v1im, todos os 20 candidatos vetoriais estão bloqueados pelo **gate temporal**: nenhum tem data de evento ou levantamento documentada em fontes públicas/oficiais auditadas.

Mas v1im mostrou claramente:
- Geometria vetorial existe e é válida (21 candidatos com `.shp` completo)
- Fenômeno é claro e separável (movimento de massa documentado)
- Autoridade de fonte é HIGH (SGB/CPRM, repositórios oficiais)
- **Única lacuna**: data documentada que ligue geometria a evento/levantamento

v1in busca recuperar essa evidência temporal a partir de **documentos já localmente acessíveis**:
- PDFs de SGB/CPRM que confirmam evento de 2022-02-15 em Petrópolis
- Registries de v1ij/v1ik que podem ter data embarcada
- Relatórios técnicos, metadata sidecars, documentação versionada
- **Sem** solicitação externa, **sem** OCR pesado, **sem** validação manual como conclusão

### O que v1in testa

**Hipótese**: "A evidência temporal documental suficiente existe em fontes locais, mas não foi estruturada em v1im"

**Teste**: Varredura auditável de documentos locais para extrair expressões temporais e linkade com candidatos bloqueados

**Resultado esperado**: Se encontrada evidência STRONG (data explícita + vínculo com evento/fenômeno/localidade/candidato), candidato pode passar temporal gate

## Definições Precisas

### O que é "evidência temporal"

**Evidência temporal = expressão temporal + vínculo explícito**

| Elemento | Definição | Exemplo |
|----------|-----------|---------|
| **Expressão temporal** | Data, período ou janela temporal extraída de texto | "2022-02-15", "fevereiro de 2022" |
| **Vínculo explícito** | Contexto documental que liga expressão a evento, fenômeno, localidade ou candidato | "deslizamento em Petrópolis em 2022-02-15" |
| **Vínculo implícito** | Contexto fraco: apenas fenômeno OU apenas localidade OU apenas ano | "ano 2022" sem evento |
| **Sem vínculo** | Data isolada de contexto | "2022" em cabeçalho de arquivo |

**Critério crítico: Data isolada NÃO é evidência forte. Precisa vínculo documentado.**

### Classificação de Força de Evidência

| Força | Definição | Quando ocorre | Atualiza gate temporal? |
|-------|-----------|---------------|------------------------|
| **STRONG_EXPLICIT_EVENT_DATE** | Data completa (YYYY-MM-DD) + fenômeno documentado + localidade | "deslizamento em Petrópolis em 2022-02-15" | **SIM** |
| **STRONG_EXPLICIT_SURVEY_DATE** | Data de levantamento/vistoria explícita + fenômeno + localidade | "vistoria do evento em 2022-02-18" | **SIM** |
| **MODERATE_EVENT_WINDOW** | Mês/ano + contexto de evento documentado | "fevereiro de 2022, deslizamentos no estado" | **NÃO** (janela, não exata) |
| **MODERATE_DOCUMENTARY_CONTEXT** | Ano + contexto claro de evento/resposta | "resposta aos deslizamentos de 2022" | **NÃO** |
| **WEAK_TEXTUAL_HINT** | Ano mencionado vagamente OU referência indireta | "2022 em nome de pasta" ou "durante ano de 2022" | **NÃO** |
| **INSUFFICIENT** | Data ausente ou sem vínculo mensurável | "arquivo armazenado em 2025" | **NÃO** |
| **CONTRADICTORY** | Múltiplas datas conflitantes para mesmo evento | "2022-02-15 vs 2022-03-01 para mesmo evento" | **NÃO** |

## O Que v1in Faz

### 1. Varredura de Documentos

- ✓ Scan de `local_runs/`, `local_only/`, PROJETO (se acessível)
- ✓ Scan de PDFs em `local_runs/protocolo_c/v1if/` (SGB/CPRM)
- ✓ Audit de registries públicas existentes (v1ij, v1ik, v1im)
- ✓ Extração de tipos: PDF, CSV, TXT, JSON, MD, XML
- ✓ Institucionalização: SGB/CPRM, PROTOCOLO_C, local, PROJETO

### 2. Extração de Expressões Temporais

- ✓ Regex-based pattern matching (sem OCR pesado)
- ✓ Padrões: YYYY-MM-DD, DD/MM/YYYY, mês de YYYY, YYYY isolado
- ✓ Contexto: 200 chars circundantes (antes/depois de expressão)
- ✓ Fenômeno detectado: deslizamento, inundação, feição de deslizamento, NONE
- ✓ Localidade detectada: PET, REC, CTB, NONE
- ✓ Candidato mencionado: nome exato ou NONE

### 3. Classificação de Vínculo

- ✓ Explícito: fenômeno + localidade + data juntos no contexto
- ✓ Implícito: fenômeno OU localidade, sem junção clara
- ✓ Nenhum: data isolada ou sem contexto relevante

### 4. Decisão de Força de Evidência

- ✓ Aplicar matriz rigorosa (tipo de data + tipo de vínculo)
- ✓ Determinar: can_update_temporal_gate (YES/NO)
- ✓ Determinar: can_update_ground_truth_readiness (YES/NO)
- ✓ Todas: can_create_training_label = NO

### 5. Linkage a Candidatos

- ✓ Match automático: menção de nome de candidato
- ✓ Match por fenômeno + localidade: correlação a candidatos conhecidos de v1ij
- ✓ Documentar força de linkage: YES, IMPLICIT, NONE

## O Que v1in NÃO Faz

- ✗ Não faz OCR pesado (só regex em arquivos de texto)
- ✗ Não solicita dados a instituições
- ✗ Não valida geometria (já feito em v1ij)
- ✗ Não cria label (sempre can_create_training_label = NO)
- ✗ Não faz overlay ou análise espacial
- ✗ Não reabrir Protocolo B (sem validação de campo)
- ✗ Não inventar data (só extrai do texto)
- ✗ Não aceitar data de arquivo (mtime) como data de evento

## Outputs de v1in

### Outputs Locais (não versionados)

```
local_runs/protocolo_c/v1in/
├── v1in_document_inventory.csv       — Todos documentos encontrados
├── v1in_temporal_expression_candidates.csv  — Expressões temporais extraídas
├── v1in_evidence_strength_decision.csv      — Decisões de força
├── v1in_summary.json                 — Resumo com stats
└── v1in_qa.csv                       — QA de validações (se gerado)
```

**Nunca versionados**: todos são outputs intermediários de auditoria local.

### Registries Públicos (versionados APENAS se evidência útil)

```
datasets/
├── documentary_temporal_evidence_registry.csv         (se STRONG encontrada)
├── schemas/documentary_temporal_evidence_schema.csv
├── documentary_temporal_candidate_linkage_matrix.csv  (se linkage encontrado)
└── schemas/documentary_temporal_candidate_linkage_schema.csv
```

**Invariante**: Registry público criado **somente** se houver evidência:
- STRONG_EXPLICIT_EVENT_DATE OU
- STRONG_EXPLICIT_SURVEY_DATE

Senão, v1in fica local e registra "nenhuma evidência forte encontrada".

## Campos do Registry Público

**documentary_temporal_evidence_registry.csv**:
- `evidence_id` — ID único
- `source_document_id` — referência ao documento
- `source_document_name_sanitized` — nome sem paths privados
- `source_institution` — SGB, CPRM, PROTOCOLO_C, etc
- `region` — PET, REC, CTB
- `event_id` — ex: PET_2022_02_15
- `candidate_asset_name` — candidato linkado (ou UNKNOWN)
- `evidence_text_excerpt_sanitized` — contexto (max 200 chars)
- `temporal_expression` — data/período extraído
- `temporal_expression_type` — DATE, MONTH_YEAR, YEAR, WINDOW, CONTEXT
- `temporal_precision_level` — EXACT (data) OU WINDOW (período)
- `phenomenon_mentioned` — tipo de fenômeno ou NONE
- `location_mentioned` — região ou NONE
- `candidate_linkage_type` — YES, IMPLICIT, NONE
- `evidence_strength` — classificação
- `accepted_as_event_date` — YES/NO/PARTIAL
- `accepted_as_survey_date` — YES/NO
- `accepted_as_context_only` — YES/NO
- `can_update_temporal_gate` — YES/NO
- `can_update_ground_truth_readiness` — YES/NO
- `can_create_training_label` — sempre NO
- `blocking_reason` — por que não atualiza gate se aplicável
- `notes` — contexto adicional

## Próximas Etapas Baseadas em Resultados

### Cenário A: Evidência STRONG Encontrada (ex: camada de feições poligonais de deslizamento fotointerpretadas)

1. **Registry público criado**: documentary_temporal_evidence_registry.csv
2. **Candidato promovido**: camada de feições poligonais de deslizamento fotointerpretadas passa temporal gate
3. **Re-execução de v1im**: com nova evidência temporal
4. **Resultado**: candidato mais próximo de READY_FOR_REFERENCE
5. **Decisão**: commit v1in + v1im + v1io (síntese final)

### Cenário B: Nenhuma Evidência STRONG (provável)

1. **Nenhum registry público**: v1in fica local em local_runs/
2. **Conclusão**: evidência temporal não está disponível em fontes localmente acessíveis
3. **Status mantém**: BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE
4. **Próximo passo**: v1io (síntese final com conclusão da lacuna)
5. **Decisão**: v1in não commitável, prosseguir para v1io

## Invariantes de v1in

```
nao_inventar_data                       = true
nao_usar_mtime_como_data                = true
nao_fazer_ocr_pesado                    = true
nao_solicitar_dados_externos            = true
data_isolada_nao_eh_forte               = true
requer_vinculo_explicito_para_strong    = true
pode_criar_label                        = false
pode_reabrir_protocolo_b                = false
pode_fazer_overlay                      = false
sem_validacao_manual_como_conclusao     = true
pdf_requer_ocr                          = true
paths_privados_sanitizados              = true
```

---

**Versão:** v1in — Documentary Temporal Evidence Extraction  
**Status:** Auditável, determinístico, read-only  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão científica.**
