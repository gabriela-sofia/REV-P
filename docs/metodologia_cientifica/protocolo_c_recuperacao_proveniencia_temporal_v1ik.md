# Protocolo C — v1ik: Recuperação de Proveniência Temporal para Candidatos Vetoriais

## 1. Objetivo Científico

A etapa **v1ik** tenta resolver, de forma auditável e conservadora, o bloqueio temporal dos melhores candidatos vetoriais bloqueados em v1ij, especialmente aqueles com **geometria, CRS e fenômeno bem definidos** mas sem data documentada de evento ou levantamento.

Exemplo principal: `camada original de feições poligonais de deslizamento fotointerpretadas` (444 features de feição de deslizamento de deslizamento) bloqueada apenas por ausência de data de evento.

**v1ik NÃO CRIA DADO.** Apenas audita proveniência temporal em fontes já existentes (sidecars locais, registries, documentação versionada, metadados públicos).

## 2. Por Que v1ij Bloqueou os Melhores Candidatos

v1ij consolidou **18 candidatos** de v1if e v1ii. Resultado:

- **14 candidatos** bloqueados por `gate_02_no_geometry` — documentos (PDFs) ou datasets sem vetor real
- **3 candidatos** bloqueados por `gate_04_no_event_date` — vetores com geometria, mas sem data documentada
- **1 candidato** bloqueado por `gate_06_no_phenomenon` — vetor genérico sem fenômeno claro

O **melhor candidato**, `camada original de feições poligonais de deslizamento fotointerpretadas`, tem:
- ✓ Geometria: SHP com 444 features Polygon
- ✓ CRS: Presente e válido
- ✓ Fenômeno: feição de deslizamento de deslizamento (movimento de massa)
- ✓ Observed not risk: YES (é ocorrência observada, não mapa de risco)
- ✗ **Data de evento: NÃO TEM**

Sem data documentada, v1ij bloqueou conservadoramente. v1ik tenta resolver este bloqueio.

## 3. Por Que Data é Gate Crítico

Sem **data de evento** documentada, não é possível:
1. Vincular o vetor a um evento específico (ex: eventos de fevereiro de 2022 em Petrópolis)
2. Garantir que a geometria representa ocorrência no período-alvo
3. Descartar possibilidade de que as feições de deslizamento sejam de eventos anteriores ou posteriores
4. Usar o vetor como ground truth operacional

Exemplos de **datas não aceitáveis** por si só:
- Data de download do arquivo
- Data de modificação do arquivo (sistema de arquivos)
- Data de publicação em repositório
- "2022" em nome de pasta ou arquivo
- Data de criação de banco de dados

**v1ik NÃO INFERE.** Só aceita data se estiver documentada em:
- Campo explícito na tabela de atributos
- Metadado técnico publicado (.prj, .xml, .dbf)
- Documentação técnica oficial citando a data
- Relatório ou análise que vincule explicitamente à data

## 4. Diferenças Críticas de Tipo de Data

### Data de Evento (Aceitável como data principal)
- Data em que o evento ocorreu (deslizamento aconteceu, inundação ocorreu)
- Exemplo: "2022-02-15" para feição de deslizamento de deslizamento do dia 15 de fevereiro de 2022
- **v1ik procura esto.**

### Data de Levantamento ou Coleta (Aceitável como contexto)
- Data em que a feição de deslizamento foi mapeada ou avaliada
- Exemplo: "2022-02-28" para avaliação pós-desastre realizada em 28 de fevereiro
- **Não é data de evento,** mas ajuda a vincular: "feição de deslizamento foi mapeada 13 dias após o evento"
- Só útil se houver documentação do evento no período anterior

### Data de Publicação (NÃO aceitável como data de evento)
- Data em que o dado foi publicado em repositório
- Exemplo: "2024-03-10" para publicação em RIGeo em 2024
- **Não prova** data de evento; pode ser levantamento de anos antes

### Data de Modificação de Arquivo (SEMPRE INVÁLIDO)
- Timestamp do sistema de arquivos (created, modified)
- Pode ser cópia, re-upload, re-processamento
- **Nunca** é data de evento
- **v1ik rejeita explicitamente** qualquer pista que seja apenas file system date

### Pista em Nome de Pasta ou Arquivo (MUITO FRACA)
- "SIG_2022_Petropolis"
- "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA_2022"
- **Nunca** é STRONG; no máximo WEAK_FILE_OR_FOLDER_HINT
- Exige corroboração com documentação explícita

## 5. Fontes de Proveniência que v1ik Audita

### 1. Sidecars Locais (read-only)
- `.prj` — pode conter metadados de data de levantamento
- `.xml` — metadados estruturados
- `.shp.xml` — metadados específicos de shapefile
- `.dbf` — pode ter campos de data como atributos
- `.qmd` — metadados QGIS
- `.json` — metadados customizados
- `.csv` — dicionário de dados com datas

**Regra:** Só se o campo/valor for explícito e documentado.

### 2. Registries Versionados (v1if, v1ii)
- Consultar `official_observed_event_vector_registry.csv`
- Consultar `targeted_official_repository_event_vector_registry.csv`
- Procurar referências cruzadas que tragam data

**Regra:** Só se a referência for suficientemente específica.

### 3. Documentação Versionada
- README.md
- datasets/README.md
- docs/metodologia_cientifica/*.md
- Relatórios técnicos incorporados

**Regra:** Só se a referência for a uma fonte primária com data clara.

### 4. Metadados Públicos
- Descrição em catálogo de repositório (RIGeo, CKAN, etc.)
- Título/abstrato/keywords
- Instituição e data de publicação

**Regra:** Data de publicação NÃO é data de evento; só contexto.

## 6. Por Que Isso NÃO Cria Label

Mesmo se v1ik recuperar uma data documentada:

- **`can_create_training_label = false`** (sempre)
- O candidato pode avançar para `TEMPORAL_CONTEXT_STRENGTHENED_STILL_BLOCKED`
- **Não** vira label supervisionado automaticamente
- Próximo passo seria **patch binding assessment** (comparar com patches Sentinel reais)
- **Ainda depois,** overlay validation
- **Só então,** possível considerar label (se outras condições forem atendidas)

Label é produto de múltiplos gates, não apenas data. v1ik resolve **um gate** (temporal).

## 7. Conservadorismo em v1ik

v1ik é **defensivo e documentado:**

- ✓ Registra **cada** fonte consultada
- ✓ Registra **cada** evidência encontrada ou não
- ✓ Classifica força de evidência (`STRONG_*`, `MODERATE_*`, `WEAK_*`, `INVALID_*`)
- ✓ Rejeita pistas fracas explicitamente
- ✓ Mantém bloqueio se evidência insuficiente
- ✓ Explica **por que** cada bloqueio persiste

Se a data não pode ser documentada, `temporal_status_after_review = TEMPORAL_GATE_BLOCKED_NO_DATE` (sem eufemismo).

## 8. Matriz de Decisão Temporal

v1ik gera `temporal_gate_decision_matrix.csv` com campos:

- `has_explicit_event_date` — data do evento em metadado/campo explícito?
- `has_explicit_survey_date` — data de levantamento em metadado/campo explícito?
- `has_event_window` — janela de tempo documentada?
- `has_documentary_linkage` — vinculado a documento/publicação oficial?
- `has_registry_cross_reference` — encontrado em outro registry?
- `has_sidecar_metadata` — metadados em sidecar local?
- `only_file_or_folder_hint` — só pista em nome (fraco)?
- `only_file_system_date` — só date de arquivo (inválido)?
- `contradictory_temporal_evidence` — evidências conflitantes?

Resultado:
- `temporal_gate_status = BLOCKED | PARTIAL_CONTEXT | PASSED`
- `can_reopen_patch_binding_preflight = YES/NO`
- `can_reopen_ground_truth_candidate = YES/NO`
- `can_create_training_label = NO` (sempre)

## 9. Regras Absolutas de v1ik

```
nao_inventar_data               = true
nao_aceitar_pistas_fracas       = true
so_data_documentada             = true
nao_enviar_email                = true
nao_criar_label                 = true
nao_fazer_overlay               = true
nao_treinar                     = true
nao_versionar_dados_pesados     = true
```

## 10. Próximas Etapas Se Gate Temporal Fosse Recuperado

1. **Patch binding assessment** — comparar vetor com patches Sentinel-DINO
2. **Overlay validation** — verificar cobertura espacial e temporal
3. **Documento de decisão** — aceitar/rejeitar candidato para ML
4. **Nunca label direto** — sempre validação cruzada

---

**Data de v1ik:** 2026-05-23  
**Versão:** v1ik-R1  
**Status:** TEMPORAL_GATE_RECOVERY_AUDIT_COMPLETE
