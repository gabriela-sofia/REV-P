# Protocolo C — Varredura Profunda de Ativos Vetoriais Locais (v1il)

## Contexto e Justificativa

### Por que v1il existe

A etapa **v1ij** consolidou 18 candidatos para ocorrências observadas (non-risk) a partir de repositórios oficiais. Destes, **nenhum passou gates mínimos de validade**: geometria, CRS, fenômeno documentado, e data temporal documentada.

A etapa **v1ik** auditou a proveniência temporal de candidatos bloqueados. Especificamente para **camada original de pontos de feições de deslizamento fotointerpretadas**, esta não entrou na cadeia consolidada de v1ij porque **não foi encontrada em repositórios oficiais públicos** — o que gerou a questão: *existe essa geometria em sistemas locais ou em repositórios privativos?*

**v1il responde**: fazer uma varredura profunda local, auditável, em repositórios privados (PROJETO) e local-only se disponível, para recuperar ativos vetoriais ausentes da cadeia oficial. Sem invenção, sem cópia de dados pesados, sem criação de labels.

### Diferença entre "candidato ausente", "bloqueado" e "ground truth candidato"

| Termo | Significado | Quando ocorre | Exemplo |
|-------|-------------|---------------|---------|
| **Candidato ausente** | Ativo que não foi encontrado em nenhuma busca oficial | v1if/v1ii não o localizaram | camada original de pontos de feições de deslizamento fotointerpretadas |
| **Candidato bloqueado** | Ativo encontrado mas que falha em um ou mais gates | Geometria OK, mas sem data documentada | camada original de feições poligonais de deslizamento fotointerpretadas (sem data documental) |
| **Ground truth candidato** | Ativo que passou todos os gates e é candidato a referência de verdade no terreno | Nunca ocorreu até agora em PET | N/A (nenhum passou gates) |
| **Asset local recuperado** | Ativo encontrado em varredura local que pode ser considerado para handoff | v1il o localiza | camada original de pontos de feições de deslizamento fotointerpretadas se encontrado em PROJETO |

## Escopo de v1il

### O que v1il faz

- ✓ Varredura read-only em REV-P, PROJETO, local_only
- ✓ Busca por nomes/termos-alvo: "camada de pontos de feições de deslizamento fotointerpretadas", "camada de feições poligonais de deslizamento fotointerpretadas", "feição de deslizamento", "deslizamento", etc.
- ✓ Mapeamento de bundles shapefile (.shp + .dbf + .shx + opcionais)
- ✓ Auditoria de headers vetoriais quando seguro (geometria, CRS, campos)
- ✓ Identificação de campos de data e fenômeno em ativos encontrados
- ✓ Preparação de handoff para nova consolidação (v1ij/v1ik reexecutada)
- ✓ Geração de registries públicos (metadata-only)
- ✓ Documentação clara de limites (o que é observação, interpretação, não-claim)

### O que v1il NÃO faz

- ✗ Não baixa dados novos
- ✗ Não copia dados pesados para repo
- ✗ Não versiona shapefile, geopackage, raster, ZIP bruto ou sidecars
- ✗ Não inferencia data (usa apenas data documentada)
- ✗ Não aceita data de sistema de arquivos como data de evento
- ✗ Não aceita nome de pasta como prova forte
- ✗ Não classifica como "risco" ou "suscetibilidade" — só "ocorrência observada"
- ✗ Não cria label de treinamento
- ✗ Não cria ground truth automaticamente
- ✗ Não envia e-mail ou cria solicitação institucional

## Bundle Shapefile: Definição e Requisitos

### Bundle mínimo completo

Um shapefile é um formato de múltiplos arquivos. O bundle mínimo completo exige:

```
dataset/
├── camada original de pontos de feições de deslizamento fotointerpretadas      (geometria)
├── camada de pontos de feições de deslizamento fotointerpretadas.dbf      (atributos)
├── camada de pontos de feições de deslizamento fotointerpretadas.shx      (índice)
├── camada de pontos de feições de deslizamento fotointerpretadas.prj       (CRS, opcional mas recomendado)
└── camada de pontos de feições de deslizamento fotointerpretadas.cpg       (codepage, opcional)
```

**Requerimento: .shp + .dbf + .shx (mínimo)**

Se faltar qualquer um destes, o bundle é **incompleto** e **não pode ser processado**.

### CRS (Coordinate Reference System)

- Se arquivo `.prj` existe → CRS extraído do arquivo
- Se `.prj` não existe → CRS extraído do header do arquivo de geometria (quando possível)
- Se nenhum está disponível → **CRS_UNKNOWN** → **BLOCKED**

### Tratamento de metadata sidecars

Sidecars são arquivos de metadata associados:

- `.shp.xml` — metadata ISO 19139
- `.xml` — metadata genérica
- Documentação versionada em README, docs, etc.

Todos são consultados, mas **nenhum inventado**. Se não existir, o field respectivo fica vazio.

## Por que Asset Local Não Vira Ground Truth Automaticamente

Um asset local recuperado (mesmo que completo) **não pode ser imediatamente classificado como "ground truth"** pelos seguintes motivos:

1. **Falta validação cruzada**: Um ativo encontrado em PROJETO ou local_only pode estar:
   - Desatualizado (versão antiga vs versão oficial)
   - Não-validado (criado localmente, nunca publicado)
   - Parcialmente correto (alguns features OK, outros não)

2. **Falta confirmação de origem**: Não sabemos:
   - Quem criou / quando
   - Se é derivado de dado oficial ou original
   - Se reflete observação de campo ou derivação de modelo

3. **Falta gate temporal**: Mesmo que encontrado, ainda não passa:
   - Gate de data de evento documentada
   - Gate de fonte confiável
   - Gate de não-sobreposição com ground truth anterior

4. **Handoff como preparação, não conclusão**: v1il prepara o asset para re-entrada em v1ij/v1ik, que **reexecutarão os gates completos** com o ativo recuperado incluído.

## Registries Públicos Gerados por v1il

### 1. `datasets/local_vector_asset_recovery_registry.csv`

Inventário de **todos os ativos vetoriais encontrados** na varredura local.

**Campos principais:**
- `recovery_asset_id` — ID único do ativo
- `sanitized_asset_name` — Nome do arquivo (sem paths privados)
- `asset_family` — Família (shapefile_bundle, standalone_vector, zip_archive, etc.)
- `region_hint` — Hint de região (PET, REC, CTB, UNKNOWN)
- `event_hint` — Hint de evento (PET_2022_02_15, HYDROLOGICAL_EVENT, etc.)
- `geometry_likely_available` — Geometria provavelmente disponível (YES/NO/UNKNOWN)
- `crs_likely_available` — CRS provavelmente disponível
- `recovery_status` — FOUND_LOCAL ou NOT_FOUND
- `public_versioning_status` — Sempre "METADATA_ONLY" (nunca versiona arquivo bruto)
- `next_required_action` — Próxima ação técnica

**Invariante:** `public_versioning_status = "METADATA_ONLY"` sempre.

### 2. `datasets/missing_vector_candidate_recovery_registry.csv`

Registro específico de **candidatos-alvo que estavam missing** (camada de pontos de feições de deslizamento fotointerpretadas, camada de feições poligonais de deslizamento fotointerpretadas, etc.)

**Campos principais:**
- `missing_candidate_id` — ID do candidato missing
- `target_name` — Nome procurado (ex: "camada original de pontos de feições de deslizamento fotointerpretadas")
- `found_status` — FOUND / NOT_FOUND / FOUND_INCOMPLETE
- `found_as_asset_name` — Se encontrado, qual nome
- `candidate_recovery_decision` — RECOVER_IF_COMPLETE_BUNDLE / CANNOT_RECOVER
- `can_enter_next_consolidation` — Sempre "NO" inicialmente
- `can_create_training_label` — Sempre "NO"
- `blocking_reason` — Por que não pode entrar automaticamente

**Invariante:** `can_create_training_label = "NO"` sempre.

### 3. `datasets/recovered_candidate_consolidation_handoff.csv`

Handoff para **próxima consolidação** (v1ij/v1ik com assets recuperados re-incluídos).

**Campos principais:**
- `handoff_id` — ID do handoff
- `recovery_asset_id` — Referência ao ativo recuperado
- `target_candidate_name` — Nome sendo entregue
- `geometry_status` — LIKELY_AVAILABLE / NEEDS_VERIFICATION
- `date_status` — Sempre "NEEDS_VERIFICATION" (v1ik vai auditar)
- `recommended_next_stage` — "rerun_v1ik_with_recovered_asset"
- `should_rerun_v1ij` — SIM/NÃO
- `should_rerun_v1ik` — SIM/NÃO se encontrado
- `can_create_training_label` — Sempre "NO"

**Invariante:** `can_create_training_label = "NO"` sempre.

## Como o Handoff Prepara Nova Consolidação Sem Criar Label

1. **v1il recupera ativo** → "camada original de pontos de feições de deslizamento fotointerpretadas encontrado em PROJETO"
2. **v1il cria handoff** → "este ativo está pronto para re-entrada em v1ij/v1ik"
3. **v1ij reexecutada** → inclui camada de pontos de feições de deslizamento fotointerpretadas como candidato novo
4. **v1ij aplica gates** → verifica geometria, CRS, fenômeno (não cria label)
5. **v1ik reexecutada** → audita proveniência temporal do ativo recuperado
6. **v1ik decisão** → se passar todos os gates → candidato, não label ainda
7. **Label? Nunca.** → Ground truth requer Protocolo B + validação de campo

## Outputs de v1il

### Outputs Locais (não versionados)

```
local_runs/protocolo_c/v1il/
├── v1il_local_vector_asset_inventory.csv       — Todos os ativos encontrados
├── v1il_shapefile_bundle_audit.csv             — Bundles mapeados
├── v1il_missing_candidate_search_log.csv       — Log de busca de missing
├── v1il_vector_header_audit.csv                — Headers vetoriais auditados
├── v1il_summary.json                           — Resumo em JSON
└── [outros outputs opcionais]
```

**Nunca staged/committed.**

### Registries Públicos (versionados)

```
datasets/
├── local_vector_asset_recovery_registry.csv
├── missing_vector_candidate_recovery_registry.csv
├── recovered_candidate_consolidation_handoff.csv
└── schemas/
    ├── local_vector_asset_recovery_schema.csv
    ├── missing_vector_candidate_recovery_schema.csv
    └── recovered_candidate_consolidation_handoff_schema.csv
```

**Staged/committed com commit message clara.**

## Limites e Documentação de Não-Claims

### Observação vs Interpretação vs Claim

| Nível | Exemplo | Valido em v1il? |
|-------|---------|-----------------|
| **Observação** | "camada original de pontos de feições de deslizamento fotointerpretadas encontrado em /PROJETO/..." | ✓ SIM |
| **Interpretação** | "Provavelmente representa ocorrência de 2022-02-15 em Petrópolis" | ✓ Como hint, não claim |
| **Claim** | "camada original de pontos de feições de deslizamento fotointerpretadas é ground truth de inundação de Petrópolis" | ✗ NÃO (requer gates completos) |
| **Claim preditivo** | "Este ativo pode prever inundações futuras" | ✗ NÃO (nunca, sem validação) |

### Limitações Explícitas

v1il documenta:

- Que ativos locais podem estar desatualizados
- Que CRS pode não estar disponível
- Que data documentada pode não estar presente
- Que "asset local" ≠ "asset oficial validado"
- Que handoff é **preparação**, não **conclusão**

## Referência: Gates Completos (não todos passados até agora)

Para v1il recuperar um ativo e preparar handoff, checamos:

| Gate | O que testa | v1il resultado |
|------|----------|-------------------|
| **Geometry** | Formato válido, sem self-intersections | Verifica se .shp/.dbf/.shx existem |
| **CRS** | Sistema de referência documentado | Verifica .prj ou header |
| **Phenomenon** | Campo de fenômeno presente | Busca em DBF fields |
| **Observed vs Risk** | É ocorrência documentada, não susceptibilidade | Não pode inferir, apenas documentar |
| **Temporal** | Data de evento ou survey documentada | v1ik vai auditar depois |

v1il **prepara o terreno** mas **não passa ninguém automaticamente**.

## Próximos Passos Após v1il

1. **camada original de pontos de feições de deslizamento fotointerpretadas não foi recuperado:**
   - Documentado como NOT_FOUND em registries
   - Não há ativo local para reentrada em v1ij/v1ik
   - Status: bloqueado por ausência em repositórios públicos e locais auditados

2. **camada original de feições poligonais de deslizamento fotointerpretadas não foi recuperado em PROJETO:**
   - Mantém bloqueio anterior (falta data documentada)
   - Nenhuma versão local encontrada para enhanced evidence
   - Status: mantém bloqueio temporal de v1ik

3. **Próximo estágio:**
   - v1im: consolidação das fontes auditadas
   - v1in: extração de evidência temporal de documentos já consolidados
   - v1io: síntese final de prontidão

---

**Versão:** v1il — 2026-05-23  
**Autor:** Protocolo C — Deep Local Vector Asset Recovery and Bundle Audit  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão.**
