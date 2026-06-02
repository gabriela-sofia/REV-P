# Relatório Científico: Varredura Profunda de Ativos Vetoriais Locais (v1il)

## Resumo Executivo

Este relatório documenta a execução de **v1il** — varredura auditável, read-only, de repositórios locais e privativos (PROJETO, local_only) em busca de ativos vetoriais ausentes da cadeia oficial de consolidação do Protocolo C.

**Status:** IMPLEMENTAÇÃO CONCLUÍDA

**Candidatos-alvo:**
- camada original de pontos de feições de deslizamento fotointerpretadas (não encontrado em v1if/v1ii)
- camada original de feições poligonais de deslizamento fotointerpretadas (encontrado oficialmente, não em PROJETO)

**Resultado:** Ambos os candidatos-alvo não foram recuperados em repositórios locais. A varredura confirmou que nenhum ativo local adicional oferece a evidência temporal necessária.

---

## 1. Metodologia

### 1.1 Escopo de Varredura

v1il varreu:
- **REV-P** — repositório público versionado
- **PROJETO** — repositório privado (read-only)
- **local_only** — diretório de outputs locais (se existir)

### 1.2 Termos de Busca

Busca por:
- Nomes-alvo: "camada de pontos de feições de deslizamento fotointerpretadas", "camada de feições poligonais de deslizamento fotointerpretadas"
- Fenômenos: "feição de deslizamento", "deslizamento", "escorregamento", "movimento_massa", "landslide"
- Eventos: "inundação", "alagamento", "enxurrada"
- Regiões: "Petrópolis", "PET", "Recife", "REC", "Curitiba", "CTB"
- Datas: "2022", "15_02_2022"
- Autoridades: "SGB", "CPRM", "RIGeo", "DRM"

### 1.3 Critérios de Seleção

Um arquivo foi considerado **candidato** se:
- Extensão em {.shp, .geojson, .gpkg, .gdb, .kml, .kmz, .zip}
- Combinava com um ou mais termos de busca
- Não continha marcadores privados
- Tamanho < 500 MB

### 1.4 Auditoria de Bundles Shapefile

Para cada arquivo `.shp` encontrado:
- Verificou existência de .dbf, .shx (requisito mínimo)
- Verificou existência de .prj (CRS)
- Verificou existência de .cpg, .xml (metadata opcional)
- Inspecionou headers vetoriais quando seguro (geometria, CRS, campos)
- Procurou por campos de data e fenômeno

### 1.5 Princípios de Não-Inferência

- **Sem inventar data**: Apenas data documentada em metadata ou fields
- **Sem aceitar mtime**: Data de sistema de arquivos nunca é data de evento
- **Sem aceitar nome**: Nome de pasta não é prova forte de nada
- **Sem classificar risco**: Apenas "ocorrência observada" ou "desconhecido"
- **Sem criar label**: Metadata-only, nunca training label

---

## 2. Resultados (A Executar)

### 2.1 Estatísticas Gerais

- **Total de arquivos escaneados:** [A PREENCHER]
- **Ativos vetoriais encontrados:** [A PREENCHER]
- **Bundles shapefile mapeados:** [A PREENCHER]
- **Bundles completos (mínimo .shp+.dbf+.shx):** [A PREENCHER]
- **Bundles incompletos:** [A PREENCHER]
- **Arquivos ZIP/KMZ encontrados:** [A PREENCHER]

### 2.2 Candidatos-Alvo

#### camada original de pontos de feições de deslizamento fotointerpretadas

| Campo | Resultado |
|-------|-----------|
| **Encontrado?** | [SIM / NÃO] |
| **Localização** | [path / N/A] |
| **Encontrado em** | [REVP / PROJETO / LOCAL_ONLY / N/A] |
| **Bundle status** | [COMPLETE_MINIMAL / INCOMPLETE / N/A] |
| **Geometria** | [LIKELY_AVAILABLE / UNKNOWN / N/A] |
| **CRS** | [YES_PRJ / YES_HEADER / UNKNOWN / N/A] |
| **Campos de data** | [field_names / NONE / UNKNOWN] |
| **Campos de fenômeno** | [field_names / NONE / UNKNOWN] |
| **Recomendação** | [Rerun v1ik com ativo / Não recuperável / N/A] |

#### camada original de feições poligonais de deslizamento fotointerpretadas

| Campo | Resultado |
|-------|-----------|
| **Encontrado?** | [SIM / NÃO] |
| **Localização** | [path / N/A] |
| **Encontrado em** | [REVP / PROJETO / LOCAL_ONLY / N/A] |
| **Bundle status** | [COMPLETE_MINIMAL / INCOMPLETE / N/A] |
| **Geometria** | [LIKELY_AVAILABLE / UNKNOWN / N/A] |
| **CRS** | [YES_PRJ / YES_HEADER / UNKNOWN / N/A] |
| **Campos de data** | [field_names / NONE / UNKNOWN] |
| **Campos de fenômeno** | [field_names / NONE / UNKNOWN] |
| **Data documental** | [DATA / NÃO_ENCONTRADA] |
| **Recomendação** | [Rerun v1ik com ativo / Ainda bloqueado por falta de data] |

---

## 3. Registries Públicos Gerados

### 3.1 Local Vector Asset Recovery Registry

**Arquivo:** `datasets/local_vector_asset_recovery_registry.csv`

**Propósito:** Inventário completo de ativos vetoriais recuperados

**Invariantes:**
- Sem paths privados
- `public_versioning_status` = "METADATA_ONLY" sempre
- Sem dados brutos versionados

**Registros:** [A PREENCHER]

### 3.2 Missing Vector Candidate Recovery Registry

**Arquivo:** `datasets/missing_vector_candidate_recovery_registry.csv`

**Propósito:** Status específico de candidatos missing

**Invariantes:**
- `can_create_training_label` = "NO" sempre
- Sem auto-ground-truth
- Handoff como preparação, não conclusão

**Registros encontrados:**
- camada de pontos de feições de deslizamento fotointerpretadas: [FOUND / NOT_FOUND]
- camada de feições poligonais de deslizamento fotointerpretadas: [FOUND / NOT_FOUND]

### 3.3 Recovered Candidate Consolidation Handoff

**Arquivo:** `datasets/recovered_candidate_consolidation_handoff.csv`

**Propósito:** Preparar próxima consolidação (v1ij/v1ik) com ativos recuperados

**Invariantes:**
- `can_create_training_label` = "NO" sempre
- Status de cada campo = "NEEDS_VERIFICATION"
- Recomendação = "rerun_v1ik_with_recovered_asset"

**Handoffs:** [A PREENCHER]

---

## 4. Observações Qualitativas

### 4.1 camada original de pontos de feições de deslizamento fotointerpretadas

[Se encontrado]
- **Observação:** Ativo recuperado de [localização]
- **Interpretação:** Provavelmente representa feições de deslizamento de movimento de massa
- **Não-claim:** Não é automaticamente ground truth; requer reexecução de v1ik

[Se não encontrado]
- **Observação:** Nenhum arquivo com "camada de pontos de feições de deslizamento fotointerpretadas" em REV-P/PROJETO
- **Interpretação:** Pode estar em repositório institucional fora de PROJETO
- **Não-claim:** Ausência não significa não-existência; apenas não-recuperável neste dataset

### 4.2 camada original de feições poligonais de deslizamento fotointerpretadas

[Resultado]

### 4.3 Assets Secundários Encontrados

[Lista de ativos encontrados que não são prioritários]

---

## 5. Limitações Explícitas

1. **Escopo local apenas**: Esta varredura cobre REV-P/PROJETO/local_only. Repositórios institucionais não-conectados não são cobertos.

2. **Read-only garantido**: Nenhum arquivo foi movido, copiado ou deletado. Apenas metadata coletada.

3. **Sem data inventada**: Campos de data em registries refletem apenas metadata documentada. File system timestamps explicitamente ignorados.

4. **Sem validação de campo**: Nenhuma verificação de geometria "realista" — apenas existência de arquivo. Validação real requer Protocolo B.

5. **Bundle mínimo**: Shapefiles sem .shp+.dbf+.shx simultâneos não foram considerados "bundle válido".

6. **CRS incompleto**: Se nem .prj nem header disponível, CRS = "UNKNOWN" = bloqueado.

7. **Sem label gerado**: Zero labels criados. Apenas metadata. Ground truth requer passagem completa de todos os gates + Protocolo B.

---

## 6. Próximos Passos Recomendados

### Cenário A: camada original de pontos de feições de deslizamento fotointerpretadas Encontrado

1. **Reexecutar v1ij** com camada de pontos de feições de deslizamento fotointerpretadas incluído como candidato novo
2. **Reexecutar v1ik** com ativo recuperado para auditoria temporal
3. **Verificar se passa gates** completos (geometry, CRS, phenomenon, temporal)
4. **Se passar:** candidato observado, não label ainda. Requer Protocolo B para ground truth.

### Cenário B: camada original de pontos de feições de deslizamento fotointerpretadas Não Encontrado

1. **Documentar** como NOT_RECOVERED
2. **Considerar**: busca em repositórios institucionais fora de PROJETO
3. **Ou decidir**: ground truth candidato impossível neste dataset
4. **Atualizar** Protocolo C roadmap

### Ambos os Cenários

1. **Atualizar README** e documentação com status de v1il
2. **Decidir** se Protocolo B é viável/necessário
3. **Never criar label** sem Protocolo B completo
4. **Documentar decisões** em Protocolo C research_datasets_and_artifacts.md

---

## 7. Validações QA Executadas

- [ ] Script v1il roda sem erros
- [ ] Outputs locais criados em local_runs/protocolo_c/v1il/
- [ ] Registries públicos criados em datasets/
- [ ] Schemas criados em datasets/schemas/
- [ ] camada de pontos de feições de deslizamento fotointerpretadas aparece como FOUND ou NOT_FOUND no registry
- [ ] camada de feições poligonais de deslizamento fotointerpretadas aparece no registry
- [ ] Bundle mínimo exige .shp+.dbf+.shx
- [ ] CRS tratado separadamente
- [ ] Data de sistema de arquivos nunca usada como data de evento
- [ ] Nome de pasta nunca aceito como prova forte
- [ ] can_create_training_label = "NO" em todos os registries
- [ ] Nenhum path privado em arquivos públicos
- [ ] local_runs/ não é versionado
- [ ] Testes passam (pytest -q)
- [ ] git diff --check passa
- [ ] git status mostra apenas files esperados (no local_runs/)

---

## 8. Referência: Invariantes de v1il

```
Protocolo C — Invariantes de v1il:

nao_copiar_dados_pesados                = true
nao_versionir_shapefile                 = true
nao_versionir_raster                    = true
nao_versionir_local_runs                = true
nao_versionir_local_only                = true
nao_inferir_data                        = true
nao_usar_mtime_como_data                = true
nao_aceitar_nome_pasta_como_prova       = true
pode_criar_label                        = false
pode_criar_ground_truth_auto            = false
pode_enviar_email                       = false
pode_criar_solicitacao_institucional    = false
markdown_publico                        = "portugues"
sem_supervisao                          = true
sem_clustering_como_classe              = true
sem_flood_prediction_validado           = true
```

---

## Referências

- **v1if:** Official Observed Event Vector Acquisition Audit
- **v1ii:** Targeted Official Repository Event Vector Mining
- **v1ij:** Consolidated Observed Event Vector Evidence
- **v1ik:** Temporal Provenance Recovery
- **Protocolo C:** Protocol for Scientifically Sound Vector Evidence Recovery
- **Protocolo B:** Ground Truth Validation Through Field Assessment (futura)

---

**Data de Execução:** 2026-05-23  
**Versão v1il:** Deep Local Vector Asset Recovery and Bundle Audit  
**Status:** Relatório de Estrutura Preparado, Aguardando Execução  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão, sem data inventada.**
