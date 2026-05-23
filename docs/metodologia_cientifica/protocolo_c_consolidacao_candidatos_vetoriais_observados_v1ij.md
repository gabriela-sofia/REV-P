# Protocolo C — v1ij: Consolidação de Candidatos Vetoriais Observados e Enriquecimento Controlado de Metadados

## 1. Objetivo Científico

A etapa **v1ij** consolida todos os candidatos observados encontrados em v1if, v1ih e v1ii em uma matriz única de decisão com gates padronizados. O objetivo **não é criar dado**, mas auditar sistematicamente quais candidatos reais já existentes em fontes oficiais podem avançar para patch-binding preflight.

## 2. O Que v1ij Consolida

### v1if: Aquisição Oficial de Vetores de Eventos Observados (SGB/CPRM)
- Download e auditoria do ZIP oficial SGB/CPRM de Petrópolis 2022
- Continha 11 PDFs de avaliação de campo por bairro
- **Resultado:** 0 vetores reais. Documentos apenas.

### v1ih: Descoberta de Dados Abertos Locais
- Auditoria de 18 candidatos locais
- Verificação de 8 fontes abertas
- Melhor candidato bloqueado: `Cicatriz_Area_A.shp` (sem data específica de evento)
- **Resultado:** 0 ground truth candidatos. Bloqueio estruturado: ausência de data de evento.

### v1ii: Mineração Dirigida em Repositórios Oficiais
- Scanners de RIGeo/SGB, CKAN Recife, Dados Abertos RJ, etc.
- 12 candidatos registrados de repositórios oficiais
- **Resultado:** 0 ground truth confirmado. Melhor candidato: mesmo `Cicatriz_Area_A.shp`.

### v1ij: Consolidação Estruturada
- **18 candidatos** consolidados de v1if + v1ii
- 10 de Petrópolis, 5 de Recife, 2 de Curitiba, 1 de outras regiões
- Matriz única de gates padronizados
- Enriquecimento controlado de metadados com evidência local/pública existente
- Preflight de patch binding se candidatos passarem gates mínimos

## 3. Por Que v1ij Não Cria Dado

v1ij **lê** registries existentes de v1if, v1ih, v1ii. Não:
- Faz novo download
- Cria novos ativos
- Sintetiza geometry
- Inventa data de evento
- Infere coordenadas
- Transforma risco em ocorrência observada

## 4. Por Que v1ij Não Treina Modelo

v1ij classifica candidatos como **bloqueados** ou **avançáveis para preflight**. Nenhum deles se torna label de treinamento:
- `can_create_training_label = false` (sempre)
- `label_creation_allowed = NO` (sempre)
- Sem target, sem classe, sem supervised learning

Se um candidato passasse preflight, ele seria elegível para **overlay assessment** (comparação com patches reais), não label direto.

## 5. Por Que v1ij Não Usa Solicitação/E-mail

v1ij **consolida apenas dados já públicos e acessíveis**. Não:
- Envia e-mail a instituições
- Abre solicitação formal
- Depende de resposta manual
- Aguarda dados privados

Se um candidato fica bloqueado por falta de metadados públicos, isso é registrado explicitamente como bloqueio estruturado.

## 6. Gates Padronizados (10 Gates Obrigatórios)

Cada candidato passa por 10 gates de decisão:

### Gate 1: Official or Traceable Source
- **PASS:** Instituição oficial (SGB, CPRM, DRM, etc.) ou portal público rastreável
- **FAIL:** Origem desconhecida ou não-oficial

### Gate 2: Vector or Georeferenced Table
- **PASS:** Geometria verificada em SHP, GeoJSON, GPKG, KML, KMZ
- **FAIL:** Documento (PDF), imagem, tabela sem coordenadas

### Gate 3: CRS or Coordinate Reference
- **PASS:** Sistema de referência espacial explícito (EPSG:xxxx)
- **UNKNOWN:** Implícito (lat/lon) ou ausente

### Gate 4: Event Date Available
- **PASS:** Campo ou metadado com data do evento (não apenas data de modificação)
- **FAIL:** Sem data de evento

### Gate 5: Event Date Compatible
- **PASS:** Data dentro da janela do evento-alvo (ex: 2022-02-15 para PET_2022_02_15)
- **FAIL:** Data anterior ao evento ou muito posterior sem justificativa

### Gate 6: Phenomenon Available
- **PASS:** Campo ou evidência de deslizamento, inundação, etc.
- **FAIL:** Sem fenômeno identificável

### Gate 7: Observed Not Risk
- **PASS:** Dado de ocorrência observada real (cicatriz de deslizamento, área inundada, etc.)
- **FAIL:** Mapa de risco/suscetibilidade, modelagem, previsão

### Gate 8: Phenomenon Separable
- **PASS:** Se múltiplos fenômenos, conseguem ser distinguidos
- **NOT_APPLICABLE:** Apenas um fenômeno
- **FAIL:** Fenômenos mistos inseparáveis

### Gate 9: Spatial Unit Usable
- **PASS:** Patch-level ou melhor (não apenas municipal ou point)
- **FAIL:** Apenas nível municipal ou pontos isolados

### Gate 10: Patch Binding Preflight Allowed
- **YES:** Resultado combinado de gates 2, 3, 4, 5, 6, 7, 9 = PASS
- **NO:** Qualquer gate falha

## 7. Como Funciona o Enriquecimento Controlado de Metadados

O enriquecimento **não é inferência livre**. Apenas consulta:

### Permitido:
- Sidecars locais existentes: `.prj`, `.xml`, `.cpg`, `.dbf`, `.shp.xml`
- Metadados públicos já registrados em registries
- Nomes de pasta/arquivo como pista documental (não como prova)
- PDFs/documentos já baixados como referência (sem OCR automático pesado)

### Proibido:
- Inventar data
- Inferir evento por proximidade geográfica
- Usar data de download/modificação do arquivo como data do evento
- Aceitar "2022" no caminho como prova suficiente
- Usar risco/suscetibilidade como ocorrência observada

### Status de Enriquecimento Permitidos:
- `ENRICHED_WITH_DOCUMENTED_EVENT_DATE` — data encontrada em metadados públicos
- `ENRICHED_WITH_DOCUMENTED_PHENOMENON` — fenômeno clarificado
- `ENRICHED_WITH_SOURCE_METADATA` — metadados de fonte complementados
- `PARTIAL_METADATA_SUPPORT` — parcialmente enriquecido
- `NO_METADATA_SUPPORT_FOUND` — sem enriquecimento disponível
- `BLOCKED_AMBIGUOUS_METADATA` — enriquecimento impossível (dados conflitantes)
- `BLOCKED_ONLY_FILE_SYSTEM_DATE` — apenas data de arquivo/pasta (não aceito)

Se um candidato recebe enriquecimento forte, gates são reavaliados. Se ainda faltar data/fenômeno/observed_not_risk, permanece bloqueado.

## 8. Quando Patch Binding Poderia Ser Permitido

Patch binding preflight requer **todos** os gates mínimos:
1. **geometry_available** = YES
2. **crs** presente
3. **event_date_compatible** = PASS
4. **observed_not_risk** = YES
5. **phenomenon_separable** = YES ou NOT_APPLICABLE
6. **spatial_unit_usable** = YES

Se candidato passar, `overlay_allowed = YES` e `label_creation_allowed = NO` (sempre).

O overlay é **assessment visual/espacial**, não label automático.

## 9. Por Que Label Segue Bloqueado

Mesmo se patch binding preflight fosse permitido:
- Nenhum label supervisionado é criado
- Nenhum target é gerado
- `can_create_training_label = false` em todas as linhas
- `label_creation_allowed = NO` em todas as linhas

O candidato seria elegível para **overlay assessment** (comparação vetorial), não label direto.

## 10. Invariantes Permanentes

```
nao_enviar_email                     = true
nao_criar_solicitacao_institucional  = true
nao_inventar_data                    = true
nao_inventar_coordenada              = true
nao_aceitar_risco_como_ocorrencia    = true
nao_aceitar_pdf_como_vetor           = true
nao_treinar_modelo                   = true
nao_criar_label                      = true
nao_reabrir_protocolo_b              = true
nao_versionar_dados_pesados          = true
dados_brutos_apenas_local_runs       = true
publicos_apenas_metadata_registries  = true
markdown_publico_em_portugues        = true
```

## 11. Outputs Gerados

- `consolidated_observed_event_vector_candidate_registry.csv` — matriz consolidada (18 candidatos)
- `consolidated_observed_event_vector_candidate_schema.csv` — schema da matriz
- `consolidated_event_vector_gate_matrix.csv` — auditoria detalhada de gates
- `consolidated_event_vector_gate_matrix_schema.csv` — schema da matriz de gates
- `patch_binding_preflight_candidate_registry.csv` — preflight (vazio ou status report)
- `patch_binding_preflight_candidate_schema.csv` — schema do preflight
- `local_runs/protocolo_c/v1ij/v1ij_consolidation_summary.json` — estatísticas
- `local_runs/protocolo_c/v1ij/v1ij_consolidation_qa.csv` — QA log

## 12. Próximas Etapas Técnicas

### Se candidatos tivessem passado:
1. **Patch binding assessment** — comparar candidato com patches Sentinel-DINO reais
2. **Overlay validation** — verificar sobreposição, cobertura temporal
3. **Documento de decisão** — aceitar/rejeitar candidato para ML

### Como nenhum candidato passou:
1. **Revisar bloqueadores** — gate_02 (14 sem geometria) é o principal
2. **Investigar cicatrizes locais** — `Cicatriz_Area_A.shp` precisa de data documental em metadados públicos
3. **Buscar metadados públicos** — explorar documentação técnica SGB, sidecars em repositórios, fontes abertas
4. **Explorar v1iii** — expandir discovery em novos repositórios para encontrar candidatos com geometria e data

---

**Data de Consolidação:** 2026-05-23  
**Versão:** v1ij-R1  
**Status:** NO_CANDIDATE_PASSED_MINIMUM_PATCH_BINDING_GATES
