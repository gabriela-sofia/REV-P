# Relatório Científico: Ancoragem Espacial de Eventos Documentados Oficiais (v1is)

## Resumo Executivo

**RESULTADO PRINCIPAL: 1 spatial anchor candidate criado**

v1is converteu as 10 unidades documentais candidatas da v1ir em spatial anchors,
aplicando a regra estrita de coordenada GPS explícita.

**Status:** AUDITORIA COMPLETA (v1is)  
**Unidades v1ir lidas:** 11  
**Spatial anchors criados:** 1 (ANEXO-II, Moinho Preto, 19/02/2022)  
**Documentary reference only:** 9 (ANEXO-III a XI — sem coordenada explícita)  
**Insuficiente:** 1 (Relatorio_Tecnico_Petropolis — sem data estruturada)  
**Sentinel readiness:** PARTIAL_PATCH_FOUND_COORD_UNVERIFIED  
**Multimodal readiness:** PARTIAL_COORD_VERIFICATION_NEEDED  

---

## 1. Regra de Elegibilidade para Spatial Anchor

Apenas unidades com **todos** os seguintes atributos podem virar spatial anchor:

| Critério | Exigência |
|----------|-----------|
| `coordinate_available` | YES |
| `coordinate_source` | CPRM_FIELD_SURVEY_REPORT |
| `ground_reference_candidate_status` | CANDIDATE_WITH_DOCUMENTED_COORDINATE |
| Latitude/longitude | Presentes e extraídas do texto do relatório |

A unidade ANEXO-II é a única a satisfazer todos os critérios.

---

## 2. Spatial Anchor Criado

### ANCHOR_PET2022_CPRM_ANEXOII_19022022

```
Anchor ID:        ANCHOR_PET2022_CPRM_ANEXOII_19022022
Unidade origem:   PET2022_CPRM_ANEXOII_19022022
Instituição:      CPRM / DIGEAP
Data vistoria:    19/02/2022
Localidade:       Moinho Preto, Petrópolis, RJ
Coordenada GPS:   -22.484251, -43.211257
Ponto de campo:   Rua Stephan Zweg
Fenômeno:         MOVEMENT_OF_MASS (solapamento + enxurrada + risco edificações)
Precisão espacial: EXACT_COORDINATE
Fonte:            CPRM_FIELD_SURVEY_REPORT
Status:           SPATIAL_ANCHOR_CANDIDATE
```

**Contexto documental:**
"Solapamento da margem direita do afluente do Rio Piabanha derrubou parte do
acostamento, um poste de iluminação e mobilizou blocos de construção. Residências
próximas, com uma casa de alvenaria e madeira parcialmente danificadas por enxurrada."

---

## 3. Unidades Sem Coordenada (Documentary Reference Only)

9 unidades permanecem como referências documentais sem âncora geoespacial:

| Anexo | Data | Localidade | Status |
|-------|------|-----------|--------|
| III | 20/02/2022 | Serra Velha, Pontilhão | DOCUMENTARY_REFERENCE_ONLY |
| IV | 22/02/2022 | Valparaíso, Rua Eugênio Barcelos | DOCUMENTARY_REFERENCE_ONLY |
| V | 23/02/2022 | Rua Teresa e imediações | DOCUMENTARY_REFERENCE_ONLY |
| VI | 24/02/2022 | Moinho Preto (revisita) | DOCUMENTARY_REFERENCE_ONLY |
| VII | 24/02/2022 | Mosella | DOCUMENTARY_REFERENCE_ONLY |
| VIII | 25–26/02/2022 | Estrada Velha, Vila Felipe | DOCUMENTARY_REFERENCE_ONLY |
| IX | 28/02/2022 | Sargento Boening | DOCUMENTARY_REFERENCE_ONLY |
| X | 01/03/2022 | Alto da Serra | DOCUMENTARY_REFERENCE_ONLY |
| XI | 02/03/2022 | Quitandinha | DOCUMENTARY_REFERENCE_ONLY |

**Motivo:** `NO_EXPLICIT_COORDINATE` — coordenada GPS não documentada nos relatórios.
Geocodificação de bairro/rua sem coordenada explícita não é permitida.

---

## 4. Prontidão Sentinel

### Janela Temporal do Anchor

```
Data de referência:    19/02/2022 (vistoria CPRM)
Janela de busca:       04/02/2022 → 06/03/2022 (±15 dias)
Revisita esperada:     5 dias (Sentinel-2A + 2B)
Resolução:             10 m (bandas espectrais 10m)
```

### Avaliação dos Registros Locais

| Registro | Patches Totais | Patches Petrópolis | Coordenadas PET |
|----------|---------------|-------------------|-----------------|
| v1ge (DINO manifest) | 12 | 4 | Não disponíveis |
| v1gt (patch extent) | 128 | 48 | `centroid_lat_lon` vazio |

### Status de Prontidão

```
sentinel_patch_candidate_available: PATCH_FOUND_COORD_UNVERIFIED
sentinel_readiness_status:          PARTIAL_PATCH_FOUND_COORD_UNVERIFIED
blocking_reason:                    PET_PATCH_CENTROID_EMPTY_CANNOT_VERIFY_PROXIMITY_TO_ANCHOR
```

**Interpretação:** Patches Sentinel/DINO para Petrópolis existem localmente (4 no manifest,
48 no registro de extensão), mas o campo `centroid_lat_lon` está vazio para todos.
Não é possível verificar se algum patch cobre o anchor (-22.484251, -43.211257).

---

## 5. Prontidão Multimodal (DINO + Sentinel)

```
DINO backbone:            facebook/dinov2-with-registers-base
DINO embedding_dim:       768
Patches PET com coords:   0 (centroid_lat_lon vazio)
dino_patch_region_match:  YES_COORD_UNVERIFIED
dino_patch_coordinate_match: CANNOT_VERIFY_CENTROID_EMPTY

multimodal_readiness_status: PARTIAL_COORD_VERIFICATION_NEEDED
can_extract_embedding:        CONDITIONAL
can_create_label:             NO  (invariante)
can_train_model:              NO  (invariante)
```

**Próximo passo:** Popular `centroid_lat_lon` no registro v1gt para patches PET
e verificar cobertura do anchor antes de extrair embeddings.

---

## 6. Gates de Elegibilidade (Spatial Anchor)

| Gate | ANEXO-II | ANEXO-III a XI |
|------|----------|----------------|
| `coordinate_available` | YES | NO |
| `coordinate_source` | CPRM_FIELD_SURVEY_REPORT | NOT_DOCUMENTED |
| `candidate_status` | CANDIDATE_WITH_DOCUMENTED_COORDINATE | CANDIDATE_DOCUMENTARY_ONLY |
| `spatial_precision` | EXACT_COORDINATE | NEIGHBORHOOD |
| **→ Elegível para anchor** | **SIM** | **NÃO** |

---

## 7. Comparação com Etapa Anterior (v1ir)

| Aspecto | v1ir (evento documental) | v1is (spatial anchor) |
|---------|--------------------------|----------------------|
| **Unidade** | Relatório com data + localidade | Ponto GPS documentado no relatório |
| **Critério principal** | EXACT_DATE + fenômeno + localidade | Coordenada GPS explícita |
| **Candidatos** | 10 (9 doc + 1 com coord) | 1 (com GPS explícito) |
| **Precisão espacial** | NEIGHBORHOOD / EXACT_COORD | EXACT_COORDINATE only |
| **Próximo passo** | v1is (este) | Verificar cobertura de patch Sentinel |

---

## 8. Invariantes

```
can_be_operational_ground_truth  = NO  (invariante absoluto)
can_create_training_label        = NO  (invariante absoluto)
can_train_model                  = NO  (invariante absoluto)
can_reopen_protocol_b            = NO  (invariante absoluto)
```

- [x] Nenhuma coordenada inventada ou inferida
- [x] Nenhuma geocodificação de bairro sem GPS explícito
- [x] Nenhum centroid de polígono usado como coordenada
- [x] Nenhum label criado
- [x] Nenhum modelo treinado
- [x] Protocolo B não reaberto
- [x] Sem path privado em arquivos públicos
- [x] local_runs/ não versionado

---

## 9. Próximos Passos (Se Autorizados)

### Para resolver o bloqueio Sentinel:

```
1. Popular centroid_lat_lon no registro v1gt (local_runs/dino_embeddings/v1gt/)
   para patches PET — operação de metadados, não de aquisição
2. Verificar se algum patch PET cobre (-22.484251, -43.211257)
3. Se patch disponível: extrair DINO embedding do patch correspondente
4. Análise visual de evidência estrutural (read-only, sem label)
```

### O Que Permanece Bloqueado:

- Ground truth operacional → requer Protocolo B (não iniciado)
- Labels de treino → invariante absoluto
- Treinamento de modelo → invariante absoluto
- Reabrir Protocolo B → invariante absoluto

---

**Data de Execução:** 2026-05-24  
**Etapa:** v1is — Official Event Unit Spatial Anchor & Sentinel Readiness  
**Spatial anchors:** 1 (ANCHOR_PET2022_CPRM_ANEXOII_19022022)  
**Coord:** -22.484251, -43.211257 (Moinho Preto, 19/02/2022)  
**Sentinel readiness:** PARTIAL_PATCH_FOUND_COORD_UNVERIFIED  
**Multimodal readiness:** PARTIAL_COORD_VERIFICATION_NEEDED  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
