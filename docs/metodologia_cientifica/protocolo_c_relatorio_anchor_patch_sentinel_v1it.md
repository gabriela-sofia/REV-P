# Relatório Científico: Auditoria de Footprint Anchor-Patch Sentinel (v1it)

## Resumo Executivo

**RESULTADO PRINCIPAL: Nenhum patch Sentinel existente cobre o anchor oficial**

v1it leu headers de 48 patches Sentinel para Petrópolis (sem nenhum pixel),
calculou footprints em EPSG:32723, e verificou containment do anchor ANEXO-II.

**Status:** AUDITORIA COMPLETA (v1it)  
**Patches avaliados:** 48 (100% com bounds lidos)  
**Patches contendo anchor:** 0  
**Patch mais próximo:** PET_00467 (~290m da borda, sem DINO embedding)  
**Status de cobertura:** NO_PATCH_COVERAGE_FOR_ANCHOR  
**Status multimodal:** BLOCKED_NO_PATCH_COVERAGE_FOR_ANCHOR  

---

## 1. Metodologia de Leitura de Header

```python
# O único acesso ao raster em v1it:
with rasterio.open(path) as src:
    crs = src.crs          # CRS (EPSG:32723)
    bounds = src.bounds    # left, right, bottom, top
    width = src.width      # pixels X
    height = src.height    # pixels Y
    count = src.count      # bandas
    dtype = src.dtypes[0]  # tipo de dado
# Nenhum .read() chamado — zero pixels lidos
```

Transformação de coordenadas:

```python
# WGS84 → EPSG:32723 via pyproj
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32723', always_xy=True)
anchor_x, anchor_y = transformer.transform(-43.211257, -22.484251)
# Resultado: (684023.44, 7512472.86)
```

---

## 2. Resultado de Containment por Patch

### Anchor em EPSG:32723

```
Latitude:   -22.484251  → Y = 7512472.86
Longitude:  -43.211257  → X = 684023.44
CRS alvo:   EPSG:32723 (UTM Zone 23S, WGS84)
```

### Resultado Geral

| Métrica | Valor |
|---------|-------|
| Patches avaliados | 48 |
| Patches com bounds lidos | 48 (100%) |
| Patches contendo anchor | 0 |
| Near-miss (< 2000m da borda) | 2 |

### 5 Patches Mais Próximos do Anchor

| Patch | Dist. Centroide | Dist. Borda | Contém Anchor | DINO |
|-------|----------------|-------------|---------------|------|
| PET_00467 | ~972 m | ~290 m | NÃO | NÃO |
| PET_00431 | — | ~1.6 km | NÃO | NÃO |
| PET_00396 | — | ~2.5 km | NÃO | NÃO |
| PET_00397 | — | ~2.6 km | NÃO | NÃO |
| PET_00362 | — | ~3.5 km | NÃO | NÃO |

### Patches com DINO Embeddings (para referência)

Os 4 patches PET com embeddings extraídos estão todos em outra região do corpus:

| Patch | Dist. Centroide ao Anchor | Contém Anchor |
|-------|--------------------------|---------------|
| PET_00016 | ~13.4 km | NÃO |
| PET_00104 | ~23.6 km | NÃO |
| PET_00119 | ~10.2 km | NÃO |
| PET_00140 | ~24.1 km | NÃO |

---

## 3. Análise de PET_00467 (Patch Mais Próximo)

```
patch_id:       PET_00467
CRS:            EPSG:32723
Bounds:         [684230, 685200] x [7511310, 7512270]
Tamanho:        970m x 960m (~97x96 pixels @ 10m)
Anchor em EPSG:32723: (684023.44, 7512472.86)

Análise de posição do anchor relativo ao patch:
  X: anchor=684023 < left=684230 → 207m a OESTE da borda esquerda
  Y: anchor=7512472 > top=7512270 → 202m ACIMA da borda superior
  Diagonal: ~290m fora da borda NW

Distância ao centroide: ~972m
DINO embedding disponível: NÃO
```

**Conclusão:** PET_00467 é quase adjacente ao anchor, mas não o cobre.
Para cobrir o anchor seria necessário um patch ~207m a oeste e ~202m ao norte
do PET_00467, ou um patch deslocado para cobrir a área de Moinho Preto.

---

## 4. Estado de Prontidão Multimodal

```
official_documented_event_available:    YES
explicit_coordinate_available:          YES
sentinel_patch_coverage_confirmed:      FALSE
dino_embedding_available:               FALSE (para patch coberto)
gis_context_available:                  YES (camada de feições poligonais de deslizamento fotointerpretadas, shapefiles CPRM)
documentary_evidence_available:         YES (relatórios CPRM/DIGEAP)
temporal_precision:                     EXACT_DATE (19/02/2022)
spatial_precision:                      EXACT_COORDINATE

multimodal_reference_status: BLOCKED_NO_PATCH_COVERAGE_FOR_ANCHOR
primary_blocker: NO_EXISTING_PATCH_COVERS_ANCHOR_COORDINATE; ACQUISITION_NEEDED
```

---

## 5. O Que Está Disponível vs. O Que Falta

### Disponível (evidência estrutural robusta)

- Relatório oficial CPRM/DIGEAP com data, localidade, fenômeno
- Coordenada GPS documentada por geólogo de campo
- Corpus de 48 patches Sentinel Petrópolis com bounds verificados
- Patch PET_00467 a ~290m do anchor (candidato próximo)
- Pipeline DINO funcional para extração de embeddings

### Falta (bloqueio imediato)

- Patch Sentinel que cubra a coordenada (-22.484251, -43.211257)
- DINO embedding para esse patch
- (Futuro, não autorizado) Validação de campo — Protocolo B

---

## 6. Próximo Passo Técnico Possível (Não Autorizado Automaticamente)

Se autorizado explicitamente:
1. Adquirir patch Sentinel-2 cobrindo (-22.484251, -43.211257)
   na janela 04/02/2022 a 06/03/2022 (±15 dias da vistoria)
2. Extrair DINO embedding do patch
3. Análise visual/estrutural read-only (sem label)

**Isso não cria label automático. Não abre Protocolo B. Não é ground truth.**

---

## 7. Invariantes

```
can_be_operational_ground_truth  = NO  (invariante absoluto)
can_create_training_label        = NO  (invariante absoluto)
can_train_model                  = NO  (invariante absoluto)
can_reopen_protocol_b            = NO  (invariante absoluto)
```

- [x] Zero pixels lidos em toda a v1it
- [x] Paths privados sanitizados (PROJETO não referenciado em público)
- [x] Nenhuma coordenada inventada
- [x] Nenhum label criado
- [x] Nenhum modelo treinado
- [x] Protocolo B não reaberto
- [x] local_runs/ não versionado

---

**Data de Execução:** 2026-05-24  
**Etapa:** v1it — Official Anchor to Sentinel Patch Footprint Audit  
**Patches PET auditados:** 48 (100% com bounds lidos via header rasterio)  
**Patches cobrindo anchor:** 0  
**Patch mais próximo (borda):** PET_00467, ~290m  
**Status:** NO_PATCH_COVERAGE_FOR_ANCHOR  
**Multimodal:** BLOCKED_NO_PATCH_COVERAGE_FOR_ANCHOR  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
