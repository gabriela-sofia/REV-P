# Protocolo C — Auditoria de Footprint Anchor-Patch Sentinel (v1it)

## Por Que v1is Foi um Avanço Real

A v1is estabeleceu o primeiro **spatial anchor com coordenada GPS explícita**
documentada em relatório oficial (CPRM/DIGEAP):

```
ANCHOR_PET2022_CPRM_ANEXOII_19022022
Localidade: Moinho Preto, Petrópolis, RJ
Data: 19/02/2022
Coordenada: -22.484251, -43.211257 (Rua Stephan Zweg)
Fonte: CPRM_FIELD_SURVEY_REPORT
```

Isso encerrou o ciclo documental. Todos os 10 candidatos documentais da v1ir foram
avaliados. O único ponto com GPS explícito virou anchor. Os demais ficaram como
referências documentais por localidade.

O limite da v1is: os patches Sentinel tinham `centroid_lat_lon` vazio — não era
possível verificar se algum cobria o anchor sem ler os arquivos raster.

---

## Por Que Testamos Footprint de Patch na v1it

A v1it resolve exatamente esse bloqueio: lendo **apenas headers/metadados**
dos arquivos raster (CRS, bounds, transform, dimensões), sem nenhum pixel,
calcula-se o footprint de cada patch e verifica-se se o anchor oficial cai
dentro de algum deles.

**O que é lido:** `rasterio.open(path)` → CRS, bounds, width, height, count, dtype  
**O que NÃO é lido:** nenhum `.read()`, nenhum `.read_band()`, nenhum dado espectral

---

## Diferença: Spatial Anchor, Patch Coverage e Label

| Conceito | Descrição | Status |
|---------|-----------|--------|
| **Spatial anchor** | Ponto GPS documentado em relatório oficial | EXISTE (ANEXO-II) |
| **Patch coverage** | Patch Sentinel cujo footprint contém o anchor | NÃO EXISTE (nenhum dos 48) |
| **DINO embedding** | Vetor 768-dim extraído do patch via DINOv2 | EXISTE apenas para patches distantes |
| **Multimodal reference** | Anchor + patch coberto + DINO embedding | BLOQUEADO por falta de patch |
| **Label de treino** | Alvo supervisionado para ML | INVARIANTE: NUNCA (sem Protocolo B) |

---

## Resultado da Auditoria v1it

### Patches Avaliados

```
Patches PET no corpus:    48
Patches com bounds lidos: 48 (100% leiveis via rasterio header)
CRS dos patches:          EPSG:32723 (UTM Zone 23S)
Tamanho típico:           ~960m x 960m (~96x96 pixels @ 10m)
Bandas:                   6
```

### Anchor e Cobertura

```
Anchor:           ANCHOR_PET2022_CPRM_ANEXOII_19022022
Coordenada WGS84: -22.484251, -43.211257
Em EPSG:32723:    (684023.44, 7512472.86)

Patches contendo anchor: 0 (de 48 avaliados)
Status final:            NO_PATCH_COVERAGE_FOR_ANCHOR
```

### Patches Mais Próximos

| Patch ID | Dist. Centroide | Dist. Borda | DINO Embedding |
|----------|----------------|-------------|----------------|
| PET_00467 | ~972 m | ~290 m | NÃO |
| PET_00431 | — | ~1.6 km | NÃO |
| PET_00396 | — | ~2.5 km | NÃO |

**Observação:** PET_00467 é o mais próximo — borda a ~290m do anchor.
O anchor está ~207m a oeste e ~202m acima do topo do patch.
Nenhum dos 4 patches com DINO embeddings está próximo do anchor
(PET_00016, 00104, 00119, 00140 estão a 10–24 km).

---

## Por Que ML Segue Bloqueado

```
Bloqueio 1 (imediato): Nenhum patch Sentinel cobre o anchor (-22.484251, -43.211257)
Bloqueio 2 (estrutural): Não há DINO embedding para o patch mais próximo (PET_00467)
Bloqueio 3 (absoluto):  Protocolo B não iniciado → sem ground truth operacional
Bloqueio 4 (absoluto):  can_create_training_label = NO (invariante)
```

Para análise estrutural read-only seria necessário:
1. Um patch Sentinel cobrindo o anchor — não existe no corpus atual
2. Extrair DINO embedding para esse patch
3. Análise visual/embedding read-only (sem label)

Para ML supervisionado seria necessário adicionalmente:
4. Protocolo B (validação de campo) — não iniciado e não autorizado
5. Ground truth operacional — invariante bloqueado

---

## Invariantes

```
can_be_operational_ground_truth  = NO  (invariante absoluto)
can_create_training_label        = NO  (invariante absoluto)
can_train_model                  = NO  (invariante absoluto)
can_reopen_protocol_b            = NO  (invariante absoluto)
```

- [x] Apenas headers raster lidos — nenhum pixel
- [x] Paths privados sanitizados em todos os outputs
- [x] Nenhuma coordenada inventada
- [x] Nenhum label criado
- [x] Protocolo B não reaberto
- [x] local_runs/ não versionado

---

**Versão:** v1it — Official Anchor to Sentinel Patch Footprint Audit  
**Patches avaliados:** 48 PET (100% com bounds lidos via header)  
**Patches cobrindo anchor:** 0  
**Patch mais próximo:** PET_00467 (~290m da borda)  
**Status multimodal:** BLOCKED_NO_PATCH_COVERAGE_FOR_ANCHOR  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
