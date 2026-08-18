# SUSC-07B — Enriquecimento com Coordenadas Reais e Overlay Refinado

> O SUSC-07B não cria ground truth de enchente por patch. Ele busca e extrai coordenadas reais rastreáveis de eventos/evidências documentais para melhorar a aderência espacial, preservando a distinção entre associação espacial, evidência documental, suscetibilidade e validação supervisionada.

---

## 1. Objetivo do SUSC-07B

Obter coordenadas reais **rastreáveis** para eventos/evidências do catálogo SUSC-07A (priorizando fontes oficiais/públicas) e rodar overlay refinado patch × evento apenas quando a geometria do evento for suficientemente rastreável — sem inventar coordenada e sem criar ground truth.

## 2. Estado herdado do SUSC-07A

61 evidências documentais; 27 com data; 11 com flag de geometria; **0 linkáveis a patch**; 0 exact_patch_overlap; 0 bbox_overlap. Bloqueio: patches têm bbox, eventos não tinham coordenada usável.

## 3. Critério de coordenada real

Coordenada aceita **somente** de fonte rastreável (GeoJSON, CSV lat/lon, KML, WKT, SHP, PDF técnico, tabela oficial, manifesto com URL+SHA256). **Proibido:** centroide de município/bairro, geocoding genérico, ponto estimado, coordenada sem fonte, inferência por nome de localidade. Região atribuída por **bounds WGS84** (sanidade, não invenção). Arquivos com nome `centroid/template/synthetic/placeholder/empty/example` são rejeitados.

## 4. Fontes locais escaneadas

`SUSC_07B_real_coordinate_source_scan.csv` — **451 arquivos** com sinais de coordenada; **33 candidatos de coordenada real** (recife 19, petropolis 9, curitiba 5); 1 vetor/doc exige parser manual.

## 5. Fila de aquisição externa

`SUSC_07B_external_coordinate_acquisition_queue.csv` — **200 linhas** (40 evidências com região × 5 fontes oficiais prioritárias por região: CPRM/SGB, Defesa Civil, INEA, APAC, GeoCuritiba, IPPUC, Águas Paraná/IAT, CEMADEN, ANA/Hidroweb), com `suggested_search_query`. `download_allowed=false`, `manual_review_required=true`.

## 6. Fontes externas consultadas/baixadas

`SUSC_07B_external_coordinate_acquisition_report.csv` — **200 `not_attempted_no_direct_url_or_no_network`** (offline-safe; sem URL direta, sem download pesado, sem API, sem geocoding). Diretório `external_event_geometry_sources/` criado e vazio (aguarda colocação manual de fonte oficial). As coordenadas reais desta passada vieram de **fontes locais já no repo**.

## 7. Coordenadas reais extraídas

`susc_07b_real_event_coordinates_v1.csv` — **13 registros** de geometria rastreável: recife 8, petropolis 5; por tipo: 3 polígono, 10 conjunto-de-pontos. Fontes reais: Charter 758 (event polygon Recife 2022), Defesa Civil Recife (risk locations/areas, pontos), patch boundary/AOI Recife, registries oficiais CPRM/âncora de Petrópolis e estações INMET. CRS `EPSG:4326`. **Todas `can_be_ground_truth=false`, `review_only=true`, `requires_manual_review=true`** (vínculo evidência↔coordenada é regional, sem token de evento explícito).

## 8. Coordenadas rejeitadas e motivo

`SUSC_07B_real_coordinate_extraction_audit.csv`: 2 `rejected_non_real_source` (nome `template/synthetic/empty`), 18 `no_in_region_coordinate` (sem coordenada dentro dos bounds das três regiões). Nada inventado.

## 9. Evidências enriquecidas

`susc_07b_event_evidence_geometry_enriched_v1.csv` — **32 de 61** evidências (todas de recife/petropolis) recebem `enriched_has_geometry=true` em **nível de região** (geometria real disponível na região), mas `can_link_to_patch_after_enrichment=false` (a evidência documental não está amarrada à sua própria coordenada; `enrichment_confidence=low`). Curitiba sem geometria real nesta passada.

## 10. Evidências linkáveis a patch antes/depois

- **Antes (07A):** 0 linkáveis; 0 overlap espacial.
- **Depois (07B):** a geometria **rastreável** habilita overlay espacial real, embora o vínculo evidência↔coordenada permaneça `requires_manual_review`. Linkabilidade a patch é **espacial** (geometria + região + CRS), nunca ground truth.

## 11. Resultado do overlay refinado

`SUSC_07B_refined_patch_event_overlay.csv` (70 linhas) e `_summary.csv`:

| região | bbox_overlap | near_patch_buffer | same_region_period | same_region_only | documentary | insufficient |
|--------|--------------|-------------------|--------------------|------------------|-------------|--------------|
| recife | 5 | 3 | 1 | 0 | 4 | 0 |
| petropolis | 0 | 1 | 18 | 1 | 8 | 0 |
| curitiba | 0 | 0 | 0 | 0 | 8 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 | 21 |

**0 `exact_patch_overlap`** (nenhuma geometria exata patch-evento). **9 relações espaciais reais** (5 bbox_overlap + 4 near_patch_buffer_candidate) em **6 patches únicos**.

## 12. Casos exact/bbox/near encontrados

- **bbox_overlap (5, Recife):** bbox do polígono de evento Charter 758 (REC_2022_05_24_30) sobrepõe o bbox de 5 patches Recife. Polígono é **candidato digitalizado review-only**, não footprint validado.
- **near_patch_buffer_candidate (4):** Recife (3) — pontos de localidade/risco da Defesa Civil dentro do bbox de 3 patches; Petrópolis (1) — coordenada oficial (registry CPRM/âncora) dentro do bbox de 1 patch.
- **exact_patch_overlap:** 0.

## 13. Casos ainda regionais/documentais

19 `same_region_period` (sobretudo Petrópolis 2022/CPRM), 1 `same_region_only`, 20 `documentary_context_only` (incl. Curitiba 8) e 21 `insufficient_for_patch_link` (região desconhecida). Camada documental do SUSC-07A preservada.

## 14. Lacunas remanescentes

- Vínculo evidência↔coordenada é regional → `requires_manual_review` em todas.
- Geometrias são candidatas/digitalizadas ou pontos de risco, **não** footprints validados de enchente.
- Pontos de risco da Defesa Civil ≠ ocorrência confirmada no patch.
- Curitiba sem coordenada real extraída nesta passada (aquisição GeoCuritiba/IPPUC pendente).
- Fontes vetoriais pesadas (SHP/GPKG/KMZ/PDF) exigem parser/aquisição manual.

## 15. Impacto no projeto

Pela primeira vez há **aderência espacial real** entre patches de suscetibilidade e geometria de evento rastreável (Recife forte; Petrópolis 1 ponto), com toda a cadeia review-only. Isso destrava o caminho para uma validação humana de aderência (SUSC-08) — sem nenhuma promoção automática a ground truth ou treino.

## 16. Próximo marco recomendado

**SUSC-08 — Validação humana da aderência espaço-temporal**: revisar os 9 casos espaciais (5 bbox + 4 near) e os candidatos CPRM Petrópolis, confirmar CRS/footprint com fonte oficial, e só então discutir critério de referência sob revisão humana. Em paralelo: aquisição oficial GeoCuritiba/IPPUC e parser vetorial (SHP/GPKG).

---

## Disclaimer obrigatório

> O SUSC-07B não cria ground truth de enchente por patch. Ele busca e extrai coordenadas reais rastreáveis de eventos/evidências documentais para melhorar a aderência espacial, preservando a distinção entre associação espacial, evidência documental, suscetibilidade e validação supervisionada.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
