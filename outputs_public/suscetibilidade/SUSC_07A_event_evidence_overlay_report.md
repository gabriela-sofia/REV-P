# SUSC-07A — Inventário e Overlay Inicial de Evidências Documentais/Eventos

> O SUSC-07A não cria ground truth de enchente por patch. Ele organiza evidências documentais e testa aderência espacial/documental entre patches suscetíveis e registros externos, preservando o status review-only e a distinção entre suscetibilidade, ocorrência observada e validação supervisionada.

---

## 1. Objetivo do SUSC-07A

Criar a primeira camada de comparação entre patches de suscetibilidade (alta/média/baixa), evidências documentais/registros de eventos, e as três regiões (Recife, Petrópolis, Curitiba), usando geometria/bbox dos patches e período/data quando existir. Produz uma **matriz de aderência documental/espacial**, não ground truth.

## 2. Relação com SUSC-03/04/05/06

- **SUSC-03** forneceu a matriz de patches (300×72) com bbox (xmin/ymin/xmax/ymax) e o proxy de suscetibilidade.
- **SUSC-04/05** auditaram proveniência e direção das features (feature cards).
- **SUSC-06A/06B** ajustaram e diagnosticaram o baseline proxy e apontaram a **lacuna**: falta evidência documental/temporal de evento. O SUSC-07A ataca exatamente essa lacuna — sem fechá-la como verdade.

## 3. O que significa evidência documental

Registro externo (relatório CPRM/SGB, Defesa Civil, índice de eventos) que **menciona** um evento (enchente/alagamento/desastre) numa região e, às vezes, numa data. É contexto observacional, não uma medição por patch.

## 4. Diferença entre evidência documental, evento observado, suscetibilidade e ground truth

| Conceito | O que é | Aqui |
|----------|---------|------|
| Suscetibilidade | Predisposição física/heurística (proxy) | matriz SUSC-03 |
| Evidência documental | Menção textual a evento numa região/data | catálogo SUSC-07A |
| Evento observado | Ocorrência confirmada com geometria/data precisa | **ausente** (sem coordenada) |
| Ground truth | Rótulo validado por patch para treino | **não existe / proibido** |

## 5. Fontes encontradas

Scan offline (`SUSC_07A_event_source_scan.csv`): **3.771 arquivos varridos**, 1.970 com termos de evento — 560 `candidate_evidence_source`, 244 `documentary_context`, 109 `needs_manual_review`, 304 `methodological_reference`, 753 `not_event_evidence`. Fontes estruturadas reais usadas no catálogo: `revp_observed_event_registry_v2dz.csv` e `revp_indice_eventos_externos_candidatos_mv1.csv`.

## 6. Eventos/evidências catalogadas

`susc_07_event_evidence_catalog_v1.csv` — **61 evidências** normalizadas, todas `can_be_ground_truth=false`, `allowed_for_training=false`, `review_only=true`.

Por `evidence_status`: 10 `approximate_spatial_evidence`, 10 `region_period_context`, 20 `documentary_context_only`, 21 `insufficient_for_patch_event_link`.

## 7. Quantidade por região

| região | evidências | patches |
|--------|-----------|---------|
| petropolis | 27 | 100 |
| curitiba | 8 | 100 |
| recife | 5 | 100 |
| unknown | 21 | — |

## 8. Quantidade com data

**27 de 61** evidências têm data/período (`temporal_status` day/period/month/year).

## 9. Quantidade com geometria

**11 de 61** têm flag de geometria — porém **sem coordenada/bbox utilizável** (`geometry_status=flagged_unvalidated_no_coordinate`). Geometria sinalizada ≠ geometria usável.

## 10. Quantidade linkável a patch

**0 de 61.** Nenhuma evidência tem coordenada precisa → `can_link_to_patch=false` em todas. Honesto e esperado.

## 11. Resultados do overlay

`SUSC_07A_patch_event_overlay_candidates.csv` — 61 associações em **nível de região** (patch_id = `REGION_LEVEL_NO_PATCH_RESOLUTION`):

| spatial_relation | nº |
|------------------|----|
| same_region_period | 19 |
| same_region_only | 1 |
| documentary_context_only | 20 |
| insufficient_for_patch_link | 21 |
| exact_patch_overlap | 0 |
| bbox_overlap | 0 |

## 12. Casos com maior aderência

**Petrópolis** concentra a maior aderência: 18 `same_region_period` (evidências com região + data, em geral relatórios CPRM 2022). Recife tem 1 `same_region_period`. São associações **regionais datadas**, não overlaps de patch — não confirmam ocorrência em patch específico.

## 13. Casos apenas regionais/documentais

20 `documentary_context_only` (texto sem data útil) e 21 `insufficient_for_patch_link` (região desconhecida). Curitiba aparece só como `documentary_context_only` (8).

## 14. Limitações

`SUSC_07A_patch_event_overlay_limitations.json`:
- **Nenhum overlap espacial de patch é possível** sem coordenada do evento.
- Associação é em nível de região; não identifica patch específico.
- Geometria sinalizada nas fontes é não-validada/sem coordenada.
- Apenas parte das evidências tem data; região desconhecida em 21 casos.
- bbox do patch existe (SUSC-03), mas falta o lado do evento.

## 15. O que ainda NÃO pode ser afirmado

- Que algum patch específico sofreu enchente.
- Que suscetibilidade alta num patch corresponde a evento observado.
- Que qualquer associação aqui seja ground truth ou rótulo de treino.

## 16. O que JÁ pode ser afirmado

- Existe evidência documental de eventos por região (sobretudo Petrópolis 2022), com data em 27 casos.
- O lado dos patches está pronto geometricamente (bbox); o gargalo é a **geometria/coordenada do evento**.
- A aderência possível hoje é **regional/documental**, devidamente rotulada e com confiança baixa.

## 17. Como isso prepara SUSC-07B/SUSC-08

- **SUSC-07B:** adquirir geometria/coordenada real dos eventos (Defesa Civil/CPRM/APAC/GeoCuritiba) para habilitar overlay bbox/point por patch — ainda sem promover a ground truth automaticamente.
- **SUSC-08:** com geometria de evento, confrontar proxy × evidência por patch e, só então, discutir critérios de referência sob revisão humana.

---

## Disclaimer obrigatório

> O SUSC-07A não cria ground truth de enchente por patch. Ele organiza evidências documentais e testa aderência espacial/documental entre patches suscetíveis e registros externos, preservando o status review-only e a distinção entre suscetibilidade, ocorrência observada e validação supervisionada.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
