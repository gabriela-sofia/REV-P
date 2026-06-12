# Recife candidate reference - human review dossier (REC_2022_05_24_30)

## 1. Identificacao do pacote
- Candidate: `REC_2022_05_24_30` | Package: `ARP_v2az_0005` | Event-patch: `FACT_v2at_0005`
- Produto Charter: `CH758_RECIFE_20220602_001` - "Landslides after effects in Recife/PE - Brazil" (2022-06-02)
- Status de referencia: `CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW`

## 2. Resumo do evento Recife maio/2022
- Janela do evento: 2022-05-24 a 2022-06-02 (chuvas extremas, deslizamentos e inundacoes na RMR).

## 3. Evidencia Charter 758
- Mapa raster em resolucao plena presente: `True`.
- Feicao identificada: LANDSLIDE_SCARS (landslide scars).
- CRS: ABSENT_OR_UNKNOWN | Validade geometrica: NOT_AVAILABLE.
- Licenca/fonte: Includes Pleiades material © CNES (2022), Distribution Airbus DS..

## 4. Evidencia APAC/ANA/INMET auditada
- APAC acumulado mensal maio/2022 presente: `True` (contexto).
- ANA HidroWeb cota Capibaribe (Sao Lourenco da Mata/RMR) presente: `True` (contexto hidrologico).
- INMET A301 Recife (chuva local): PRECIP_FULL_GAP - vazia, nao utilizavel.
- Cemaden local presente: `False` (pendente).

## 5. O que cada fonte prova
- Charter 758: existencia de produto cartografico oficial de deslizamento em Recife (ancora espacial).
- ANA cota: que houve resposta hidrologica datada na bacia do Capibaribe na janela.
- APAC mensal: magnitude do mes do evento.

## 6. O que cada fonte NAO prova
- Charter raster NAO prova geometria vetorial nem CRS; NAO e flood extent.
- ANA cota NAO prova precipitacao local nem mancha de inundacao.
- APAC mensal NAO e serie horaria/de estacao (nao fecha C2).
- INMET A301 vazia NAO e substituida por proxy regional (A320 Joao Pessoa e outra cidade/estado).

## 7. Gates C0-C7
- C0_PROVENANCE: `PASS_FOR_REVIEW`
- C1_TEMPORALITY: `TEMPORALITY_SUPPORTED_FOR_HUMAN_REVIEW`
- C2_VALID_SERIES_OR_STATION: `PARTIAL_FOR_HUMAN_REVIEW`
- C3_SPATIAL_ANCHOR: `PASS`
- C4_CANDIDATE_GEOMETRY: `MAP_PRESENT_PENDING_VECTOR_CRS`
- C5_HUMAN_REVIEW: `PENDING`
- C6_CANDIDATE_REFERENCE: `CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW`
- C7_FINAL_GROUND_TRUTH: `BLOCKED`

## 8. C5 checklist
- O mapa Charter realmente cobre Recife? -> recomendacao: `KEEP_PENDING` (Confirmar visualmente; nao inferir cobertura.)
- A feicao e deslizamento, inundacao, dano ou multihazard? -> recomendacao: `MARK_HAZARD_AMBIGUOUS` (Landslide scar nao e flood extent; aguardar confirmacao CENAD.)
- O produto e raster ou vetor? -> recomendacao: `REQUEST_MORE_EVIDENCE` (Raster nao e geometria vetorial.)
- Existe CRS? -> recomendacao: `REQUEST_MORE_EVIDENCE` (Sem CRS legivel, geometria nao promove.)
- A geometria pode ser revisada manualmente? -> recomendacao: `KEEP_PENDING` (Apenas mapa revisavel; geometria depende de vetor/CRS.)
- A evidencia temporal local existe? -> recomendacao: `REQUEST_MORE_EVIDENCE` (Cemaden/APAC local ainda necessarios.)
- ANA cota e apenas contexto hidrologico? -> recomendacao: `KEEP_PENDING` (Cota nao e precipitacao nem flood extent.)
- APAC PDF mensal e suficiente para C1, mas nao C2 completo? -> recomendacao: `KEEP_PENDING` (Agregado mensal: contexto de C1, nao serie de estacao para C2.)

## 9. C6 candidate reference pending
- `CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW` - decisao de referencia deferida a revisao humana, sem promocao.

## 10. Requests pendentes
- Charter vetor/CRS (6 itens) -> templates em request_templates/.
- Cemaden/APAC chuva local (2 itens) -> templates em request_templates/.

## 11. Decisao recomendada
- Manter como `CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW`.
- Solicitar vetor/CRS do Charter (CENAD/Charter).
- Solicitar serie local Cemaden/APAC (Recife/RMR).
- NAO promover a ground truth final.

## 12. Guardrails
- can_create_ground_truth=false; can_create_label=false; can_create_negative=false; can_train_model=false.
- request_pack_is_not_evidence=true; human_review_dossier_is_not_ground_truth=true.
- charter_raster_is_not_vector_geometry=true; ana_stage_is_not_precipitation=true.
- apac_pdf_is_not_station_series=true; inmet_proxy_is_not_local_station=true.
- candidate_reference_is_not_final_truth=true; C7 BLOCKED.
