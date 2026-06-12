# C5/C6 adjudication checklist - REC_2022_05_24_30

| stage | question | current evidence | recommendation | cannot infer |
| --- | --- | --- | --- | --- |
| C5_HUMAN_REVIEW | O mapa Charter realmente cobre Recife? | Mapa raster presente; produto 'Recife/PE' | KEEP_PENDING | Confirmar visualmente; nao inferir cobertura. |
| C5_HUMAN_REVIEW | A feicao e deslizamento, inundacao, dano ou multihazard? | Identificado como LANDSLIDE_SCARS (landslide scars); falta confirmacao oficial | MARK_HAZARD_AMBIGUOUS | Landslide scar nao e flood extent; aguardar confirmacao CENAD. |
| C5_HUMAN_REVIEW | O produto e raster ou vetor? | Raster (PNG) presente; vetor nao confirmado | REQUEST_MORE_EVIDENCE | Raster nao e geometria vetorial. |
| C5_HUMAN_REVIEW | Existe CRS? | CRS ABSENT_OR_UNKNOWN | REQUEST_MORE_EVIDENCE | Sem CRS legivel, geometria nao promove. |
| C5_HUMAN_REVIEW | A geometria pode ser revisada manualmente? | Validade geometrica: NOT_AVAILABLE | KEEP_PENDING | Apenas mapa revisavel; geometria depende de vetor/CRS. |
| C5_HUMAN_REVIEW | A evidencia temporal local existe? | Status temporal: NO_SERIES_AVAILABLE; A301 PRECIP_FULL_GAP | REQUEST_MORE_EVIDENCE | Cemaden/APAC local ainda necessarios. |
| C5_HUMAN_REVIEW | ANA cota e apenas contexto hidrologico? | ANA Capibaribe (Sao Lourenco da Mata/RMR) presente | KEEP_PENDING | Cota nao e precipitacao nem flood extent. |
| C5_HUMAN_REVIEW | APAC PDF mensal e suficiente para C1, mas nao C2 completo? | APAC mensal presente | KEEP_PENDING | Agregado mensal: contexto de C1, nao serie de estacao para C2. |
| C6_CANDIDATE_REFERENCE | Cemaden/APAC local ainda e necessario? | Sim; pendente de download manual/solicitacao | REQUEST_MORE_EVIDENCE | C2 so completa com serie local. |
| C6_CANDIDATE_REFERENCE | Ha base para candidate reference? | C1=TEMPORALITY_SUPPORTED_FOR_HUMAN_REVIEW; C3=PASS; C4=MAP_PRESENT_PENDING_VECTOR_CRS | KEEP_PENDING | Manter CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW. |
| C6_CANDIDATE_REFERENCE | Ha base para ground truth final? | Nao: C2 parcial, C4 sem vetor/CRS, C7 bloqueado | KEEP_PENDING | Nao. Ground truth final permanece proibido (C7 BLOCKED). |
