# SUSC worktree audit before SUSC-17C4

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-17C4: `edc3db3 feat: audita aquisicao oficial e cobertura patch SUSC-17C3`.

Status: auditoria pre-programacao para `SUSC-17C4 - Official Artifact Ingestion`, esteira fail-closed review-only para ingerir, classificar e validar artefatos oficiais/cartograficos ja existentes ou referenciados pelo 17C3, sem inventar geometria, data, coordenada ou footprint.

## Comandos executados

- `git branch --show-current`, `git diff --cached/--name-only`, `git ls-files --others`, `git log -1`
- leitura dos outputs 17C3 (source targets, missing manifest, patch coverage, decisions, summary)
- localizacao real de `PKG_FR_PET_001`, `PKG_FR_REC_002`, SGB/CPRM PDFs/ZIP, geometrias REC_2022_05_24_30
- preflight validators 17C3/17C2/17A/16D

## Estado objetivo

- Branch: `marco/pre-unificacao-gates-mv1`. Staged: vazio.
- Entradas consumidas modificadas localmente: nenhuma.
- Preflight 17C3/17C2/17A/16D: PASSED.

## Artefatos oficiais localizados (reais, nao inventados)

- `datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758/derived/event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson` (TRACKED): MultiPolygon real digitalizado do Charter 758 para REC_2022_05_24_30, CRS EPSG:4326, `review_status=provided_unreviewed`, props com `patch_id=REC_00019` (esquema Protocolo C), `source_crs=EPSG:32725`. -> unica geometria de evento ingerivel.
- `local_runs/protocolo_c/v1if/raw_official_sources/Relatorio_Tecnico_Petropolis.pdf` (4.3MB, gitignored): SGB/CPRM, PDF sem vetor.
- `local_runs/protocolo_c/v1if/raw_official_sources/anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip` (20MB, gitignored): SGB/CPRM, abre como `BadZipFile` (formato invalido/corrompido) -> sem vetor extraivel por parser simples.
- `PKG_FR_PET_001` (DRM-RJ/NADE) e `PKG_FR_REC_002` (COMPDEC/Defesa Civil PE): nao presentes localmente -> `missing_source_artifact` (PENDING_FORMAL_REQUEST).
- `PET_2024_03_21_28` (Valparaiso-Floresta): NOT_ACQUIRED.
- PE3D/MDE: camada fisica contextual, nao evento.

## Cobertura de patches (teste real bbox x grade)

O MultiPolygon Charter 758 (bbox lon[-34.942,-34.921] lat[-8.001,-7.982]) cai FORA da grade SUSC `recife_*` (lon[-35.142,-34.944] lat[-8.227,-8.014]) -> 0 intersecoes. O `patch_id=REC_00019` da property pertence ao esquema de patches do Protocolo C, NAO a grade de suscetibilidade `susc_features_by_patch`. Logo nao ha patch-link valido na grade SUSC.

## Politica de ingestao fail-closed (esta sprint)

- geojson/vetor explicito -> ingerir como candidato (qa_status=needs_review), testar contra grade.
- PDF sem vetor/coordenada -> bloquear (`blocked_pdf_only_no_vector`); sem OCR pesado (regra 10).
- ZIP sem vetor (ou corrompido) -> bloquear (`blocked_unknown_format`); sem download/raster pesado.
- artefato ausente -> nao cria referencia; gera solicitacao formal.
- PE3D/MDE -> `blocked_context_layer_not_event`.
- nenhuma geometria/coordenada/bbox inventada; centroide de bairro nunca vira geometria forte.
- arquivos privados em `local_runs/` referenciados como ponteiro; nunca stageados; sha256/tamanho de bruto privado nao registrados.

## Sujeira fora do escopo (preservada)

11 tracked `revp_v2e*`; 473 untracked. Preservados.

## Decisao fail-closed

Prosseguir review-only. Stage seletivo somente do pacote 17C4. Score v6 intacto, score v7 inexistente, nada vira ground truth, 17B nao executado, SAR runtime nao executado.
