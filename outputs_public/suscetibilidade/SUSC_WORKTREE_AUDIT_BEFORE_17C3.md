# SUSC worktree audit before SUSC-17C3

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-17C3: `f4c2dc9 feat: prepara execucao SAR de footprints candidatos SUSC-17C2`.

Status: auditoria pre-programacao para `SUSC-17C3 - Official Source Acquisition & Patch Coverage Audit`, esteira programatica review-only para registrar fontes oficiais faltantes, priorizar aquisicao real, auditar cobertura de patches e decidir o caminho (aquisicao oficial / expansao de grade / SAR runtime / troca de canary).

## Comandos executados

- `git branch --show-current`, `git status --short`, `git diff --name-only`, `git ls-files --others --exclude-standard`, `git log --oneline -1`
- leitura dos outputs 17A/17C/17C2
- busca das referencias `PKG_FR_PET_001`, `PET_2022_02_15`, `PET_2024_03_21_28`, `REC_2022_05_24_30`, `COMPDEC`, `DRM-RJ`, `SGB`, `CPRM`, `PE3D`, `v1hz`, `v1hr`
- teste bbox do International Charter contra as 3 grades de patches

## Estado objetivo

- Branch: `marco/pre-unificacao-gates-mv1`. Staged: vazio.
- Entradas consumidas modificadas localmente: nenhuma.

## Referencias oficiais encontradas (reais, nao inventadas)

- `datasets/official_observed_event_vector_registry.csv` (6 linhas): SGB/CPRM (PDF/ZIP ingeridos), DRM-RJ/NADE `PKG_FR_PET_001` (PENDING_FORMAL_REQUEST), Defesa Civil Petropolis, COMPDEC/Defesa Civil PE `PKG_FR_REC_002` (PENDING_FORMAL_REQUEST), Prefeitura Curitiba/IPPUC.
- `datasets/event_sentinel_temporal_window_registry.csv`: `PET_2024_03_21_28` (Chuvas fim de marco 2024, Valparaiso-Floresta), com janelas pre/pos, `acquisition_status=NOT_ACQUIRED`.
- `datasets/external_evidence_registry.csv`: `recife_pe3d_mde` PE3D/MDE (terrain, tier STRONG) = camada fisica contextual, NAO evento.
- `outputs_public/suscetibilidade/susc_17c2_candidate_footprints.geojson`: footprint International Charter REC 2022-05-24 (unica geometria de evento disponivel).

## Cobertura de patches (teste real bbox x grade)

O bbox do International Charter (Recife, lon -34.94..-34.92, lat -8.00..-7.98) cai FORA das 3 grades:
- recife: lon[-35.142,-34.944] lat[-8.227,-8.014] -> 0 intersecoes (adjacente a borda NE).
- curitiba/petropolis: 0 (outra regiao).
Achado: a unica geometria de evento disponivel esta fora da grade Recife.

## Status das fontes prioritarias

- DRM-RJ `PKG_FR_PET_001`: PENDING_FORMAL_REQUEST, NOT_INGESTED -> P0 missing_source_artifact.
- COMPDEC `PKG_FR_REC_002`: PENDING_FORMAL_REQUEST, NOT_INGESTED -> P0 missing_source_artifact.
- SGB/CPRM Relatorio_Tecnico_Petropolis.pdf + anexos.zip: DOWNLOAD_OK/INGESTED mas sem vetor -> pdf_only (precisa extracao vetor/coordenada).
- PET_2024 Valparaiso: NOT_ACQUIRED -> missing geometry.
- PE3D/MDE: contextual_physical_layer (nao evento).

## Sujeira fora do escopo (preservada)

11 tracked `revp_v2e*`; 473 untracked. Preservados.

## Nota de reprodutibilidade

Preflight usa apenas validators especificos (16D/17A/17C/17C2). Sem reexecutar pipelines que sujam outputs 16A/16B/16C. Qualquer arquivo fora do escopo sujado sera restaurado com `git checkout --` antes do commit.

## Decisao fail-closed

Prosseguir review-only: registrar fontes oficiais faltantes a partir de registries reais; nenhum pacote/coordenada/poligono/data/fonte inventado; PDF sem vetor fica bloqueado; area de risco nunca vira evento; PE3D/MDE fica contextual; geometria fora da grade nao gera patch-link. Stage seletivo somente do pacote 17C3.
