# SUSC worktree audit before SUSC-17C2

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-17C2: `e74bc2d feat: prepara aquisicao forte de referencias SUSC-17C`.

Status: auditoria pre-programacao para executar `SUSC-17C2 - Sentinel-1/SAR Footprint Execution`, execucao controlada review-only dos canaries SAR priorizados pelo 17C.

## Comandos executados

- `git branch --show-current`, `git status --short`, `git diff --name-only`, `git diff --cached --name-only`, `git ls-files --others --exclude-standard`, `git log --oneline -1`
- inspecao dos artefatos 17C (priority canary, sar feasibility, sentinel1 task plan, date resolution) e 17A (registry, class policy, quality tiers)
- deteccao de runtime SAR (bibliotecas e credenciais)
- verificacao de geometria do International Charter e de bbox dos patches

## Estado objetivo

- Branch confirmada: `marco/pre-unificacao-gates-mv1`.
- Area staged antes do 17C2: vazia.
- Entradas consumidas pelo 17C2 modificadas localmente: nenhuma.

## 5 canaries confirmados (17C)

1. `S17C_E_SUSC13A_00001` - International Charter, Recife, 2022-05-24 (geometria oficial).
2. `S17C_W_S16AWIN_00003` - Recife, 2014-01-15 (SAR window).
3. `S17C_W_S16AWIN_00004` - Recife, 2014-01-16 (SAR window).
4. `S17C_W_S16AWIN_00005` - Recife, 2014-01-21 (SAR window).
5. `S17C_W_S16AWIN_00006` - Recife, 2014-01-23 (SAR window).

## Capacidade de execucao SAR

- Bibliotecas `earthengine-api` e `pystac_client` presentes no ambiente.
- Credenciais GEE/STAC AUSENTES (`EARTHENGINE_TOKEN`/`GOOGLE_APPLICATION_CREDENTIALS` vazios).
- Sem opt-in `SUSC_17C2_SAR_RUNTIME=1`.
- Conclusao: SEM runtime SAR. Os 4 canaries SAR ficam `blocked_no_runtime_access` com task specs (GEE/STAC) prontas. Nenhum footprint SAR sera fabricado.

## Caminho International Charter (geometria oficial)

- `susc_13a_strong_observed_events_parsed_v1.csv` traz bbox oficial real do evento 2022-05-24 (mapa de inundacao do International Charter, produto de sensoriamento remoto).
- Esse bbox sera materializado como 1 footprint candidato vetorial leve (sem raster, sem invencao), `evidence_class=technical_remote_sensing_flood_footprint_candidate`, `qa_status=needs_review`, `ground_truth=false`.

## Cruzamento com patches

- `susc_features_by_patch_v1.csv` tem bbox por patch (`xmin/ymin/xmax/ymax`, WGS84); 100 patches Recife.
- O bbox do International Charter (lat -8.001..-7.982) fica fora da cobertura da grade Recife (lat ate -8.013) -> 0 patches intersectados. Achado honesto: o evento esta a margem da grade atual. `candidate_patch_link_count=0`, com razao explicita.

## Sujeira fora do escopo (preservada)

- 11 tracked `revp_v2e*`; 473 untracked (MV2, `schema_mv2_*`, temporal assets, `revp_*`). Preservados.

## Nota de reprodutibilidade

Preflight usa apenas validators especificos (16D/17A/17C). Sem reexecutar pipelines que sujam outputs 16A/16B/16C. Qualquer arquivo fora do escopo sujado sera restaurado com `git checkout --` antes do commit.

## Decisao fail-closed

Prosseguir com 17C2 review-only: footprint candidato apenas de geometria oficial real ja commitada; SAR bloqueado honestamente por falta de runtime; nenhum raster bruto criado/stageado; nenhum footprint vira ground truth; nenhuma elegibilidade forte/calibracao/17B antes de QA humano. Stage seletivo somente do pacote 17C2.
