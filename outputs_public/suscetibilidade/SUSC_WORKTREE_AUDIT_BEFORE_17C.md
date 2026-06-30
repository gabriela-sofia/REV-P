# SUSC worktree audit before SUSC-17C

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-17C: `661fb2f feat: formaliza protocolo de evidencia observacional SUSC-17A`.

Status: auditoria pre-programacao para executar `SUSC-17C - Strong Reference Acquisition Canary`, esteira review-only para sair de "registro forte insuficiente" para "candidatos fortes datados e verificaveis", usando evento oficial/documental como ancora e Sentinel-1/SAR como caminho tecnico.

## Comandos executados

- `git branch --show-current`
- `git status --short`
- `git diff --name-only`
- `git diff --cached --name-only`
- `git ls-files --others --exclude-standard`
- `git log --oneline -1`
- inspecao das fontes 16A/16B/16C/16D/17A e dos catalogos de evento datado (13A/13B/13C, 14A, 15A/15B, 16A sentinel window plan)

## Bloqueio atual do 17B confirmado (a partir do summary 17A)

- 65 referencias patch-linked fortes.
- Apenas 2 footprints distintos.
- Apenas Curitiba.
- 0 `event_date` nas referencias fortes.

## Resultado objetivo

- Branch atual confirmada: `marco/pre-unificacao-gates-mv1`.
- Area staged antes do 17C: vazia.
- Arquivos consumidos pelo 17C modificados localmente: nenhum.
- Decisao: prosseguir com SUSC-17C sem tocar na sujeira fora do escopo.

## Entradas consumidas pelo 17C (somente leitura)

- `datasets/suscetibilidade/susc_13a_strong_observed_events_parsed_v1.csv` (taxonomia observed/risk/alert/admin; evento datado International Charter)
- `datasets/suscetibilidade/susc_13c_consolidated_observed_event_catalog_v1.csv`
- `datasets/suscetibilidade/susc_16a_sentinel_event_window_plan_v1.csv` (161 janelas Recife datadas, S1-elegiveis, pre/pos + AOI)
- `datasets/suscetibilidade/susc_16a_local_sar_flood_footprint_candidates_v1.csv` (3 candidatos SAR, BLOCKED sem raster)
- `datasets/suscetibilidade/susc_16a_observed_footprint_catalog_v1.csv`
- `datasets/external_evidence_registry.csv` (SGB/CPRM, GeoCuritiba, Defesa Civil Curitiba, ESIG/EMLURB)
- `outputs_public/suscetibilidade/susc_17a_reference_evidence_registry.csv` e summary 17A
- `outputs_public/suscetibilidade/SUSC_16B_footprint_evidence_quality_audit.csv`
- `outputs_public/suscetibilidade/susc_16d_*` (calibracao candidata review-only)
- stubs GEE/STAC existentes: `scripts/suscetibilidade/susc_16a_gee_sentinel1_flood_mapping_stub.js`, `susc_16a_stac_sentinel1_query_stub.py`

Nenhuma dessas entradas aparece em `git diff --name-only`.

## Fontes prioritarias presentes vs ausentes

Presentes em artefatos reais (serao inventariadas): Defesa Civil (Recife/Curitiba), International Charter, INMET, INEA/RJ, SGB/CPRM, GeoCuritiba/IPPUC, ESIG/EMLURB e artefatos internos 16A/16C/17A.

Ausentes nos artefatos do projeto (nao serao fabricadas; ficarao como lacuna de aquisicao no relatorio): CEMADEN, APAC Recife/PE, ANA/Hidroweb, S2iD como fonte autonoma.

## Classificacao dos arquivos sujos

### C - modificacoes tracked fora do escopo (preservadas)

11 arquivos `revp_v2e*` (docs/v2es/v2et/v2eu, execution_reports e tables). Preservados; fora do stage seletivo do 17C.

### E - untracked fora do escopo (preservados)

473 arquivos untracked de ciclos MV2, schemas `schema_mv2_*`, temporal assets e relatorios `revp_*`. Preservados.

## Nota de reprodutibilidade

O preflight 17C usa apenas validators especificos (16A/16B/16C/16D/17A), sem reexecutar pipelines que regeneram outputs 16A/16B/16C por ordenacao nao determinista. Qualquer arquivo fora do escopo que for sujado sera restaurado com `git checkout --` antes do commit.

## Decisao fail-closed

Prosseguir com SUSC-17C e stagear somente os artefatos novos do pacote SUSC-17C e a auditoria 17C. Nenhum evento, data, geometria, fonte ou coordenada sera inventado: campos ausentes recebem `not_available`/`unknown`/`insufficient`/`blocked`. Nenhum footprint vira ground truth; nenhum alerta vira ocorrencia; nenhuma area de risco vira evento ocorrido.
