# Auditoria do worktree antes do SUSC-17C6

Data local: 2026-06-30.

Branch: `marco/pre-unificacao-gates-mv1`.

HEAD observado: `8b07efab2eccd21d84e838dbd9dd80ec51476b3a` (`feat: audita expansao de grade patch SUSC-17C5`).

Area staged antes do marco: vazia.

## Escopo do marco

O marco `SUSC-17C6 - Canario de Aplicabilidade Multimodal Review-Only` deve criar apenas artefatos candidatos, publicos e de revisao. Ele nao altera a grade oficial SUSC, nao cria patch oficial, nao cria vinculo oficial patch-evento, nao altera score v6, nao cria score v7, nao cria treino, modelo, label ou ground truth.

## Estado cientifico preservado

- Charter 758 existe como geometria oficial candidata observacional real para `REC_2022_05_24_30`.
- A geometria cai fora da grade SUSC Recife atual.
- A intersecao com a grade SUSC atual e zero.
- A distancia ao patch SUSC mais proximo registrada no SUSC-17C5 e `1398.4 m`.
- `REC_00019` pertence ao namespace historico do Protocolo C e nao e patch SUSC.
- O SUSC-17B continua bloqueado.
- Os pacotes formais P0 `PKG_FR_PET_001` e `PKG_FR_REC_002` seguem ausentes.
- Runtime SAR segue indisponivel.

## Sujeira preexistente

Antes de editar o SUSC-17C6, o worktree ja continha alteracoes fora deste escopo. A auditoria local observou 11 arquivos tracked modificados e centenas de arquivos untracked. Essas alteracoes devem ser preservadas e nao devem ser revertidas por este marco.

## Validadores executados antes do 17C6

Comando executado:

```powershell
python scripts\suscetibilidade\validate_susc_17c5_patch_grid_expansion_review.py
python scripts\suscetibilidade\validate_susc_17c4_official_artifact_ingestion.py
python scripts\suscetibilidade\validate_susc_17c3_official_source_acquisition.py
python scripts\suscetibilidade\validate_susc_17c2_sar_footprint_execution.py
python scripts\suscetibilidade\validate_susc_17c_strong_reference_acquisition.py
python scripts\suscetibilidade\validate_susc_17a_reference_evidence_protocol.py
python scripts\suscetibilidade\validate_susc_16d_calibration_candidate.py
```

Resultado:

- `PASSED: SUSC-17C5 validations passed`
- `PASSED: SUSC-17C4 validations passed`
- `PASSED: SUSC-17C3 validations passed`
- `PASSED: SUSC-17C2 validations passed`
- `PASSED: SUSC-17C validations passed`
- `PASSED: SUSC-17A validations passed`
- `PASSED: SUSC-16D validations passed`

## Insumos 17C5/17C4 lidos

- `outputs_public/suscetibilidade/susc_17c5_patch_grid_inventory.csv`
- `outputs_public/suscetibilidade/susc_17c5_charter758_grid_coverage_audit.csv`
- `outputs_public/suscetibilidade/susc_17c5_candidate_expansion_aoi.geojson`
- `outputs_public/suscetibilidade/susc_17c5_patch_grid_expansion_options.csv`
- `outputs_public/suscetibilidade/susc_17c5_expansion_risk_policy.json`
- `outputs_public/suscetibilidade/susc_17c5_summary.json`
- `outputs_public/suscetibilidade/susc_17c4_candidate_geometries.geojson`
- `outputs_public/suscetibilidade/susc_17c4_extracted_reference_candidates.csv`

## Observacao de escopo

Qualquer stage, commit ou push deve ser seletivo. Este marco nao autoriza `git add -A` nem push.
