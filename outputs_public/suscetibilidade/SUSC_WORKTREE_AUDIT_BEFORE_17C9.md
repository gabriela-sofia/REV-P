# SUSC-17C9 - Auditoria de worktree antes da implementacao

- Marco: `SUSC-17C9 - Materializacao de Artefatos Fonte para Patches Candidatos`
- Categoria metodologica: `Materializacao de Insumos Reais para Extracao Multimodal`
- Branch esperada: `marco/pre-unificacao-gates-mv1`
- HEAD auditado antes da programacao: `ed8ab357b5fcc725eba7bc3b1790d979b1a254cc`
- Commit anterior: `feat: extrai features reais controladas para patches candidatos SUSC-17C8`
- Area staged no inicio: vazia
- Sujeira preexistente: preservada fora do escopo 17C9

## Escopo permitido

Criar somente artefatos review-only do SUSC-17C9 para inventariar insumos, auditar pipelines, declarar requisitos por patch candidato, planejar exports leves e registrar bloqueios metodologicos.

## Escopo bloqueado

- Nao alterar score v6.
- Nao criar score v7.
- Nao alterar datasets oficiais de patches.
- Nao criar patch oficial.
- Nao criar patch-link oficial.
- Nao promover candidato para 17B.
- Nao inventar feature real.
- Nao copiar valor de patch oficial para patch candidato.
- Nao interpolar valor de patch oficial vizinho.
- Nao usar placeholder como dado real.
- Nao usar footprint Charter como feature de suscetibilidade.
- Nao usar dado pos-evento como feature pre-evento.
- Nao executar SAR.
- Nao executar DINO/SatMAE.
- Nao baixar raster pesado.
- Nao commitar artefato bruto pesado.
- Manter `review_only=true`, `trainable=false`, `ground_truth=false`.

## Validadores pre-programacao rodados

- `python scripts/suscetibilidade/validate_susc_17c8_controlled_feature_extraction.py`: passou
- `python scripts/suscetibilidade/validate_susc_17c7_candidate_feature_extraction_plan.py`: passou
- `python scripts/suscetibilidade/validate_susc_17c6_multimodal_applicability_canary.py`: passou
- `python scripts/suscetibilidade/validate_susc_17c5_patch_grid_expansion_review.py`: passou
- `python scripts/suscetibilidade/validate_susc_17c4_official_artifact_ingestion.py`: passou
- `python scripts/suscetibilidade/validate_susc_17c3_official_source_acquisition.py`: passou
- `python scripts/suscetibilidade/validate_susc_17c2_sar_footprint_execution.py`: passou
- `python scripts/suscetibilidade/validate_susc_17c_strong_reference_acquisition.py`: passou
- `python scripts/suscetibilidade/validate_susc_17a_reference_evidence_protocol.py`: passou
- `python scripts/suscetibilidade/validate_susc_16d_calibration_candidate.py`: passou

## Insumos que devem ser lidos pelo 17C9

- Outputs 17C8 de tentativa controlada de extracao.
- Outputs 17C7 de plano de extracao e politica anti-vazamento.
- Outputs 17C6 de grade candidata, links candidatos, contrato multimodal e contrato de embedding.
- Manifests publicos/locais de features, Sentinel-2, GEE/STAC e DINO/SatMAE, apenas como referencia.

## Decisao metodologica inicial

O 17C9 nao deve produzir feature final. O pacote deve transformar o bloqueio correto do 17C8 em plano operacional verificavel, mantendo qualquer artefato ausente como bloqueio explicito.
