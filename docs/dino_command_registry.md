# Registro de comandos DINO Sentinel-first

Este registro lista os principais comandos para reproduzir localmente o fluxo DINO Sentinel-first do REV-P. Os outputs de execução ficam em `local_runs/` e não são destinados ao Git.

Use `--force` apenas quando substituir um diretório local de execução for intencional. Use `--resume` e `--skip-existing` para preservar execuções locais parciais.

## QA e auditorias

Executar testes:

```powershell
python -m pytest -q
```

Verificar erros de espaçamento antes do commit:

```powershell
git diff --check
```

Auditar arquivos proibidos fora de `local_runs/`:

```powershell
Get-ChildItem . -Recurse -Force |
  Where-Object {
    $_.FullName -notmatch "\\.git\\" -and
    $_.FullName -notmatch "\\local_runs\\" -and (
      $_.Name -match "cbers|CBERS" -or
      $_.Name -match "\.tif$|\.tiff$|\.zip$|\.npy$|\.npz$|\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.parquet$|\.index$" -or
      $_.FullName -match "\\data\\" -or
      $_.FullName -match "\\outputs\\" -or
      $_.FullName -match "\\patches\\" -or
      $_.FullName -match "\\archive_drive\\"
    )
  } |
  Select-Object FullName
```

## v1fw — scaffold de extração

Apenas dry-run por padrão:

```powershell
python scripts\dino\revp_v1fw_dino_embedding_extraction_scaffold.py --force
```

Com preflight local:

```powershell
python scripts\dino\revp_v1fw_dino_embedding_extraction_scaffold.py --asset-preflight local_runs\dino_asset_preflight\v1fv\dino_local_asset_preflight_v1fv.csv --force
```

## v1fx — execução smoke de embeddings

Execução explícita, limite pequeno:

```powershell
python scripts\dino\revp_v1fx_dino_smoke_embedding_execution.py --execute --limit 5 --force --allow-cpu --skip-model-if-unavailable
```

Permitir download do modelo apenas quando intencional:

```powershell
python scripts\dino\revp_v1fx_dino_smoke_embedding_execution.py --execute --limit 5 --force --allow-cpu --allow-model-download
```

## v1fy — análise exploratória do corpus

```powershell
python scripts\dino\revp_v1fy_dino_embedding_corpus_analysis.py --force
```

Flags úteis:

```powershell
python scripts\dino\revp_v1fy_dino_embedding_corpus_analysis.py --limit 5 --top-k 3 --seed 42 --force
```

## v1fz — corpus regional balanceado

Subconjunto balanceado:

```powershell
python scripts\dino\revp_v1fz_dino_balanced_embedding_corpus.py --execute --per-region-limit 2 --force --allow-cpu --allow-model-download
```

Regiões específicas:

```powershell
python scripts\dino\revp_v1fz_dino_balanced_embedding_corpus.py --execute --regions Curitiba Petropolis Recife --per-region-limit 2 --force --allow-cpu
```

## v1ga — consistência estrutural

```powershell
python scripts\dino\revp_v1ga_dino_embedding_structural_consistency_analysis.py --force
```

## v1gb — revisão visual estrutural local

```powershell
python scripts\dino\revp_v1gb_dino_embedding_local_visual_structural_review.py --force
```

## v1gc — diagnóstico geo-estrutural

```powershell
python scripts\dino\revp_v1gc_dino_embedding_geo_structural_diagnostics.py --force
```

## v1gd — robustez a perturbações

```powershell
python scripts\dino\revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py --force --allow-cpu --force-cpu --allow-model-download
```

Apenas proxy de teste offline:

```powershell
python scripts\dino\revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py --force --embedding-proxy-for-tests
```

## v1ge — corpus Sentinel expandido

Execução balanceada expandida:

```powershell
python scripts\dino\revp_v1ge_dino_expanded_sentinel_embedding_corpus.py --execute --per-region-limit 4 --batch-size 4 --force --allow-cpu --force-cpu --allow-model-download
```

Retomar execução parcial:

```powershell
python scripts\dino\revp_v1ge_dino_expanded_sentinel_embedding_corpus.py --execute --per-region-limit 4 --resume --skip-existing --allow-cpu
```

Limitar trabalho total:

```powershell
python scripts\dino\revp_v1ge_dino_expanded_sentinel_embedding_corpus.py --execute --limit 12 --batch-size 4 --force --allow-cpu
```

## v1gf — índice de evidência estrutural

```powershell
python scripts\dino\revp_v1gf_dino_structural_evidence_index.py --force
```

Usar um manifest de embedding específico:

```powershell
python scripts\dino\revp_v1gf_dino_structural_evidence_index.py --embedding-manifest local_runs\dino_embeddings\v1ge\dino_expanded_embedding_manifest_v1ge.csv --force
```

## v1gg — pacote de revisão humana

```powershell
python scripts\dino\revp_v1gg_dino_human_review_package.py --force
```

## v1gh — diagnóstico estrutural longitudinal

```powershell
python scripts\dino\revp_v1gh_dino_longitudinal_structural_diagnostics.py --force
```

## v1gi — rastreador de proveniência estrutural

```powershell
python scripts\dino\revp_v1gi_dino_structural_provenance_tracker.py --force
```

## v1gj — auditoria de prontidão multimodal

Auditoria apenas. Não habilita execução multimodal.

```powershell
python scripts\dino\revp_v1gj_multimodal_readiness_audit.py --force
```

Guardrails obrigatórios:

- `multimodal_execution_enabled=false`
- `multimodal_training_enabled=false`
- `multimodal_hold=true`

## v1gk — auditoria de reprodutibilidade

```powershell
python scripts\dino\revp_v1gk_dino_pipeline_reproducibility_audit.py --force
```

## v1gn — monitor de saúde da execução

Executar após limpeza local, retomada ou execução expandida de embeddings:

```powershell
python scripts\dino\revp_v1gn_dino_execution_health_monitor.py --force
```

O monitor reporta `HEALTHY`, `WARNING` ou `DEGRADED`. Verifica disponibilidade local de embeddings, integridade de manifests, embeddings corrompidos, divergência de dimensões, presença de outputs upstream, hashes estruturais duplicados e guardrails de execução local.

## v1go — orquestrador leve

Validar um estágio sem execução:

```powershell
python scripts\dino\revp_v1go_dino_pipeline_orchestrator.py --stage v1ga --validate-only
```

Pré-visualizar o comando de um estágio sem executar:

```powershell
python scripts\dino\revp_v1go_dino_pipeline_orchestrator.py --stage v1fx --dry-run
```

Validar o pipeline completo registrado:

```powershell
python scripts\dino\revp_v1go_dino_pipeline_orchestrator.py --stage all --validate-only
```

Guardrails operacionais:

- validar antes de executar;
- inspecionar os comandos do dry-run antes de estágios pesados;
- nunca ativar multimodal pelo orquestrador;
- manter `local_runs/` isolado;
- reexecutar v1gn após qualquer execução local retomada de embeddings.

## v1gp — auditoria de prontidão para release no GitHub

Executar auditoria completa de prontidão antes de considerar um commit ou PR:

```powershell
python scripts\dino\revp_v1gp_dino_github_release_readiness_audit.py --force
```

A auditoria verifica:

- artefatos proibidos (`.npz`, `.npy`, `.tif`, `.tiff`, `.vrt`, `.aux.xml`) fora de `local_runs/`
- caminhos absolutos privados em arquivos versionáveis
- proteções metodológicas (`review_only`, `supervised_training`, `labels_created`, `predictive_claims`, `multimodal`)
- existência de documentação obrigatória e cobertura no README
- cobertura de scripts e testes para blocos recentes (v1gn, v1go, v1gp)
- completude do registro de comandos

Produz um status de prontidão: `READY_FOR_LOCAL_COMMIT`, `READY_WITH_REVIEW_NOTES` ou `BLOCKED`.

Outputs: `local_runs/dino_embeddings/v1gp/`

## v1gq — baseline de vulnerabilidade multicritério GIS

Modo apenas auditoria (sem GIS root — todos os indicadores BLOCKED):

```powershell
python scripts\dino\revp_v1gq_gis_multicriteria_vulnerability_baseline.py --force
```

Com GIS data root (habilita cálculo de indicadores):

```powershell
python scripts\dino\revp_v1gq_gis_multicriteria_vulnerability_baseline.py --gis-root <path_to_gis_root> --force
```

O script audita a prontidão e computa um baseline de vulnerabilidade multicritério para os 12 patches Sentinel do v1ge usando quatro indicadores: distância ao rio, uso do solo, densidade populacional e densidade viária. O índice de vulnerabilidade é apenas um proxy estrutural e interpretável — não é verdade de campo, não é rótulo, não é alvo supervisionado.

Indicadores disponíveis (com dados GIS locais): distância ao rio (todas as regiões), densidade viária (Recife apenas).

Indicadores bloqueados: uso do solo (fiona não instalado; shapefiles existem apenas para Petrópolis), densidade populacional (sem dados censitários encontrados).

O índice é PARTIAL (2/4 indicadores) para patches de Recife e BLOCKED (1/4 indicadores) para Curitiba e Petrópolis.

Guardrails obrigatórios:

- `vulnerability_index_is_ground_truth=false`
- `labels_created=false`
- `supervised_training=false`
- `predictive_claims=false`
- `multimodal_execution_enabled=false`

Outputs: `local_runs/dino_embeddings/v1gq/`

## v1gr — auditoria de prontidão e conversão de uso do solo GIS

Modo apenas auditoria (sem GIS root — escaneia apenas REV-P):

```powershell
python scripts\dino\revp_v1gr_gis_land_use_readiness_and_conversion_audit.py --force
```

Com GIS data root (habilita inventário de uso do solo do PROJETO e tentativa de conversão):

```powershell
python scripts\dino\revp_v1gr_gis_land_use_readiness_and_conversion_audit.py --gis-root <path_to_gis_root> --force
```

O script inventaria arquivos de uso do solo, audita dependências GIS (fiona, geopandas, pyogrio), lê a tabela de atributos DBF do FBDS sem fiona, constrói um mapeamento candidato de classe para pontuação e avalia a viabilidade de conversão por região. Não modifica outputs do v1gq e não cria rótulos.

Resultados atuais (com gis-root do PROJETO): Petrópolis PARTIAL (DBF legível, conversão de geometria bloqueada — fiona ausente), Curitiba BLOCKED, Recife BLOCKED. Prontidão para reexecução do v1gq: PARTIAL_READY.

Guardrails obrigatórios:

- `land_use_is_ground_truth=false`
- `labels_created=false`
- `supervised_training=false`
- `predictive_claims=false`
- `multimodal_execution_enabled=false`

Outputs: `local_runs/dino_embeddings/v1gr/`

## v1gs — habilitação de geometria de uso do solo GIS e reexecução parcial de v1gq

Auditar e converter a geometria de uso do solo de Petrópolis, depois reexecutar v1gq com o GeoJSON WGS84 convertido:

```powershell
python scripts\dino\revp_v1gs_gis_land_use_geometry_enablement.py --gis-root <path_to_gis_root> --force
```

Sem `--gis-root` (apenas auditoria, toda geometria BLOCKED):

```powershell
python scripts\dino\revp_v1gs_gis_land_use_geometry_enablement.py --force
```

O script lê o shapefile FBDS de Petrópolis usando pyogrio → geopandas → fiona (nessa ordem), reprojeta para WGS84, salva em `local_runs/dino_embeddings/v1gs/converted/petropolis_land_use_v1gs.geojson` e reexecuta v1gq com `--land-use-geojson-petropolis` escrevendo em `local_runs/dino_embeddings/v1gq_rerun_v1gs/`.

Resultado atual: geometria AVAILABLE (pyogrio), 6861 feições, 6 classes. No entanto, os centroides dos patches Sentinel de Petrópolis ficam ~2–3 km ao sul do limite de cobertura do FBDS — o indicador land_use permanece BLOCKED para todos os patches de Petrópolis (lacuna real de cobertura de dados, não um erro de processamento).

Guardrails obrigatórios:

- `land_use_is_ground_truth=false`
- `labels_created=false`
- `supervised_training=false`
- `predictive_claims=false`
- `multimodal_execution_enabled=false`

Outputs: `local_runs/dino_embeddings/v1gs/` e (se reexecução realizada) `local_runs/dino_embeddings/v1gq_rerun_v1gs/`

## v1gt — auditoria de expansão de cobertura de uso do solo GIS

Auditoria com escopo dino-corpus (12 patches do v1ge):

```powershell
python scripts\dino\revp_v1gt_gis_land_use_coverage_expansion_audit.py --gis-root <path_to_gis_root> --patch-scope dino-corpus --force
```

Auditoria com escopo full-manifest (128 patches do v1fu):

```powershell
python scripts\dino\revp_v1gt_gis_land_use_coverage_expansion_audit.py --gis-root <path_to_gis_root> --patch-scope full-manifest --force
```

Modo apenas auditoria (sem GIS root — limites dos TIFs não resolvidos):

```powershell
python scripts\dino\revp_v1gt_gis_land_use_coverage_expansion_audit.py --force
```

O script avalia a cobertura de fontes de uso do solo para cada patch usando o GeoJSON WGS84 do FBDS de Petrópolis gerado pelo v1gs. Status de cobertura: `COVERED`, `BBOX_OVERLAP_NO_CENTROID`, `UNCOVERED`, `NO_TIF`.

Resultados atuais (dino-corpus, com gis-root do PROJETO): 0 COVERED, 2 apenas BBOX (patches de Petrópolis com sobreposição de bbox mas centroides ~2–3 km ao sul do limite do FBDS), 10 UNCOVERED (Curitiba + Recife sem fonte de uso do solo). Status geral: `BBOX_PARTIAL`.

Resultados atuais (full-manifest, 128 patches): 33 COVERED, 13 apenas BBOX, 82 UNCOVERED. Status geral: `PARTIAL`.

Cinco candidatos de expansão documentados (MapBiomas para todas as regiões, FBDS estendido para Petrópolis, grade IBGE LULC) — todos `NOT_ACQUIRED`.

Guardrails obrigatórios:

- `land_use_is_ground_truth=false`
- `labels_created=false`
- `supervised_training=false`
- `predictive_claims=false`
- `multimodal_execution_enabled=false`

Outputs: `local_runs/dino_embeddings/v1gt/`

## Preparação do commit

Antes do commit final, executar:

```powershell
python -m pytest -q
git diff --check
git status --short
```

Não adicionar ao stage nem commitar `local_runs/`, `.npz`, `.npy`, rasters, checkpoints ou outputs PNG de revisão local.
