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

## v1gg — pacote de revisão supervisora

```powershell
python scripts\dino\revp_v1gg_dino_review_gate_package.py --force
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

> Nota: entre v1gu e v2bm o repositório avançou em várias trilhas (evidência
> estrutural, pacotes de revisão, exportação TCC, Protocolo C, geometria
> evento-patch). Este registro de comandos não cataloga cada estágio
> intermediário; o índice canônico de versões está nos próprios scripts e
> testes. As entradas v2bn/v2bo abaixo retomam a trilha multimodal/ground
> truth no próximo identificador livre.

## v2bn — feature table multimodal (review-only readiness)

Construir uma feature table compacta e auditável, uma linha por entrada
Sentinel, unindo o spine canônico (v1fu) com o manifesto real de embeddings
DINOv2 (v1ge) e flags de disponibilidade (GIS, evidência, binding). Não cria
label, não cria negativo, não habilita treino. Embeddings densos são
referenciados por URI/hash/dim — nunca copiados para CSV.

```powershell
python scripts\multimodal\revp_v2bn_multimodal_feature_table_builder.py --force
```

Usar um manifesto de embedding específico:

```powershell
python scripts\multimodal\revp_v2bn_multimodal_feature_table_builder.py --embedding-manifest local_runs\dino_embeddings\v1ge\dino_expanded_embedding_manifest_v1ge.csv --force
```

Reconhecer escrita em `local_runs/` explicitamente (comportamento padrão):

```powershell
python scripts\multimodal\revp_v2bn_multimodal_feature_table_builder.py --allow-local-runs --force
```

Estado atual (inputs reais): 128 linhas, 12 com embedding real 768D, 12
review-eligible, 0 allowed_for_training. A divergência histórica "0 vs 12
embeddings" é reconciliada como `HISTORICAL_STALE_ZERO_EMBEDDINGS` (registry
fail-closed v1ph, 0 vetores densos parseados) vs `LOCAL_MANIFEST_AVAILABLE` /
`PUBLIC_FINAL_REPORT_ONLY` (12 embeddings reais), sem editar artefatos antigos.

Guardrails obrigatórios:

- `labels_created=false`
- `formal_negative_count=0`
- `supervised_training_enabled=false`
- `multimodal_execution_enabled=false`
- `multimodal_training_enabled=false`
- DINOv2 frozen, sem fine-tuning, sem early/pixel fusion

Outputs: `local_runs/multimodal/v2bn/`

## v2bo — scaffold de ground truth e training gate (sem criar labels)

Preparar o protocolo de label e de negativos sem produzir nenhum label. Emite
o registro scaffold de patches (colunas de label vazias/NA), a política de
label, a política de negativos (ausência não é negativo) e o training gate
bloqueado.

```powershell
python scripts\multimodal\revp_v2bo_ground_truth_training_gate_scaffold.py --force
```

Apontar para a feature table v2bn explicitamente:

```powershell
python scripts\multimodal\revp_v2bo_ground_truth_training_gate_scaffold.py --feature-table local_runs\multimodal\v2bn\multimodal_feature_table_core_v2bn.csv --force
```

Política de negativos (explícita): absence of evidence is not negative;
pseudo-absence is not formal negative; random background is not formal
negative; distance from anchor is not formal negative; matched negatives só com
critério formal e evidência comparável; unknown stays unknown.

Quando o gate for liberado por ground truth auditável, os primeiros modelos
devem ser baselines leves sobre embeddings congelados (Logistic Regression,
Random Forest, HistGradientBoosting/XGBoost, MLP raso), validados com
grupos/blocks — nunca random split simples.

Guardrails obrigatórios:

- `labels_created=false`
- `formal_negative_count=0`
- `supervised_training=false`
- `multimodal_execution_enabled=false`

Outputs: `local_runs/ground_truth/v2bo/`

## v2bp — adjudicação autônoma de evidência e auditoria de consistência evento-patch

Reinterpreta todo "human review required" como **auditoria autônoma estruturada**:
lê os artefatos existentes (registry de pacotes evento-patch do v2at, catálogo de
fontes, overlays, feature table v2bn, scaffold v2bo) e decide por patch/evento,
com regras explícitas, se a evidência é consistente, contraditória, circular,
insuficiente ou genuinamente ambígua. Não cria label, não cria negativo, não
libera treino.

```powershell
python scripts\multimodal\revp_v2bp_autonomous_evidence_adjudication.py --force
```

Apontar registry de pacotes específico:

```powershell
python scripts\multimodal\revp_v2bp_autonomous_evidence_adjudication.py --package-registry datasets\v2at_event_patch_package_registry.csv --force
```

Estado atual (172 pacotes reais do v2at): **1 auto-rejeitado** (evento/patch
UNKNOWN), **55 candidate-positive** auto-validados (READY_FOR_GT, held for
overlay — Recife REC_2022_05_24_30), **116 blocked** (secondary, falta overlay
patch-evento), **0 needs_user_decision**. Região comparada com acento
normalizado (`Petropolis` = `Petrópolis`). Candidate-positive **não é label**:
`gt_patch_flood_observed=NA`, `allowed_for_training=False`.

Guardrails obrigatórios:

- `labels_created=false`
- `formal_negatives_created=false`
- `allowed_for_training_count=0`
- `promotion_to_operational_gt=false`
- `candidate_positive_is_not_label=true`
- `multimodal_execution_enabled=false`

Outputs: `local_runs/ground_truth/v2bp/`

## v2bq — resolver de geometria de overlay patch-evento

Ataca o blocker técnico `NO_PATCH_EVENT_OVERLAY_GEOMETRY` deixado pelo v2bp para
os 55 candidate-positives: descobre dinamicamente geometria real no repo,
normaliza CRS para EPSG:4326, valida, e **calcula a interseção patch-evento de
verdade** (shapely quando disponível; fail-closed se ausente). Não inventa
geometria, não cria label, não libera treino. Centroides/pontos são suporte
fraco (nunca viram overlay). CRS desconhecido bloqueia. `NEEDS_USER_DECISION` só
para ambiguidade geométrica real (múltiplas geometrias conflitantes).

```powershell
python scripts\multimodal\revp_v2bq_patch_event_overlay_geometry_resolver.py --force
```

Estado atual (dados reais): 55 candidate-positives, 103 fontes GeoJSON
descobertas, 1 patch polygon real (`REC_00019`) e 1 event polygon real
(`REC_2022_05_24_30`, charter758 digitized candidate, `provided_unreviewed`).
Resultado: **1 overlay computado → `OVERLAY_REJECT_NO_INTERSECTION`** (patch e
evento não se sobrepõem no espaço — achado geométrico real), **54
`OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING`**, 0 resolvidos, 0 needs_user. Overlay
resolvido move para `READY_FOR_FORMAL_GT_PROTOCOL` — **não** é label e **não**
libera treino.

Guardrails obrigatórios:

- `labels_created=false`
- `allowed_for_training_count=0`
- `geometry_invented=false`
- `centroid_promoted_to_overlay=false`
- `overlay_equals_label=false`
- `promotion_to_operational_gt=false`

Outputs: `local_runs/ground_truth/v2bq/` (+ sidecars GeoJSON em
`local_runs/ground_truth/v2bq/geometries/` quando derivados)

## v2br — reconciliação geométrica e recuperação de boundaries

Duas frentes autônomas que preparam nova rodada de overlay após o v2bq.
**Frente A:** audita a não-interseção do `REC_00019` (qualidade de CRS, bbox,
centroide, área, axis-order, lineage; distância patch-evento; cross-check com
pontos Defesa Civil; teste de hipóteses de erro) e decide se é robusta ou se
deve ser segurada. **Frente B:** recupera boundaries dos candidatos bloqueados a
partir de bounds de header de raster gravados (v1fs), reprojetando para WGS84.
Não inventa geometria; centroides/pontos nunca viram boundary; não-interseção
nunca vira negativo; treino segue bloqueado.

```powershell
python scripts\multimodal\revp_v2br_geometry_reconciliation_and_boundary_recovery.py --force
```

Estado atual (dados reais): **REC_00019 → `NON_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED`**
(MEDIUM; ~26.3 km de separação; mesmo CRS EPSG:32725→4326 sem axis-order risk;
evento charter758 `provided_unreviewed`/`can_be_ground_truth=false`; pontos
Defesa Civil não alinham com patch nem evento → segurar, não descartar);
candidate_positive_status = `HELD_FOR_GEOMETRY_RECONCILIATION`. **Frente B: 36
boundaries recuperadas** (bounds EPSG:32725 do v1fs reprojetados→WGS84, sidecars
em `recovered_patch_boundaries/`), **18 `PATCH_BOUNDARY_NOT_FOUND`** (short-ids
sem bounds), 36 podem reexecutar overlay, 0 needs_user.

Guardrails obrigatórios:

- `labels_created=false`
- `allowed_for_training_count=0`
- `negative_from_non_intersection=false`
- `geometry_invented=false`
- `centroid_promoted_to_boundary=false`
- `event_polygon_promoted_to_gt=false`

Outputs: `local_runs/ground_truth/v2br/` (+ `recovered_patch_boundaries/`)

## v2bs — overlay retry nos boundaries recuperados + confiabilidade do evento

Reexecuta o overlay usando as 36 boundaries recuperadas pelo v2br (mais
`REC_00019` held/diagnóstico) contra o polígono disponível do evento
`REC_2022_05_24_30`, com uma camada metodológica extra: **classifica a
confiabilidade do polígono do evento antes de promover qualquer caso a protocolo
formal de GT**. Calcula interseção real (shapely), mas overlay positivo **não**
vira label e não-interseção **não** vira negativo enquanto o evento for
`provided_unreviewed`/`can_be_ground_truth=false`.

```powershell
python scripts\multimodal\revp_v2bs_recovered_boundary_overlay_retry.py --force
```

Estado atual (dados reais): **37 reprocessados** (36 recuperados + REC_00019),
**0 intersectam** o polígono do evento, **37 `OVERLAY_NO_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED`**
(held, não-negativos), 0 blocked, 0 needs_user. Confiabilidade do evento:
**`EVENT_GEOMETRY_RELIABILITY_BLOCKED_CONFLICTS_WITH_DEFENSE_CIVIL_POINTS`**
(400 pontos Defesa Civil, 0 dentro do polígono, mais próximo ~13 km →
`POINT_SUPPORT_CONFLICTING`), `gt_promotion_allowed=false`,
`recommended_use=USE_FOR_OVERLAY_QA_ONLY`. 0 prontos para GT formal, 0 que
entrariam se o evento fosse confiável (nenhum intersecta).

Guardrails obrigatórios:

- `labels_created=false`
- `allowed_for_training_count=0`
- `positive_label_from_overlay=false`
- `negative_label_from_non_intersection=false`
- `event_polygon_promoted_to_gt=false`
- `defense_civil_points_promoted_to_polygon=false`

Outputs: `local_runs/ground_truth/v2bs/`

## v2bt — geometrias alternativas de evento + resolução de confiabilidade

Resolve o blocker `EVENT_GEOMETRY_RELIABILITY_BLOCKED_CONFLICTS_WITH_DEFENSE_CIVIL_POINTS`
do v2bs: audita os 400 pontos Defesa Civil, decide o destino do polígono
charter758 e constrói **geometrias alternativas QA-only** a partir dos pontos
(convex hull, buffered point union, envelopes de cluster DBSCAN). Ponto, hull,
buffer e cluster são `POINT_DERIVED_EVENT_GEOMETRY_CANDIDATE` — **nunca** ground
truth, label ou negativo. Não inventa geometria.

```powershell
python scripts\multimodal\revp_v2bt_event_geometry_alternative_resolver.py --force
```

Estado atual (dados reais): 400 pontos auditados; **charter →
`CHARTER_POLYGON_REJECTED_FOR_EVENT_QA`** (0 pontos dentro, mais próximo ~13 km;
não invalida o evento histórico, só a geometria); **5 geometrias alternativas
QA-only criadas/prontas** (1 convex_hull, 2 buffer_union [250m/500m], 2
cluster_envelope via DBSCAN eps=1000m), todas score HIGH/MEDIUM, todas
`can_use_for_formal_gt=false`/`can_create_label=false`; fila de retry com escopo
`37_RETRIED_PATCHES`; 0 needs_user; 0 fontes oficiais externas adquiridas.

Backends: shapely, pyproj, sklearn (DBSCAN) — fail-closed se ausentes
(`GEOMETRY_BACKEND_UNAVAILABLE`).

Guardrails obrigatórios:

- `labels_created=false`
- `allowed_for_training=false`
- `points_not_promoted_to_gt`
- `point_hull_not_promoted_to_gt`
- `buffer_union_not_promoted_to_gt`
- `charter_polygon_not_promoted_to_gt`
- `no_geometry_invented`

Outputs: `local_runs/ground_truth/v2bt/` (+ `alternative_event_geometries/`)

## v2bu — auditoria de sensibilidade de overlay com geometrias alternativas

Reexecuta o overlay dos patches com boundary disponível (escopo
`37_RETRIED_PATCHES` = 36 boundaries recuperadas + REC_00019) contra as 5
geometrias alternativas QA-only do v2bt, gerando uma **matriz de sensibilidade
geométrica**. A pergunta não é "quais patches são positivos", mas "quais patches
mostram compatibilidade geométrica robusta com a geometria QA-only do evento sob
diferentes métodos de reconstrução". Interseção QA **não** é label; não-interseção
**não** é negativo; nada libera treino.

```powershell
python scripts\multimodal\revp_v2bu_alternative_event_overlay_sensitivity.py --force
```

Estado atual (dados reais): 37 patches × 5 alternativas = **185 overlays
pairwise**; **1 `QA_COMPATIBLE_ROBUST`** (REC_00276, intersecta 4 métodos
incl. buffer_250/cluster/hull, ratio 0.88, `ready_for_formal_gt_review=True`),
**1 `QA_COMPATIBLE_METHOD_DEPENDENT`** (REC_00299, só hull+buffer_500
permissivos, ratio 0.33), **0 buffer-only**, **35 `QA_NOT_COMPATIBLE`**, 0
needs_user. `gt_patch_flood_observed=NA` e `allowed_for_training=False` em
todos, mesmo no caso ready-for-review.

Critérios: `QA_COMPATIBLE_ROBUST` = ≥3 alternativas, ≥2 famílias de método
(hull/buffer/cluster), com geometria "tight" (buffer≤250 ou cluster);
`METHOD_DEPENDENT` = intersecta sem consenso robusto; `BUFFER_ONLY` = só buffers;
`NOT_COMPATIBLE` = nenhuma interseção.

Guardrails obrigatórios:

- `labels_created=false`
- `allowed_for_training=false`
- `no_positive_label_from_qa_overlay`
- `no_negative_label_from_no_intersection`
- `alternative_geometry_not_promoted_to_gt`
- `ready_for_formal_review_not_training_ready`

Outputs: `local_runs/ground_truth/v2bu/`

## v2bv — dossiê formal QA + scaffold de protocolo de negativos comparáveis

Transforma o resultado de sensibilidade do v2bu numa camada formal de decisão
metodológica, **sem cruzar a fronteira de label**. Frente A: consolida o patch
robusto (REC_00276) num `FORMAL_QA_POSITIVE_CANDIDATE_DOSSIER`
(`STRONG_QA_POSITIVE_CANDIDATE_HELD_FOR_FORMAL_FOOTPRINT_VALIDATION`). Frente B:
registra o method-dependent (REC_00299) separado
(`METHOD_DEPENDENT_HELD_FOR_TIGHTER_EVENT_GEOMETRY`). Frente C: scaffold de
negativos comparáveis a partir dos noncompatible, sob critérios rígidos, **sem
criar negativo** (ausência/não-compatibilidade não são negativos). Frente D:
gate formal de GT + gap analysis dizendo o que falta.

```powershell
python scripts\multimodal\revp_v2bv_formal_qa_dossier_and_negative_protocol.py --force
```

Estado atual (dados reais): **1 dossiê positivo forte** (REC_00276, ratio 0.88,
`formal_gt_ready=false`/`gt=NA`/`train=false`), **1 method-dependent** (REC_00299),
**35 negativos scaffoldados** → 14 `COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY` (≤8km
do footprint QA) + 21 `NOT_COMPARABLE_DISTANCE_TOO_FAR` (>8km); `formal_protocol_exists=false`
→ nenhum negativo criável; 7 gaps de GT (footprint formal, protocolo positivo,
protocolo de negativos, blocking espacial, anti-leakage, target). `can_train_supervised_model=false`.

Guardrails obrigatórios:

- `labels_created=false`
- `positive_candidate_not_promoted_to_label`
- `negative_candidate_not_promoted_to_label`
- `no_negative_from_absence`
- `no_negative_from_noncompatibility`
- `formal_gt_ready_false`
- `training_still_blocked`

Outputs: `local_runs/ground_truth/v2bv/`

## v2bw — validação de footprint oficial do evento + reconciliação de fontes

Etapa de **assembly + reconciliação + gate**, offline-determinística, **sem
download externo**. Prova formalmente que o gargalo de GT do evento
`REC_2022_05_24_30` (cheias do Recife, mai/2022) é a **ausência de footprint
oficial poligonal revisado**, não falta de cálculo. Inventaria e classifica
todas as fontes locais (contexto oficial Charter 758, evidência pontual Defesa
Civil, polígono charter media-derived rejeitado, geometrias QA-derived do v2bt),
parsa as geometrias leves, reconcilia charter vs pontos vs QA vs REC_00276 vs
REC_00299 vs negativos comparáveis e emite a decisão `OFFICIAL_FOOTPRINT_NOT_FOUND`.
Sem web: registra `EXTERNAL_WEB_SEARCH_NOT_PERFORMED` e segue. **Não cria label,
não cria negativo formal, não libera treino.**

```powershell
python scripts\multimodal\revp_v2bw_official_event_footprint_validation.py --force
```

Estado atual (dados reais): **69 fontes** descobertas (31 oficiais), **0 fontes
com geometria oficial poligonal**, **6 candidatos de geometria** (1 media-derived
charter rejeitado + 5 QA-derived). Decisão: `OFFICIAL_FOOTPRINT_NOT_FOUND`
(detalhe `OFFICIAL_FOOTPRINT_NOT_FOUND_BUT_POINT_DERIVED_QA_GEOMETRY_AVAILABLE`).
Charter: `CHARTER_POLYGON_REJECTED_FOR_EVENT_QA`. REC_00276:
`ALIGNED_WITH_QA_ONLY_GEOMETRY_NO_OFFICIAL_FOOTPRINT` (dossiê forte mantido, não
label). **14 negativos comparáveis** reavaliados, todos `formal_negative_label_created=false`.
`allowed_for_training_count=0`, `supervised_training_enabled=false`.

Guardrails obrigatórios (todos `PASS`/`BLOCKED_EXPECTED`):

- `labels_created_false`
- `formal_positive_not_created`
- `formal_negative_not_created`
- `no_label_from_official_candidate`
- `no_negative_from_non_intersection`
- `no_negative_from_absence`
- `qa_geometry_not_promoted_to_gt`
- `charter_polygon_not_repromoted`
- `official_source_not_label_by_itself`
- `allowed_for_training_false`
- `training_still_blocked`
- `no_geometry_invented`
- `no_heavy_outputs`
- `private_absolute_paths_removed`

Outputs: `local_runs/ground_truth/v2bw/`

## v2bx — protocolo formal de GT em modo dry-run + auditoria de prontidão anti-leakage

Como o v2bw provou que **não existe footprint oficial poligonal revisado**, o v2bx
para de procurá-lo e modela um **protocolo formal em modo dry-run**: o que
aconteceria se o projeto declarasse explicitamente a geometria QA-derived dos
pontos Defesa Civil como referência operacional provisória. Modela **duas
trilhas**: Trilha A (oficial estrita) → `BLOCKED_OFFICIAL_FOOTPRINT_NOT_FOUND`;
Trilha B (referência QA-derived declarada) → `PROTOCOL_DRY_RUN_ONLY_QA_DERIVED_REFERENCE`.
Deriva candidatos positivos/negativos em preview, registra
`would_label_if_protocol_approved`, planeja split anti-leakage e audita os gates
que ainda bloqueiam labels reais. **Nunca cria label** (`label_created=false`),
`gt_patch_flood_observed` fica `NA` em tudo, `allowed_for_training=false` em tudo.
Offline-determinístico.

```powershell
python scripts\multimodal\revp_v2bx_formal_gt_protocol_dry_run.py --force
```

Estado atual (dados reais): **1 dry-run positive** (REC_00276, would_be_positive=true,
gt=NA, train=false, `PROTOCOL_DRY_RUN_ONLY_OFFICIAL_FOOTPRINT_NOT_FOUND`),
**1 method-dependent held** (REC_00299, não promovido), **14 dry-run negatives**
(comparáveis QA-only ≤8km, `formal_negative_label_created=false`) + **21 excluídos**
(too-far). **0 conflitos** positivo/negativo. Split plan: 37 linhas todas
`SPLIT_BLOCKED_TOO_FEW_POSITIVES` (1 positivo não é treinável), agrupamento por
evento/região/spatial block/source family (sem random split). `label_creation_allowed=false`,
`can_train_supervised_model=false`, `can_train_dry_run_model=false`.

Guardrails obrigatórios (todos `PASS`):

- `labels_created_false`
- `formal_positive_not_created`
- `formal_negative_not_created`
- `dry_run_candidate_not_label`
- `gt_patch_flood_observed_all_na`
- `allowed_for_training_false`
- `no_negative_from_absence`
- `no_negative_from_noncompatibility_without_protocol`
- `method_dependent_not_promoted`
- `official_footprint_missing_blocks_formal_gt`
- `qa_geometry_not_promoted_to_gt`
- `dry_run_split_not_training_ready`
- `training_still_blocked`
- `no_heavy_outputs`
- `private_absolute_paths_removed`

Outputs: `local_runs/ground_truth/v2bx/`

## v2by — planejador de expansão de coorte + descoberta de candidatos dry-run

O v2bx produziu um protocolo dry-run correto mas estatisticamente insuficiente
(1 positivo). O gargalo deixou de ser "como montar o protocolo" e virou
`TOO_FEW_POSITIVES_FOR_ANY_TRAINING_OR_EVALUATION`. O v2by **não treina e não cria
label**: escaneia todo o universo de eventos/patches do repo e planeja onde a
cadeia v2bp→v2bx poderia ser repetida para construir massa crítica. Para cada
evento/patch audita os sinais (contexto oficial, evidência pontual, geometria
poligonal, geometria QA-derivável, boundary de patch, embedding DINO, GIS),
classifica readiness, monta fila priorizada e projeta — conservadoramente, sem
inventar números — o yield potencial. Offline-determinístico.

```powershell
python scripts\multimodal\revp_v2by_cohort_expansion_candidate_discovery.py --force
```

Estado atual (dados reais): **4 eventos candidatos / 114 patches**. Distribuição:
**1 já processado** (REC_2022_05_24_30, urban_flood, pontos+polígono+QA+36
boundaries), **2 LOW** (PET_2022_02_15 e PET_2024_03_21_28, mass_movement,
contexto oficial mas SEM geometria/pontos locais → `EXPANSION_EVENT_CONTEXT_ONLY`),
**1 BLOCKED** (Curitiba, registry ausente/rejeitado → `EXPANSION_EVENT_BLOCKED_SOURCE_INSUFFICIENT`).
**0 HIGH / 0 MEDIUM**. Geometria/pontos só existem para Recife → expansão
majoritariamente bloqueada, gargalo = aquisição de geometria/pontos para os
outros eventos. Yield: REC `NO_CHANGE` (já contado), demais `NOT_ESTIMABLE`/`BLOCKED`.
Fila: 3 eventos, `needs_user_decision=false` (blockers técnicos). Treino bloqueado
até a coorte alcançar ≥10 positivos (heurística conservadora de planejamento, não
limiar estatístico validado).

Guardrails obrigatórios (todos `PASS`):

- `labels_created_false`
- `formal_positive_not_created`
- `formal_negative_not_created`
- `dry_run_projection_not_label`
- `no_negative_from_absence`
- `no_random_background_negative`
- `method_dependent_not_promoted`
- `expansion_queue_not_training_ready`
- `allowed_for_training_false`
- `training_still_blocked`
- `no_geometry_invented`
- `no_heavy_outputs`
- `private_absolute_paths_removed`

Outputs: `local_runs/ground_truth/v2by/`

## v2bz — aquisição de evidência de expansão + resolvedor de escopo de hazard

O v2by mostrou que a coorte está travada em 1 positivo porque os eventos
não-Recife não têm geometria/pontos locais. O v2bz audita, para os eventos
LOW/BLOCKED (Petrópolis PET_2022_02_15 e PET_2024_03_21_28; Curitiba
CUR_EVENT_REGISTRY_MISSING), que evidência já existe localmente, classifica cada
fonte/geometria, **resolve o escopo de hazard** (flood vs mass_movement vs
multi-hazard vs out-of-scope) e prepara um scaffold de reparo do registry de
Curitiba — **sem inventar evento nem geometria** e sem misturar mass_movement com
flood. Não cria label/negativo/treino. Busca web não executada
(`EXTERNAL_WEB_SEARCH_NOT_PERFORMED`), termos públicos só logados.

```powershell
python scripts\multimodal\revp_v2bz_expansion_evidence_acquisition_and_scope_resolver.py --force
```

Estado atual (dados reais): **3 eventos-alvo / 193 fontes locais inventariadas**
(8 OFFICIAL_CONTEXT, 102 QA_DERIVED catálogos, 83 UNVERIFIED) — **0 com geometria
vetorial event-specific**. Escopo: **2 `HAZARD_SCOPE_MASS_MOVEMENT_SEPARATE_COHORT`**
(Petrópolis, `can_join_flood_cohort=false`) + **1 `HAZARD_SCOPE_FLOOD_COMPATIBLE`**
(Curitiba, hazard detectado do candidato reparado). Curitiba: registry reparável —
scaffold referencia candidato real `CUR_2022_01_15` (urban_flooding, 2 candidatos
do v1uv) sem inventar, `CURITIBA_EVENT_REGISTRY_REPAIR_SCAFFOLD_READY`. Fila: 3
eventos, `needs_user_decision=false`. Todas as geometrias `can_support_formal_gt=false`.
Treino bloqueado: `COHORT_EXPANSION_DATA_NOT_READY`.

Guardrails obrigatórios (todos `PASS`):

- `labels_created_false`
- `formal_positive_not_created`
- `formal_negative_not_created`
- `no_event_invented`
- `no_geometry_invented`
- `no_negative_from_absence`
- `hazard_scope_not_collapsed`
- `mass_movement_not_forced_into_flood`
- `registry_repair_not_label`
- `acquisition_queue_not_training_ready`
- `allowed_for_training_false`
- `training_still_blocked`
- `no_heavy_outputs`
- `private_absolute_paths_removed`

Outputs: `local_runs/ground_truth/v2bz/`

## v2ca — binding de registry de eventos Curitiba + pipeline de aquisição de evidência

Transformar Curitiba de `CUR_EVENT_REGISTRY_MISSING` numa coorte flood-compatible
rastreável. Repara o registry a partir do registry de candidatos real v1uv (dois
eventos oficiais `urban_flooding`: `CUR_2022_01_15` e `CUR_2022_01_05` — nunca
inventa evento), inventaria fontes locais de Curitiba, audita todos os patches
Curitiba da feature table v2bn, recupera boundaries dos patches a partir dos bounds
de header de raster gravados no audit v1fs (EPSG:32722 reprojetado para WGS84, sem
abrir raster pesado), cria candidatos de binding patch-evento e a fila para repetir
a cadeia v2bp→v2bq→v2bt→v2bu→v2bx. Não cria label, não cria negativo formal, não
libera treino; área de risco nunca vira footprint de evento; ausência nunca vira
negativo; `can_support_formal_gt` permanece `false` até footprint validado.

```powershell
python scripts\multimodal\revp_v2ca_curitiba_event_registry_binding_and_acquisition.py --force
```

Estado atual (inputs reais): 2 eventos reparados (`CUR_2022_01_05`, `CUR_2022_01_15`,
oficiais, `can_create_training_label=false`); 119 fontes inventariadas (2 contexto
oficial / 51 derivadas / 66 não-verificadas); 0 geometria/pontos event-specific →
escopo CONTEXT_ONLY; 43 patches na região; **43/43 boundaries recuperadas**
(EPSG:32722→WGS84); 4 patches com embedding DINO; 86 bindings patch-evento (2×43);
8 prontos para adjudicação (4 patches com embedding × 2 eventos); **0 prontos para
overlay** (evento sem geometria); decisão `CURITIBA_GEOMETRY_OR_POINT_EVIDENCE_NOT_READY`.

Guardrails obrigatórios (todos PASS):

- `labels_created_false`
- `formal_positive_not_created`
- `formal_negative_not_created`
- `no_event_invented`
- `no_geometry_invented`
- `registry_repair_not_label`
- `risk_area_not_event_footprint`
- `no_negative_from_absence`
- `binding_not_label`
- `acquisition_queue_not_training_ready`
- `allowed_for_training_false`
- `training_still_blocked`
- `no_heavy_outputs`
- `private_absolute_paths_removed`

Próximos módulos quando geometria/pontos forem adquiridos: `CURITIBA_BLOCKED_ACQUIRE_GEOMETRY`
→ `CURITIBA_QA_GEOMETRY_FROM_POINTS` (se houver pontos) ou validação de footprint →
`CURITIBA_V2BP_ADJUDICATION` → `CURITIBA_OVERLAY_RETRY` → dry-run.

Outputs: `local_runs/ground_truth/v2ca/`

## Preparação do commit

Antes do commit final, executar:

```powershell
python -m pytest -q
git diff --check
git status --short
```

Não adicionar ao stage nem commitar `local_runs/`, `.npz`, `.npy`, rasters, checkpoints ou outputs PNG de revisão local.
