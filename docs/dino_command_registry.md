# DINO Sentinel-first command registry

This registry lists the main commands for reproducing the REV-P DINO Sentinel-first workflow locally. Runtime outputs go under `local_runs/` and are not intended for Git.

Use `--force` only when replacing a local runtime directory is intentional. Use `--resume` and `--skip-existing` when preserving partial local runs.

## QA and audits

Run tests:

```powershell
python -m pytest -q
```

Check whitespace issues before commit:

```powershell
git diff --check
```

Audit forbidden files outside `local_runs/`:

```powershell
Get-ChildItem . -Recurse -Force |
  Where-Object {
    $_.FullName -notmatch "\\.git\\" -and
    $_.FullName -notmatch "\\local_runs\\" -and (
      $_.Name -match "CLAUDE|AGENTS|codex|Codex|claude|Claude|cbers|CBERS" -or
      $_.Name -match "\.tif$|\.tiff$|\.zip$|\.npy$|\.npz$|\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.parquet$|\.index$" -or
      $_.FullName -match "\\data\\" -or
      $_.FullName -match "\\outputs\\" -or
      $_.FullName -match "\\patches\\" -or
      $_.FullName -match "\\archive_drive\\"
    )
  } |
  Select-Object FullName
```

## v1fw — extraction scaffold

Dry-run only by default:

```powershell
python scripts\dino\revp_v1fw_dino_embedding_extraction_scaffold.py --force
```

With local preflight:

```powershell
python scripts\dino\revp_v1fw_dino_embedding_extraction_scaffold.py --asset-preflight local_runs\dino_asset_preflight\v1fv\dino_local_asset_preflight_v1fv.csv --force
```

## v1fx — smoke embedding execution

Explicit execution, small limit:

```powershell
python scripts\dino\revp_v1fx_dino_smoke_embedding_execution.py --execute --limit 5 --force --allow-cpu --skip-model-if-unavailable
```

Allow model download only when intentional:

```powershell
python scripts\dino\revp_v1fx_dino_smoke_embedding_execution.py --execute --limit 5 --force --allow-cpu --allow-model-download
```

## v1fy — exploratory corpus analysis

```powershell
python scripts\dino\revp_v1fy_dino_embedding_corpus_analysis.py --force
```

Useful flags:

```powershell
python scripts\dino\revp_v1fy_dino_embedding_corpus_analysis.py --limit 5 --top-k 3 --seed 42 --force
```

## v1fz — balanced regional corpus

Balanced subset:

```powershell
python scripts\dino\revp_v1fz_dino_balanced_embedding_corpus.py --execute --per-region-limit 2 --force --allow-cpu --allow-model-download
```

Specific regions:

```powershell
python scripts\dino\revp_v1fz_dino_balanced_embedding_corpus.py --execute --regions Curitiba Petropolis Recife --per-region-limit 2 --force --allow-cpu
```

## v1ga — structural consistency

```powershell
python scripts\dino\revp_v1ga_dino_embedding_structural_consistency_analysis.py --force
```

## v1gb — local visual structural review

```powershell
python scripts\dino\revp_v1gb_dino_embedding_local_visual_structural_review.py --force
```

## v1gc — geo-structural diagnostics

```powershell
python scripts\dino\revp_v1gc_dino_embedding_geo_structural_diagnostics.py --force
```

## v1gd — perturbation robustness

```powershell
python scripts\dino\revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py --force --allow-cpu --force-cpu --allow-model-download
```

Offline test proxy only:

```powershell
python scripts\dino\revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py --force --embedding-proxy-for-tests
```

## v1ge — expanded Sentinel corpus

Expanded balanced run:

```powershell
python scripts\dino\revp_v1ge_dino_expanded_sentinel_embedding_corpus.py --execute --per-region-limit 4 --batch-size 4 --force --allow-cpu --force-cpu --allow-model-download
```

Resume partial run:

```powershell
python scripts\dino\revp_v1ge_dino_expanded_sentinel_embedding_corpus.py --execute --per-region-limit 4 --resume --skip-existing --allow-cpu
```

Limit total work:

```powershell
python scripts\dino\revp_v1ge_dino_expanded_sentinel_embedding_corpus.py --execute --limit 12 --batch-size 4 --force --allow-cpu
```

## v1gf — structural evidence index

```powershell
python scripts\dino\revp_v1gf_dino_structural_evidence_index.py --force
```

Use a specific embedding manifest:

```powershell
python scripts\dino\revp_v1gf_dino_structural_evidence_index.py --embedding-manifest local_runs\dino_embeddings\v1ge\dino_expanded_embedding_manifest_v1ge.csv --force
```

## v1gg — human review package

```powershell
python scripts\dino\revp_v1gg_dino_human_review_package.py --force
```

## v1gh — longitudinal structural diagnostics

```powershell
python scripts\dino\revp_v1gh_dino_longitudinal_structural_diagnostics.py --force
```

## v1gi — structural provenance tracker

```powershell
python scripts\dino\revp_v1gi_dino_structural_provenance_tracker.py --force
```

## v1gj — multimodal readiness audit

This is an audit only. It does not enable multimodal execution.

```powershell
python scripts\dino\revp_v1gj_multimodal_readiness_audit.py --force
```

Required guardrails:

- `multimodal_execution_enabled=false`
- `multimodal_training_enabled=false`
- `multimodal_hold=true`

## v1gk — reproducibility audit

```powershell
python scripts\dino\revp_v1gk_dino_pipeline_reproducibility_audit.py --force
```

## Commit preparation

Before a final commit, run:

```powershell
python -m pytest -q
git diff --check
git status --short
```

Do not stage or commit `local_runs/`, `.npz`, `.npy`, rasters, checkpoints, or local PNG review outputs.
