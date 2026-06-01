# v1qq — Local DINO Execution Config Template

## Objetivo

Gerar templates seguros para configurar execução smoke real de DINO. Nenhum path absoluto real é versionado.

## Como usar

1. Copie `configs/revp_dino_local_execution.env.example` para um arquivo local (não versionado).
2. Preencha os campos `<ABSOLUTE_...>` com caminhos reais locais.
3. Execute os passos do checklist em ordem.
4. Nunca comite o arquivo com paths reais preenchidos.

## PowerShell steps

- `set_model_path`: Set to local DINOv2 model directory. Never commit this value.
- `set_sentinel_root`: Root directory containing Sentinel TIF files locally.
- `lock_download`: Never change to true — prevents accidental model downloads.
- `enable_offline`: Force HuggingFace hub offline mode.
- `run_v1qn`: Audit local roots and model directory.
- `run_v1qo`: Reconcile smoke sample with local files.
- `run_v1qg`: Verify model config/weights without loading.
- `run_v1qi`: Audit local asset metadata (no pixel read unless env set).
- `dry_run_v1qj`: Dry-run by default. Review outputs before enabling real execution.
- `enable_real_execution_manual`: Enable real embedding execution ONLY after all gates pass. Manual step.
- `enable_pixel_read_manual`: Enable pixel reading ONLY for real execution run. Manual step.
