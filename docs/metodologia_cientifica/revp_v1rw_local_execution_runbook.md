# v1rw — Local Execution Runbook

## Objetivo

Guia prático para a próxima fase real: configurar modelo local e roots Sentinel, rodar DINO em dry-run e validar antes de qualquer embedding real.

## Variáveis de ambiente obrigatórias

| Var | Obrigatório | Descrição |

| --- | --- | --- |

| `REVP_DINO_MODEL_PATH` | yes | Path to local DINOv2+registers model dir |

| `REVP_SENTINEL_LOCAL_ROOT` | yes | Root dir of local Sentinel TIF files |

| `REVP_DINO_VISUAL_ROOT` | yes | Root dir of visual assets for DINO queue |

| `REVP_DINO_ASSET_ROOT` | yes | Alternate root for DINO assets |

| `HF_HUB_OFFLINE` | yes | Disable HuggingFace Hub downloads |

| `REVP_DINO_ALLOW_DOWNLOAD` | yes | Forbid automatic model download |

| `REVP_DINO_DRY_RUN` | yes | Run in dry-run mode first |

| `REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH` | no | Path to filled A/B review CSV |

| `REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH` | no | Path to filled supervisor decision CSV |

| `REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH` | no | Path to filled external intake CSV |

## Passos

**S01** [CONFIG] Configure all required env vars in local shell profile
- Comando: `export REVP_DINO_MODEL_PATH=... (see env vars table)`
- Esperado: env vars readable
- Em falha: Abort; never commit paths

**S02** [CONFIG] Set HF_HUB_OFFLINE=1 and REVP_DINO_ALLOW_DOWNLOAD=false
- Comando: `export HF_HUB_OFFLINE=1; export REVP_DINO_ALLOW_DOWNLOAD=false`
- Esperado: No internet calls
- Em falha: Abort

**S03** [DRYRUN] Run local readiness audit in dry-run mode
- Comando: `python scripts/dino/revp_v1qn_local_root_environment_audit.py`
- Esperado: LOCAL_DINO_READINESS status reported
- Em falha: Check missing roots/model

**S04** [DRYRUN] Run smoke asset reconciliation
- Comando: `python scripts/dino/revp_v1qo_smoke_asset_local_reconciliation.py`
- Esperado: Assets resolved or repair suggestions generated
- Em falha: Check repair suggestions

**S05** [DRYRUN] Run local smoke run readiness gate
- Comando: `python scripts/dino/revp_v1qr_local_smoke_run_readiness_gate.py`
- Esperado: GATE_PASS or GATE_FAIL with reasons
- Em falha: Fix blockers before continuing

**S06** [VALIDATE] Confirm dry-run is still true; review outputs
- Comando: `echo $REVP_DINO_DRY_RUN`
- Esperado: true
- Em falha: Do NOT proceed if false unintentionally

**S07** [EXECUTE] Set dry_run=false and run controlled smoke embedding
- Comando: `export REVP_DINO_DRY_RUN=false; python scripts/dino/revp_v1qj_controlled_real_smoke_embedding_executor.py`
- Esperado: Embeddings written to gitignored output dir only
- Em falha: Check error log; never commit embeddings

**S08** [VALIDATE] Run guardrail scan on any new CSV outputs
- Comando: `python scripts/protocolo_c/revp_v1ru_cross_block_guardrail_audit.py`
- Esperado: GUARDRAIL_CLEAN
- Em falha: Fix violations before staging

**S09** [COMMIT_PREP] Run commit readiness package
- Comando: `python scripts/protocolo_c/revp_v1rv_commit_readiness_package.py`
- Esperado: Recommended/excluded files classified
- Em falha: Review ambiguous files manually

**S10** [NEVER] Never commit raster or embedding files
- Comando: `—`
- Esperado: —
- Em falha: —

## Regras críticas

1. `REVP_DINO_DRY_RUN=true` PRIMEIRO. Só mude para false depois de validar dry-run.

2. `HF_HUB_OFFLINE=1` SEMPRE antes de rodar qualquer script DINO.

3. NUNCA commitar `.tif`, `.tiff`, `.npy`, `.npz`, paths absolutos ou `local_runs/`.

4. Outputs reais de embeddings ficam em `local_runs/` (gitignored).

5. Rodar guardrail scan (v1ru) antes de qualquer staging.
