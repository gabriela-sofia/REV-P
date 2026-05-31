"""REV-P v1pq — Controlled smoke embedding executor.

Executes embedding only if:
  - v1pp says backend/model available
  - REVP_DINO_DRY_RUN != true (default: true → dry run)
  - REVP_DINO_ALLOW_DOWNLOAD != true (default: false)

Never trains. Never creates labels. Never creates ground truth.
Embeddings are 768D review-only vectors.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

from revp_v1pn_v1pt_dino_execution_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, SCHEMAS,
    _f, _p, assert_no_forbidden_true, can_execute_embedding,
    is_fixture_or_synthetic, normalize_region, path_hash,
    probe_backend, require_no_abs_paths, sha256_short, validate_vector,
    vector_stats, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import read_csv, ROOT

OUT_RESULTS = _p("REVP_V1PQ_OUT_RESULTS", DATASETS / "dino_controlled_smoke_embedding_results_v1pq.csv")
OUT_FAILURES = _p("REVP_V1PQ_OUT_FAILURES", DATASETS / "dino_controlled_smoke_embedding_failures_v1pq.csv")
OUT_SUM = _p("REVP_V1PQ_OUT_SUM", DATASETS / "dino_controlled_smoke_embedding_summary_v1pq.csv")
SCH_RES = _p("REVP_V1PQ_SCH_RES", SCHEMAS / "dino_controlled_smoke_embedding_results_v1pq_schema.csv")
SCH_FAIL = _p("REVP_V1PQ_SCH_FAIL", SCHEMAS / "dino_controlled_smoke_embedding_failures_v1pq_schema.csv")
SCH_SUM = _p("REVP_V1PQ_SCH_SUM", SCHEMAS / "dino_controlled_smoke_embedding_summary_v1pq_schema.csv")
DOC = _p("REVP_V1PQ_DOC", DOCS / "revp_v1pq_controlled_smoke_embedding_executor.md")
IN_QUEUE = _p("REVP_V1PQ_IN_QUEUE", DATASETS / "dino_embedding_execution_queue_v1po.csv")
REVP_SOURCE_ROOT = Path(os.environ.get("REVP_DINO_SOURCE_ROOT", str(ROOT)))

MAX_EXECUTE = int(os.environ.get("REVP_DINO_MAX_EXECUTE", "5"))
DRY_RUN = os.environ.get("REVP_DINO_DRY_RUN", "true").lower() == "true"

RESULTS_FIELDS = [
    "embedding_run_id", "queue_id", "visual_asset_id", "patch_id", "alias",
    "region", "model_source", "vector_dim", "embedding", "vector_sha256_16",
    "execution_mode", "dino_allowed_use", "can_create_label", "can_train_model",
    "target_created", "status", "blocked_reason", "notes",
]
FAILURE_FIELDS = [
    "failure_id", "queue_id", "visual_asset_id", "patch_id",
    "failure_stage", "error_type", "error_message_short", "status", "blocked_reason",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _load_model(info: dict[str, Any]) -> Any | None:
    """Load model from local path or name. Never downloads if allow_download=False."""
    if not info["can_execute"]:
        return None
    allow_dl = info["allow_download"]
    mp = info["model_path"]
    mn = info["model_name"]
    transformers_available = info["transformers"][0]
    if transformers_available:
        try:
            from transformers import AutoImageProcessor, AutoModel
            source = mp if (mp and Path(mp).exists()) else (mn if (mn and allow_dl) else None)
            if source is None:
                return None
            processor = AutoImageProcessor.from_pretrained(source, local_files_only=not allow_dl)
            model = AutoModel.from_pretrained(source, local_files_only=not allow_dl)
            model.eval()
            return ("transformers", processor, model)
        except Exception:
            pass
    return None


def _embed_image(model_bundle: Any, img_path: Path) -> list[float] | None:
    """Generate 768D embedding from image file. Returns None on error."""
    try:
        from PIL import Image
        import torch
        kind, processor, model = model_bundle
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token or mean pool
        if hasattr(outputs, "last_hidden_state"):
            vec = outputs.last_hidden_state[0, 0, :].tolist()
        elif hasattr(outputs, "pooler_output"):
            vec = outputs.pooler_output[0].tolist()
        else:
            return None
        return vec if isinstance(vec, list) else list(vec)
    except Exception:
        return None


def execute(queue: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    info = probe_backend()
    model_bundle = None

    if not DRY_RUN and info["can_execute"]:
        model_bundle = _load_model(info)

    model_src = info["model_path"] or info["model_name"] or "none"

    for item in queue[:MAX_EXECUTE]:
        qid = item.get("queue_id", "")
        asid = item.get("visual_asset_id", "")
        patch = item.get("patch_id", "UNKNOWN")
        alias = item.get("alias", "")
        region = normalize_region(item.get("region", ""))
        rel = item.get("relative_path", "")
        img_path = REVP_SOURCE_ROOT / rel

        run_id = f"V1PQ_RUN_{len(results)+len(failures)+1:05d}"

        if DRY_RUN:
            results.append({
                "embedding_run_id": run_id, "queue_id": qid, "visual_asset_id": asid,
                "patch_id": patch, "alias": alias, "region": region,
                "model_source": model_src, "vector_dim": "0", "embedding": "[]",
                "vector_sha256_16": "", "execution_mode": "DRY_RUN",
                "dino_allowed_use": "EMBEDDING_SKIPPED_DRY_RUN",
                "can_create_label": "false", "can_train_model": "false", "target_created": "false",
                "status": "EMBEDDING_SKIPPED_DRY_RUN", "blocked_reason": "dry_run=true", "notes": "",
            })
            continue

        if not info["can_execute"]:
            failures.append({
                "failure_id": f"V1PQ_FAIL_{len(failures)+1:05d}",
                "queue_id": qid, "visual_asset_id": asid, "patch_id": patch,
                "failure_stage": "model_probe", "error_type": "MODEL_UNAVAILABLE",
                "error_message_short": info["final_status"],
                "status": "EMBEDDING_SKIPPED_MODEL_UNAVAILABLE",
                "blocked_reason": "no_local_model",
            })
            continue

        if model_bundle is None:
            failures.append({
                "failure_id": f"V1PQ_FAIL_{len(failures)+1:05d}",
                "queue_id": qid, "visual_asset_id": asid, "patch_id": patch,
                "failure_stage": "model_load", "error_type": "MODEL_LOAD_FAILED",
                "error_message_short": "model_bundle_is_none",
                "status": "EMBEDDING_BLOCKED_NO_MODEL",
                "blocked_reason": "model_load_failed",
            })
            continue

        if not img_path.exists():
            failures.append({
                "failure_id": f"V1PQ_FAIL_{len(failures)+1:05d}",
                "queue_id": qid, "visual_asset_id": asid, "patch_id": patch,
                "failure_stage": "image_load", "error_type": "IMAGE_NOT_FOUND",
                "error_message_short": "image_path_does_not_exist",
                "status": "EMBEDDING_BLOCKED_RUNTIME_ERROR",
                "blocked_reason": "image_not_found",
            })
            continue

        try:
            vec = _embed_image(model_bundle, img_path)
            if vec is None:
                raise RuntimeError("embed returned None")
            status, blocked = validate_vector(vec)
            if status != "VALID_REVIEW_ONLY":
                failures.append({
                    "failure_id": f"V1PQ_FAIL_{len(failures)+1:05d}",
                    "queue_id": qid, "visual_asset_id": asid, "patch_id": patch,
                    "failure_stage": "vector_validation", "error_type": "INVALID_VECTOR",
                    "error_message_short": blocked,
                    "status": "EMBEDDING_BLOCKED_INVALID_DIM" if "DIMENSION" in blocked else "EMBEDDING_BLOCKED_RUNTIME_ERROR",
                    "blocked_reason": blocked,
                })
                continue
            st = vector_stats(vec)
            vsha = sha256_short(",".join(f"{x:.6g}" for x in vec))
            results.append({
                "embedding_run_id": run_id, "queue_id": qid, "visual_asset_id": asid,
                "patch_id": patch, "alias": alias, "region": region,
                "model_source": model_src, "vector_dim": str(len(vec)),
                "embedding": json.dumps(vec),
                "vector_sha256_16": vsha, "execution_mode": "REAL_EXECUTION",
                "dino_allowed_use": "EMBEDDING_EXECUTED_REVIEW_ONLY",
                "can_create_label": "false", "can_train_model": "false", "target_created": "false",
                "status": "EMBEDDING_EXECUTED_REVIEW_ONLY",
                "blocked_reason": "", "notes": f"norm={_f(st['norm'])}",
            })
        except Exception as exc:
            failures.append({
                "failure_id": f"V1PQ_FAIL_{len(failures)+1:05d}",
                "queue_id": qid, "visual_asset_id": asid, "patch_id": patch,
                "failure_stage": "embedding", "error_type": type(exc).__name__,
                "error_message_short": str(exc)[:120],
                "status": "EMBEDDING_BLOCKED_RUNTIME_ERROR",
                "blocked_reason": "runtime_exception",
            })

    return results, failures


def run() -> None:
    queue = read_csv(IN_QUEUE)
    results, failures = execute(queue)
    for label, rows in (("v1pq_results", results), ("v1pq_failures", failures)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)

    mode = "DRY_RUN" if DRY_RUN else "REAL"
    executed = sum(1 for r in results if r.get("status") == "EMBEDDING_EXECUTED_REVIEW_ONLY")
    summary = [
        {"stat_key": "queue_items_processed", "stat_value": str(len(results) + len(failures))},
        {"stat_key": "embeddings_attempted", "stat_value": str(0 if DRY_RUN else len(results) + len(failures))},
        {"stat_key": "embeddings_executed_review_only", "stat_value": str(executed)},
        {"stat_key": "failures", "stat_value": str(len(failures))},
        {"stat_key": "dry_run", "stat_value": str(DRY_RUN).lower()},
        {"stat_key": "execution_mode", "stat_value": mode},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
    ]
    write_csv(OUT_RESULTS, results, RESULTS_FIELDS)
    write_csv(OUT_FAILURES, failures, FAILURE_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_RES, RESULTS_FIELDS, "v1pq_smoke_embedding_results")
    write_schema(SCH_FAIL, FAILURE_FIELDS, "v1pq_smoke_embedding_failures")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pq_smoke_embedding_summary")
    write_doc(DOC, "v1pq — Controlled Smoke Embedding Executor", [
        "## Objetivo",
        "Executar embeddings somente se backend/modelo disponível e dry-run=false. "
        "Default: dry-run=true (apenas plano, sem execução).",
        "## Guardrails",
        "Não baixa modelo. Não treina. Não cria label. Vetores são representação "
        "768D review-only.",
        f"## Resultado",
        f"Modo: {mode}. Resultados: {len(results)}. Falhas: {len(failures)}. "
        f"Embeddings executados: {executed}.",
    ])
    print(f"[v1pq] mode={mode} results={len(results)} failures={len(failures)} executed={executed}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pq controlled smoke embedding executor").parse_args()
    run()
