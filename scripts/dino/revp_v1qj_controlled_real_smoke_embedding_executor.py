"""REV-P v1qj — Controlled real smoke embedding executor (fail-closed).

Executes REAL 768D DINOv2 embeddings ONLY when every gate passes:
  * v1qg local model audit ready (config + weights, offline, no download);
  * v1qi assets resolved / preprocessing ready;
  * REVP_DINO_DRY_RUN=false;
  * REVP_DINO_PIXEL_READ_ALLOWED=true;
  * HF_HUB_OFFLINE=1, REVP_DINO_ALLOW_DOWNLOAD=false.

Default is dry-run. On any failed gate, writes empty/fail-closed outputs with
the correct status. Never downloads, never trains, never creates labels.
Vectors are review-only visual descriptors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, SCHEMAS,
    _f, _p, assert_no_forbidden_true, embedding_columns, env_int, env_str,
    env_true, normalize_region, path_hash, read_csv, require_no_abs_paths,
    safe_model_probe, sha256_short, validate_vector, vector_stats,
    vector_to_columns, write_csv, write_doc, write_schema,
)

IN_SEL = _p("REVP_V1QJ_IN_SEL", DATASETS / "dino_smoke_sample_selection_v1qh.csv")
IN_ASSET = _p("REVP_V1QJ_IN_ASSET", DATASETS / "dino_local_asset_preprocessing_audit_v1qi.csv")
IN_MODEL = _p("REVP_V1QJ_IN_MODEL", DATASETS / "dino_local_model_offline_summary_v1qg.csv")
OUT_STORE = _p("REVP_V1QJ_OUT_STORE", DATASETS / "dino_smoke_embeddings_feature_store_v1qj.csv")
OUT_MAN = _p("REVP_V1QJ_OUT_MAN", DATASETS / "dino_smoke_embedding_execution_manifest_v1qj.csv")
OUT_FAIL = _p("REVP_V1QJ_OUT_FAIL", DATASETS / "dino_smoke_embedding_failures_v1qj.csv")
OUT_SUM = _p("REVP_V1QJ_OUT_SUM", DATASETS / "dino_smoke_embedding_summary_v1qj.csv")
SCH_STORE = _p("REVP_V1QJ_SCH_STORE", SCHEMAS / "dino_smoke_embeddings_feature_store_v1qj_schema.csv")
SCH_MAN = _p("REVP_V1QJ_SCH_MAN", SCHEMAS / "dino_smoke_embedding_execution_manifest_v1qj_schema.csv")
SCH_FAIL = _p("REVP_V1QJ_SCH_FAIL", SCHEMAS / "dino_smoke_embedding_failures_v1qj_schema.csv")
SCH_SUM = _p("REVP_V1QJ_SCH_SUM", SCHEMAS / "dino_smoke_embedding_summary_v1qj_schema.csv")
DOC = _p("REVP_V1QJ_DOC", DOCS / "revp_v1qj_controlled_real_smoke_embedding_executor.md")

META_FIELDS = [
    "embedding_id", "smoke_id", "patch_id", "alias", "region", "visual_asset_id",
    "relative_path", "path_hash", "model_name", "model_path_hash", "embedding_dim",
    "l2_normalized", "vector_norm", "dino_allowed_use", "review_only",
    "can_create_label", "can_train_model", "target_created",
]
STORE_FIELDS = META_FIELDS + embedding_columns(EXPECTED_DINO_DIM)
MAN_FIELDS = [
    "manifest_id", "smoke_id", "patch_id", "relative_path", "path_hash",
    "execution_mode", "embedding_attempted", "embedding_valid", "vector_norm",
    "status", "blocked_reason", "notes",
]
FAIL_FIELDS = [
    "failure_id", "smoke_id", "patch_id", "failure_stage", "error_type",
    "error_message_short", "status", "blocked_reason",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _gate_status() -> tuple[str, dict[str, Any]]:
    """Evaluate all gates. Returns (gate_status, context)."""
    dry_run = env_true("REVP_DINO_DRY_RUN", True)
    pixel_allowed = env_true("REVP_DINO_PIXEL_READ_ALLOWED", False)
    allow_dl = env_true("REVP_DINO_ALLOW_DOWNLOAD", False)
    model_path = env_str("REVP_DINO_MODEL_PATH", "")
    probe = safe_model_probe(model_path or None)

    model_summary = {r.get("stat_key", ""): r.get("stat_value", "") for r in read_csv(IN_MODEL)}
    model_ready = model_summary.get("model_ready", "false") == "true" or (
        probe["model_path_exists"] and probe["config_exists"]
        and probe["weights_exists"] and not allow_dl
        and probe["transformers_available"]
    )

    ctx = {
        "dry_run": dry_run, "pixel_allowed": pixel_allowed, "allow_dl": allow_dl,
        "model_path": model_path, "probe": probe, "model_ready": model_ready,
    }
    if dry_run:
        return ("DRY_RUN", ctx)
    if not model_ready:
        return ("MODEL_MISSING", ctx)
    if allow_dl:
        return ("MODEL_MISSING", ctx)
    if not pixel_allowed:
        return ("PIXEL_BLOCKED", ctx)
    return ("EXECUTE", ctx)


def _load_model(model_path: str, allow_dl: bool) -> Any | None:
    try:
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=not allow_dl)
        model = AutoModel.from_pretrained(model_path, local_files_only=not allow_dl)
        model.eval()
        return (processor, model)
    except Exception:
        return None


def _embed(bundle: Any, img_path: Path, l2: bool) -> list[float] | None:
    try:
        from PIL import Image
        import torch
        processor, model = bundle
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            vec = outputs.last_hidden_state[0, 0, :].tolist()
        elif hasattr(outputs, "pooler_output"):
            vec = outputs.pooler_output[0].tolist()
        else:
            return None
        vec = list(vec)
        if l2:
            import math
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
        return vec
    except Exception:
        return None


def _resolved_paths() -> dict[str, Path]:
    """Map smoke_id -> resolved local Path from v1qi audit (ready rows only)."""
    from revp_v1qg_v1qm_smoke_embedding_common import resolve_local_asset
    out: dict[str, Path] = {}
    for r in read_csv(IN_ASSET):
        if r.get("status") != "ASSET_READY_FOR_DINO_PREPROCESSING":
            continue
        rel = r.get("relative_path", "")
        p = resolve_local_asset(rel) if rel else None
        if p is not None:
            out[r.get("smoke_id", "")] = p
    return out


def run() -> None:
    gate, ctx = _gate_status()
    sel = read_csv(IN_SEL)
    max_exec = env_int("REVP_DINO_MAX_EXECUTE", 32)
    l2 = env_true("REVP_DINO_L2_NORMALIZE", True)
    model_path = ctx["model_path"]
    model_name = env_str("REVP_DINO_MODEL_NAME", "") or (Path(model_path).name if model_path else "")
    model_path_hash = path_hash(model_path) if model_path else ""

    store: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    status_map = {
        "DRY_RUN": "DINO_SMOKE_EMBEDDINGS_DRY_RUN_ONLY",
        "MODEL_MISSING": "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED",
        "PIXEL_BLOCKED": "DINO_SMOKE_EMBEDDINGS_PIXEL_READ_BLOCKED_FAIL_CLOSED",
    }

    if gate != "EXECUTE":
        final = status_map.get(gate, "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED")
        for i, r in enumerate(sel[:max_exec], 1):
            manifest.append({
                "manifest_id": f"V1QJ_MAN_{i:05d}", "smoke_id": r.get("smoke_id", ""),
                "patch_id": r.get("patch_id", ""), "relative_path": r.get("relative_path", ""),
                "path_hash": r.get("path_hash", ""), "execution_mode": gate,
                "embedding_attempted": "false", "embedding_valid": "false",
                "vector_norm": "", "status": final,
                "blocked_reason": gate.lower(), "notes": "",
            })
        _write(store, manifest, failures, final, gate, 0, 0)
        print(f"[v1qj] gate={gate} status={final} valid=0")
        return

    # EXECUTE path — all gates passed.
    resolved = _resolved_paths()
    bundle = _load_model(model_path, ctx["allow_dl"])
    valid = 0
    attempted = 0
    if bundle is None:
        final = "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED"
        for i, r in enumerate(sel[:max_exec], 1):
            failures.append({
                "failure_id": f"V1QJ_FAIL_{i:05d}", "smoke_id": r.get("smoke_id", ""),
                "patch_id": r.get("patch_id", ""), "failure_stage": "model_load",
                "error_type": "MODEL_LOAD_FAILED", "error_message_short": "bundle_is_none",
                "status": final, "blocked_reason": "model_load_failed",
            })
        _write(store, manifest, failures, final, gate, 0, len(failures))
        print(f"[v1qj] gate={gate} status={final} valid=0")
        return

    for i, r in enumerate(sel[:max_exec], 1):
        smoke_id = r.get("smoke_id", "")
        patch = (r.get("patch_id", "") or "UNKNOWN").upper()
        alias = r.get("alias", "") or patch
        region = normalize_region(r.get("region", ""))
        rel = r.get("relative_path", "")
        ph = r.get("path_hash", "") or (path_hash(rel) if rel else "")
        img = resolved.get(smoke_id)
        man = {
            "manifest_id": f"V1QJ_MAN_{i:05d}", "smoke_id": smoke_id, "patch_id": patch,
            "relative_path": rel, "path_hash": ph, "execution_mode": "REAL_EXECUTION",
            "embedding_attempted": "false", "embedding_valid": "false", "vector_norm": "",
            "status": "", "blocked_reason": "", "notes": "",
        }
        if img is None:
            man["status"] = "DINO_SMOKE_EMBEDDINGS_ASSETS_MISSING_FAIL_CLOSED"
            man["blocked_reason"] = "asset_not_ready"
            failures.append({
                "failure_id": f"V1QJ_FAIL_{len(failures)+1:05d}", "smoke_id": smoke_id,
                "patch_id": patch, "failure_stage": "asset_resolution",
                "error_type": "ASSET_NOT_READY", "error_message_short": "no_ready_local_asset",
                "status": "DINO_SMOKE_EMBEDDINGS_ASSETS_MISSING_FAIL_CLOSED",
                "blocked_reason": "asset_not_ready",
            })
            manifest.append(man)
            continue
        attempted += 1
        man["embedding_attempted"] = "true"
        vec = _embed(bundle, img, l2)
        vstatus, blocked = validate_vector(vec)
        if vstatus != "VALID_REVIEW_ONLY":
            man["status"] = "DINO_SMOKE_EMBEDDINGS_EXECUTION_FAILED_FAIL_CLOSED"
            man["blocked_reason"] = blocked or "invalid_vector"
            failures.append({
                "failure_id": f"V1QJ_FAIL_{len(failures)+1:05d}", "smoke_id": smoke_id,
                "patch_id": patch, "failure_stage": "embedding",
                "error_type": "INVALID_VECTOR", "error_message_short": blocked or "embed_failed",
                "status": "DINO_SMOKE_EMBEDDINGS_EXECUTION_FAILED_FAIL_CLOSED",
                "blocked_reason": blocked or "invalid_vector",
            })
            manifest.append(man)
            continue
        assert vec is not None  # validate_vector guarantees a non-None vector here
        st = vector_stats(vec)
        valid += 1
        meta = {
            "embedding_id": f"V1QJ_EMB_{valid:05d}", "smoke_id": smoke_id,
            "patch_id": patch, "alias": alias, "region": region,
            "visual_asset_id": r.get("visual_asset_id", ""), "relative_path": rel,
            "path_hash": ph, "model_name": model_name, "model_path_hash": model_path_hash,
            "embedding_dim": str(len(vec)), "l2_normalized": str(l2).lower(),
            "vector_norm": _f(st["norm"]), "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
            "review_only": "true", "can_create_label": "false",
            "can_train_model": "false", "target_created": "false",
        }
        row = dict(meta)
        row.update(vector_to_columns(vec, EXPECTED_DINO_DIM))
        store.append(row)
        man["embedding_valid"] = "true"
        man["vector_norm"] = _f(st["norm"])
        man["status"] = "DINO_SMOKE_EMBEDDINGS_READY_REVIEW_ONLY"
        man["notes"] = f"sha={sha256_short(','.join(f'{x:.6g}' for x in vec))}"
        manifest.append(man)

    if valid > 0:
        final = "DINO_SMOKE_EMBEDDINGS_READY_REVIEW_ONLY"
    elif attempted > 0:
        final = "DINO_SMOKE_EMBEDDINGS_EXECUTION_FAILED_FAIL_CLOSED"
    else:
        final = "DINO_SMOKE_EMBEDDINGS_ASSETS_MISSING_FAIL_CLOSED"
    _write(store, manifest, failures, final, gate, valid, len(failures))
    print(f"[v1qj] gate={gate} status={final} valid={valid} failures={len(failures)}")


def _write(store: list[dict[str, Any]], manifest: list[dict[str, Any]],
           failures: list[dict[str, Any]], final: str, gate: str,
           valid: int, n_fail: int) -> None:
    for label, rows in (("v1qj_store", store), ("v1qj_manifest", manifest),
                        ("v1qj_failures", failures)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    summary = [
        {"stat_key": "execution_gate", "stat_value": gate},
        {"stat_key": "embeddings_valid_768d", "stat_value": str(valid)},
        {"stat_key": "embeddings_failed", "stat_value": str(n_fail)},
        {"stat_key": "embedding_dim", "stat_value": str(EXPECTED_DINO_DIM)},
        {"stat_key": "l2_normalized", "stat_value": str(env_true("REVP_DINO_L2_NORMALIZE", True)).lower()},
        {"stat_key": "dry_run", "stat_value": str(env_true("REVP_DINO_DRY_RUN", True)).lower()},
        {"stat_key": "pixel_read_allowed", "stat_value": str(env_true("REVP_DINO_PIXEL_READ_ALLOWED", False)).lower()},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "final_status", "stat_value": final},
    ]
    require_no_abs_paths(summary, "v1qj_summary")
    assert_no_forbidden_true(summary, "v1qj_summary")
    write_csv(OUT_STORE, store, STORE_FIELDS)
    write_csv(OUT_MAN, manifest, MAN_FIELDS)
    write_csv(OUT_FAIL, failures, FAIL_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_STORE, STORE_FIELDS, "v1qj_smoke_embeddings_feature_store")
    write_schema(SCH_MAN, MAN_FIELDS, "v1qj_smoke_embedding_execution_manifest")
    write_schema(SCH_FAIL, FAIL_FIELDS, "v1qj_smoke_embedding_failures")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qj_smoke_embedding_summary")
    write_doc(DOC, "v1qj — Controlled Real Smoke Embedding Executor", [
        "## Objetivo",
        "Executar embeddings DINOv2 768D reais somente quando todos os gates passam "
        "(modelo local offline, assets prontos, dry-run=false, pixel read autorizado). "
        "Default: dry-run.",
        "## Guardrails",
        "Não baixa modelo. Não treina. Não cria rótulo/target/ground truth. Vetores "
        "são descritores visuais review-only.",
        "## Status",
        f"**{final}**. Vetores válidos 768D: {valid}.",
    ])


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qj controlled real smoke embedding executor").parse_args()
    run()
