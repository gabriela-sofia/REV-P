"""Shared helpers for REV-P DINO execution bridge v1qa-v1qf.

Connects the expanded visual queue (v1pw) to the execution harness (v1pq).
Never creates labels, targets, ground truth, or training data.
Never downloads models. Never runs inference.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, SCHEMAS, ROOT,
    _p, assert_no_forbidden_true, normalize_region, path_hash,
    require_no_abs_paths, sanitized_rel_path, sha256_short,
    write_csv, write_doc, write_schema, read_csv,
)

__all__ = [
    "DATASETS", "DOCS", "SCHEMAS", "ROOT",
    "_p", "assert_no_forbidden_true", "normalize_region", "path_hash",
    "require_no_abs_paths", "sanitized_rel_path", "sha256_short",
    "write_csv", "write_doc", "write_schema", "read_csv",
    "BRIDGE_FORBIDDEN_FIELDS", "EXECUTION_STATUSES",
    "read_expanded_queue", "read_backend_probe",
    "dedup_queue", "validate_queue_item", "mask_local",
    "make_powershell_command",
]

BRIDGE_FORBIDDEN_FIELDS = (
    "can_create_label", "can_train_model", "target_created",
    "dino_can_create_label", "dino_can_train_model", "dino_target_field_created",
    "can_be_used_as_class", "can_infer_same_event", "dino_can_validate_event",
)

EXECUTION_STATUSES = frozenset({
    "READY_FOR_DRY_RUN_ONLY",
    "READY_FOR_LOCAL_MODEL_EXECUTION",
    "BLOCKED_MODEL_UNAVAILABLE",
    "BLOCKED_INVALID_QUEUE_ITEM",
    "BLOCKED_GUARDRAIL_VIOLATION",
})

_V1PW_QUEUE = _p("REVP_V1QA_IN_QUEUE",
                  DATASETS / "dino_review_only_execution_queue_expanded_v1pw.csv")
_V1PP_SUMMARY = _p("REVP_V1QA_IN_BACKEND",
                    DATASETS / "dino_backend_model_probe_summary_v1pp.csv")


def read_expanded_queue(path: Path | None = None) -> list[dict[str, str]]:
    return read_csv(path or _V1PW_QUEUE)


def read_backend_probe(path: Path | None = None) -> dict[str, str]:
    rows = read_csv(path or _V1PP_SUMMARY)
    return {r.get("stat_key", ""): r.get("stat_value", "") for r in rows}


def dedup_queue(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for r in items:
        key = (r.get("patch_id", ""), r.get("path_hash", "") or path_hash(r.get("relative_path", "")))
        k = "|".join(key)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def validate_queue_item(r: dict[str, str]) -> tuple[bool, str]:
    """Return (valid, blocked_reason). valid=True means safe to queue."""
    if r.get("can_create_label", "false") == "true":
        return (False, "guardrail_can_create_label_true")
    if r.get("can_train_model", "false") == "true":
        return (False, "guardrail_can_train_model_true")
    if r.get("target_created", "false") == "true":
        return (False, "guardrail_target_created_true")
    if r.get("dino_allowed_use", "") not in (
        "REVIEW_ONLY_REPRESENTATION", "EXPLORATORY_SIMILARITY_ONLY",
        "EXPLORATORY_REPRESENTATION_ONLY", "VISUAL_CONTEXT_ONLY",
    ):
        if r.get("dino_allowed_use", "").startswith("BLOCKED"):
            return (False, f"dino_use_blocked:{r.get('dino_allowed_use')}")
    return (True, "")


def mask_local(rel: str) -> str:
    if rel.startswith("local_runs/") or rel.startswith("local_runs\\"):
        return f"local_only:{path_hash(rel)}"
    return rel


def make_powershell_command(cmd_type: str, params: dict[str, str]) -> tuple[str, str]:
    """Return (powershell_command, safety_note) for a given command type."""
    if cmd_type == "set_model_path":
        mp = params.get("model_path", "<path_to_local_dino_model>")
        return (
            f'$env:REVP_DINO_MODEL_PATH = "{mp}"',
            "Set to an absolute path of a locally downloaded DINOv2 model directory.",
        )
    if cmd_type == "keep_download_false":
        return (
            '$env:REVP_DINO_ALLOW_DOWNLOAD = "false"',
            "Never change to true without explicit authorization; no model download by default.",
        )
    if cmd_type == "enable_dry_run":
        return (
            '$env:REVP_DINO_DRY_RUN = "true"',
            "Default mode; no embedding execution occurs.",
        )
    if cmd_type == "disable_dry_run_manual":
        return (
            '# MANUAL: $env:REVP_DINO_DRY_RUN = "false"  # Only set after model path is configured',
            "Requires manual confirmation and a valid REVP_DINO_MODEL_PATH.",
        )
    if cmd_type == "set_queue_path":
        qp = params.get("queue_path", "<path_to_queue_csv>")
        return (
            f'$env:REVP_V1PQ_QUEUE_PATH = "{qp}"',
            "Point executor to the expanded v1qa queue.",
        )
    if cmd_type == "run_v1pq":
        return (
            "python scripts/dino/revp_v1pq_controlled_smoke_embedding_executor.py",
            "Run only after model path and queue path are configured.",
        )
    if cmd_type == "run_v1pr":
        return (
            "python scripts/dino/revp_v1pr_import_smoke_embeddings_feature_store.py",
            "Run after v1pq to import valid embeddings into feature store.",
        )
    if cmd_type == "run_v1ps":
        return (
            "python scripts/dino/revp_v1ps_smoke_embedding_review_products.py",
            "Run after v1pr to generate review products.",
        )
    if cmd_type == "run_v1pt":
        return (
            "python scripts/dino/revp_v1pt_dino_execution_bundle.py",
            "Run last to consolidate execution results.",
        )
    return ("# unknown command type", "")
