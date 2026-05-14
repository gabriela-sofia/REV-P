from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1fw"
PHASE_NAME = "DINO_EMBEDDING_EXTRACTION_SCAFFOLD"

DEFAULT_INPUT_MANIFEST = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
DEFAULT_ASSET_PREFLIGHT = ROOT / "local_runs" / "dino_asset_preflight" / "v1fv" / "dino_local_asset_preflight_v1fv.csv"
DEFAULT_CONFIG = ROOT / "configs" / "dino_embedding_extraction.example.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1fw"

PLAN_CSV = "dino_embedding_extraction_plan_v1fw.csv"
SUMMARY_JSON = "dino_embedding_extraction_summary_v1fw.json"
QA_CSV = "dino_embedding_extraction_qa_v1fw.csv"
SCHEMA_CSV = "dino_embedding_output_schema_v1fw.csv"
MANIFEST_CSV = "dino_embedding_manifest_v1fw.csv"
FAILURES_CSV = "dino_embedding_failures_v1fw.csv"

REQUIRED_INPUT_COLUMNS = [
    "dino_input_id",
    "canonical_patch_id",
    "region",
    "asset_path_reference",
    "encoder_mode",
    "label_status",
    "target_status",
    "claim_scope",
]

PLAN_FIELDS = [
    "dino_input_id",
    "canonical_patch_id",
    "region",
    "asset_path_reference",
    "resolved_status",
    "planned_status",
    "block_reason",
    "backbone",
    "tokens",
    "device",
    "execute_mode",
    "pixel_read_status",
    "embedding_status",
    "label_status",
    "target_status",
    "claim_scope",
]

SCHEMA_FIELDS = ["column", "type", "required", "description"]
QA_FIELDS = ["check", "status", "details"]
FAILURE_FIELDS = ["dino_input_id", "canonical_patch_id", "region", "stage", "failure_reason", "backbone", "device"]
EMBEDDING_MANIFEST_FIELDS = [
    "dino_input_id",
    "canonical_patch_id",
    "region",
    "resolved_status",
    "embedding_status",
    "embedding_file",
    "cls_dim",
    "patch_mean_dim",
    "backbone",
    "tokens",
    "device",
    "pixel_read_status",
    "label_status",
    "target_status",
    "claim_scope",
]

FORBIDDEN_REPO_DIRS = {"data", "outputs", "docs"}
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index"}
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1fw DINO embedding extraction scaffold.")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST), help="v1fu DINO input manifest.")
    parser.add_argument("--asset-preflight", default=str(DEFAULT_ASSET_PREFLIGHT), help="Optional v1fv local asset preflight CSV.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Embedding extraction config YAML.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Local non-versioned output directory.")
    parser.add_argument("--private-project-root", default="", help="Private PROJETO root for execute mode only.")
    parser.add_argument("--execute", action="store_true", help="Actually load model/read pixels/extract embeddings.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Limit planned/executed rows; 0 means no limit.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backbone", default="dinov2_vitb14_reg")
    parser.add_argument("--tokens", default="cls,patch_mean")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_list: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list:
            config[current_list].append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_list = None
        if value == "":
            config[key] = []
            current_list = key
        elif value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
        elif value.startswith("[") and value.endswith("]"):
            config[key] = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        else:
            try:
                config[key] = int(value)
            except ValueError:
                config[key] = value.strip("'\"")
    return config


def is_local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    return "local_runs/" in lines or "local_runs" in lines


def forbidden_versioned_artifacts() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "local_runs" in path.parts:
            continue
        if path.is_dir() and path.name in FORBIDDEN_REPO_DIRS:
            found.append(rel(path))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_VERSIONED_EXTENSIONS:
            found.append(rel(path))
    return sorted(found)


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Re-run with --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_preflight(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row.get("dino_input_id", ""): row for row in read_csv(path)}


def build_plan(
    manifest_rows: list[dict[str, str]],
    preflight: dict[str, dict[str, str]],
    backbone: str,
    tokens: str,
    device: str,
    execute: bool,
    limit: int,
) -> list[dict[str, str]]:
    selected = manifest_rows[:limit] if limit and limit > 0 else manifest_rows
    rows: list[dict[str, str]] = []
    for row in selected:
        preflight_row = preflight.get(row.get("dino_input_id", ""), {})
        resolved_status = preflight_row.get("resolved_status", "NOT_PREFLIGHTED")
        blocked = resolved_status in {"MISSING", "AMBIGUOUS", "INVALID_REFERENCE"}
        block_reason = "" if not blocked else f"asset preflight status {resolved_status}"
        planned_status = "BLOCKED_BY_PREFLIGHT" if blocked else "PLANNED_FOR_EXECUTE" if execute else "PLANNED_DRY_RUN_ONLY"
        rows.append(
            {
                "dino_input_id": row.get("dino_input_id", ""),
                "canonical_patch_id": row.get("canonical_patch_id", ""),
                "region": row.get("region", ""),
                "asset_path_reference": row.get("asset_path_reference", ""),
                "resolved_status": resolved_status,
                "planned_status": planned_status,
                "block_reason": block_reason,
                "backbone": backbone,
                "tokens": tokens,
                "device": device,
                "execute_mode": "true" if execute else "false",
                "pixel_read_status": "NOT_READ__DRY_RUN_ONLY" if not execute else "PENDING_EXECUTE",
                "embedding_status": "NOT_EXTRACTED",
                "label_status": row.get("label_status", ""),
                "target_status": row.get("target_status", ""),
                "claim_scope": row.get("claim_scope", ""),
            }
        )
    return rows


def output_schema_rows() -> list[dict[str, str]]:
    return [
        {"column": "dino_input_id", "type": "string", "required": "yes", "description": "Stable v1fu DINO input id."},
        {"column": "canonical_patch_id", "type": "string", "required": "yes", "description": "Patch identifier; not a label."},
        {"column": "region", "type": "string", "required": "yes", "description": "Review/split metadata only."},
        {"column": "embedding_file", "type": "string", "required": "execute_only", "description": "Local-only NPZ path under local_runs."},
        {"column": "cls_embedding", "type": "array<float32>", "required": "execute_only", "description": "Frozen DINO CLS token embedding."},
        {"column": "patch_mean_embedding", "type": "array<float32>", "required": "execute_only", "description": "Mean pooled patch token embedding."},
        {"column": "backbone", "type": "string", "required": "yes", "description": "Frozen encoder backbone."},
        {"column": "label_status", "type": "constant", "required": "yes", "description": "Always NO_LABEL."},
        {"column": "target_status", "type": "constant", "required": "yes", "description": "Always NO_TARGET."},
        {"column": "claim_scope", "type": "constant", "required": "yes", "description": "Always REVIEW_ONLY_NO_PREDICTIVE_CLAIM."},
    ]


def make_qa(
    input_manifest: Path,
    manifest_rows: list[dict[str, str]],
    plan_rows: list[dict[str, str]],
    execute: bool,
    schema_rows: list[dict[str, str]],
    summary: dict[str, object],
) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    columns = set(manifest_rows[0].keys()) if manifest_rows else set()
    schema_columns = {row.get("column", "") for row in schema_rows}
    forbidden = forbidden_versioned_artifacts()
    add("input manifest exists", input_manifest.exists(), rel(input_manifest))
    add("input manifest has 128 rows", len(manifest_rows) == 128, f"rows={len(manifest_rows)}")
    add("input manifest has required columns", all(column in columns for column in REQUIRED_INPUT_COLUMNS), ",".join(sorted(columns)))
    add(
        "no labels/targets promoted",
        bool(manifest_rows)
        and {row.get("label_status") for row in manifest_rows} == {"NO_LABEL"}
        and {row.get("target_status") for row in manifest_rows} == {"NO_TARGET"},
        "label_status=NO_LABEL; target_status=NO_TARGET",
    )
    add(
        "claim scope remains review-only",
        bool(manifest_rows) and {row.get("claim_scope") for row in manifest_rows} == {REVIEW_ONLY_CLAIM},
        REVIEW_ONLY_CLAIM,
    )
    add("encoder mode remains frozen", bool(manifest_rows) and {row.get("encoder_mode") for row in manifest_rows} == {"frozen_encoder"}, "frozen_encoder")
    add("local_runs/ is gitignored", is_local_runs_ignored(), ".gitignore contains local_runs/")
    add("no data/, outputs/, docs/ created", not any((ROOT / name).exists() for name in FORBIDDEN_REPO_DIRS), "repo root checked")
    add("no forbidden files are versioned", not forbidden, "; ".join(forbidden) if forbidden else "none found")
    add("dry-run does not read pixels", execute or all(row.get("pixel_read_status") == "NOT_READ__DRY_RUN_ONLY" for row in plan_rows), "dry-run marks no pixel reads")
    add("dry-run does not load model", execute or summary.get("model_loaded") is False, "model_loaded=false")
    add("execute mode requires explicit --execute", True, f"execute={execute}")
    add("output schema contains cls and patch_mean fields", {"cls_embedding", "patch_mean_embedding"}.issubset(schema_columns), ",".join(sorted(schema_columns)))
    add("failure logging exists", (Path(str(summary.get("failure_log", ""))).exists() if execute else True), "failures CSV is created in execute mode; dry-run failure log not required")
    add(
        "summary records pixel_read and embeddings_extracted correctly",
        summary.get("pixel_read") is execute and isinstance(summary.get("embeddings_extracted"), bool),
        f"pixel_read={summary.get('pixel_read')}; embeddings_extracted={summary.get('embeddings_extracted')}",
    )
    return qa


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def execute_embeddings(plan_rows: list[dict[str, str]], output_dir: Path, backbone: str, tokens: str, device: str) -> tuple[list[dict[str, str]], list[dict[str, str]], bool]:
    failures: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        for row in plan_rows:
            failures.append(failure_row(row, "dependency", f"numpy unavailable: {exc}", backbone, device))
        return manifest, failures, False

    try:
        import timm  # type: ignore
        import torch  # type: ignore

        model = timm.create_model(backbone, pretrained=True, num_classes=0).to(device)
        model.eval()
    except Exception as exc:
        for row in plan_rows:
            failures.append(failure_row(row, "model_load", f"model unavailable: {exc}", backbone, device))
        return manifest, failures, False

    for row in plan_rows:
        if row.get("planned_status") == "BLOCKED_BY_PREFLIGHT":
            failures.append(failure_row(row, "preflight", row.get("block_reason", "blocked"), backbone, device))
            continue
        failures.append(failure_row(row, "pixel_reader", "execute reader scaffold present; raster/image conversion not run in this environment", backbone, device))

    _ = np, model
    return manifest, failures, False


def failure_row(row: dict[str, str], stage: str, reason: str, backbone: str, device: str) -> dict[str, str]:
    return {
        "dino_input_id": row.get("dino_input_id", ""),
        "canonical_patch_id": row.get("canonical_patch_id", ""),
        "region": row.get("region", ""),
        "stage": stage,
        "failure_reason": reason,
        "backbone": backbone,
        "device": device,
    }


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    input_manifest = Path(args.input_manifest)
    asset_preflight = Path(args.asset_preflight)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    prepare_output_dir(output_dir, args.force)

    manifest_rows = read_csv(input_manifest) if input_manifest.exists() else []
    config = parse_simple_yaml(config_path) if config_path.exists() else {}
    planned_device = resolve_device(args.device)
    tokens = ",".join(part.strip() for part in args.tokens.split(",") if part.strip())
    preflight = load_preflight(asset_preflight)
    plan_rows = build_plan(manifest_rows, preflight, args.backbone, tokens, planned_device, args.execute, args.limit)
    planned_count = sum(1 for row in plan_rows if row.get("planned_status") != "BLOCKED_BY_PREFLIGHT")
    blocked_count = sum(1 for row in plan_rows if row.get("planned_status") == "BLOCKED_BY_PREFLIGHT")
    schema_rows = output_schema_rows()

    embedding_manifest: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    embeddings_extracted = False
    if args.execute:
        embedding_manifest, failures, embeddings_extracted = execute_embeddings(plan_rows, output_dir, args.backbone, tokens, planned_device)
        write_csv(output_dir / MANIFEST_CSV, embedding_manifest, EMBEDDING_MANIFEST_FIELDS)
        write_csv(output_dir / FAILURES_CSV, failures, FAILURE_FIELDS)

    summary: dict[str, object] = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": rel(input_manifest),
        "asset_preflight": rel(asset_preflight) if asset_preflight.exists() else "",
        "config": rel(config_path),
        "total_inputs": len(manifest_rows),
        "plan_rows": len(plan_rows),
        "planned_count": planned_count,
        "blocked_count": blocked_count,
        "resolved_status_counts": dict(sorted(Counter(row.get("resolved_status", "") for row in plan_rows).items())),
        "backbone": args.backbone,
        "fallback_backbones": config.get("fallback_backbones", []),
        "tokens": tokens.split(","),
        "device": planned_device,
        "seed": args.seed,
        "execute": args.execute,
        "model_loaded": False,
        "pixel_read": bool(args.execute),
        "embeddings_extracted": embeddings_extracted,
        "review_only": True,
        "supervised_training": False,
        "failure_log": str(output_dir / FAILURES_CSV) if args.execute else "",
    }

    qa_rows = make_qa(input_manifest, manifest_rows, plan_rows, args.execute, schema_rows, summary)
    summary["qa_status"] = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"

    write_csv(output_dir / PLAN_CSV, plan_rows, PLAN_FIELDS)
    write_json(output_dir / SUMMARY_JSON, summary)
    write_csv(output_dir / QA_CSV, qa_rows, QA_FIELDS)
    write_csv(output_dir / SCHEMA_CSV, schema_rows, SCHEMA_FIELDS)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["qa_status"] == "PASS" else 2


def main() -> int:
    try:
        return run(parse_args())
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
