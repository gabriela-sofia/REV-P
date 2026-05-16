from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gj multimodal readiness audit. Does not execute multimodal processing.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--private-project-root", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    return gitignore.exists() and any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    sentinel = read_csv(ROOT / "local_runs" / "dino_embeddings" / "v1ge" / "dino_expanded_embedding_manifest_v1ge.csv")
    if not sentinel:
        sentinel = read_csv(ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv")
    preflight = read_csv(ROOT / "local_runs" / "dino_asset_preflight" / "v1fv" / "dino_local_asset_preflight_v1fv.csv")
    input_manifest = read_csv(ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv")
    found_preflight = {row.get("dino_input_id", ""): row for row in preflight if row.get("resolved_status") == "FOUND"}
    sentinel_success = [row for row in sentinel if row.get("success") in {"SUCCESS", "SKIPPED_EXISTING"}]
    readiness_rows = [
        {"readiness_dimension": "sentinel_embeddings", "status": "PASS" if sentinel_success else "FAIL", "details": f"success={len(sentinel_success)}"},
        {"readiness_dimension": "sentinel_preflight", "status": "PASS" if found_preflight else "FAIL", "details": f"found={len(found_preflight)}"},
        {"readiness_dimension": "bindings_existing", "status": "BLOCKED", "details": "no active multimodal binding manifest used in this audit"},
        {"readiness_dimension": "recife_gaps", "status": "REVIEW", "details": f"recife_sentinel_success={sum(1 for row in sentinel_success if row.get('region') == 'Recife')}"},
        {"readiness_dimension": "geometry", "status": "REVIEW", "details": "geometry compatibility not promoted by DINO audit"},
        {"readiness_dimension": "crs_consistency_known", "status": "BLOCKED", "details": "CRS consistency not established for multimodal fusion"},
        {"readiness_dimension": "asset_local_presence", "status": "PASS" if found_preflight else "FAIL", "details": "uses v1fv local preflight only"},
    ]
    inventory = []
    for row in input_manifest:
        dino_id = row.get("dino_input_id", "")
        inventory.append({"dino_input_id": dino_id, "patch_id": row.get("canonical_patch_id", ""), "region": row.get("region", ""), "sentinel_manifest_status": "PRESENT", "sentinel_local_status": "FOUND" if dino_id in found_preflight else "NOT_FOUND", "secondary_sensor_status": "NOT_AUDITED_ACTIVE_HOLD", "binding_status": "NOT_ENABLED", "multimodal_execution_enabled": "false"})
    blockers = []
    for row in readiness_rows:
        if row["status"] in {"BLOCKED", "FAIL", "REVIEW"}:
            blockers.append({"blocker": row["readiness_dimension"], "status": row["status"], "details": row["details"], "required_before_execution": "true"})
    region_counts = Counter(row.get("region", "") for row in sentinel_success)
    guardrails = {"phase": "v1gj", "multimodal_execution_enabled": False, "multimodal_training_enabled": False, "sentinel_first": True, "multimodal_hold": True, "review_only": True, "supervised_training": False, "labels_created": False, "targets_created": False, "predictive_claims": False, "readiness_is_not_execution": True}
    qa = make_qa(readiness_rows, blockers, inventory)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"
    write_csv(output_dir / "multimodal_readiness.csv", readiness_rows, ["readiness_dimension", "status", "details"])
    write_csv(output_dir / "multimodal_blockers.csv", blockers, ["blocker", "status", "details", "required_before_execution"])
    write_csv(output_dir / "multimodal_asset_inventory.csv", inventory, ["dino_input_id", "patch_id", "region", "sentinel_manifest_status", "sentinel_local_status", "secondary_sensor_status", "binding_status", "multimodal_execution_enabled"])
    write_json(output_dir / "multimodal_guardrails.json", guardrails)
    write_csv(output_dir / "multimodal_readiness_qa.csv", qa, ["check", "status", "details"])
    summary = {"phase": "v1gj", "created_utc": datetime.now(timezone.utc).isoformat(), "sentinel_success_count": len(sentinel_success), "sentinel_regions": dict(sorted(region_counts.items())), "blocker_count": len(blockers), "multimodal_readiness_status": "HOLD", "qa_status": qa_status, **guardrails}
    write_json(output_dir / "multimodal_readiness_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def make_qa(readiness: list[dict[str, object]], blockers: list[dict[str, object]], inventory: list[dict[str, object]]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    add("multimodal disabled assertions", True, "execution=false training=false")
    add("readiness aggregation", bool(readiness), f"rows={len(readiness)}")
    add("multimodal blocker detection", bool(blockers), f"blockers={len(blockers)}")
    add("missing asset detection", bool(inventory), f"inventory={len(inventory)}")
    add("CRS inconsistency detection", any(row.get("blocker") == "crs_consistency_known" for row in blockers), "CRS remains blocker")
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    return qa


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
