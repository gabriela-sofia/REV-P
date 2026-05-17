from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1go"
STAGES = {
    "v1fw": {"script": "revp_v1fw_dino_embedding_extraction_scaffold.py", "deps": [], "default_args": ["--force"], "multimodal": False},
    "v1fx": {"script": "revp_v1fx_dino_smoke_embedding_execution.py", "deps": ["v1fw"], "default_args": ["--execute", "--limit", "5", "--force", "--allow-cpu", "--skip-model-if-unavailable"], "multimodal": False},
    "v1fy": {"script": "revp_v1fy_dino_embedding_corpus_analysis.py", "deps": ["v1fx"], "default_args": ["--force"], "multimodal": False},
    "v1fz": {"script": "revp_v1fz_dino_balanced_embedding_corpus.py", "deps": ["v1fx"], "default_args": ["--execute", "--per-region-limit", "2", "--force", "--allow-cpu", "--skip-model-if-unavailable"], "multimodal": False},
    "v1ga": {"script": "revp_v1ga_dino_embedding_structural_consistency_analysis.py", "deps": ["v1fz"], "default_args": ["--force"], "multimodal": False},
    "v1gb": {"script": "revp_v1gb_dino_embedding_local_visual_structural_review.py", "deps": ["v1fz"], "default_args": ["--force"], "multimodal": False},
    "v1gc": {"script": "revp_v1gc_dino_embedding_geo_structural_diagnostics.py", "deps": ["v1fz"], "default_args": ["--force"], "multimodal": False},
    "v1gd": {"script": "revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py", "deps": ["v1fz"], "default_args": ["--force", "--allow-cpu", "--skip-model-if-unavailable"], "multimodal": False},
    "v1ge": {"script": "revp_v1ge_dino_expanded_sentinel_embedding_corpus.py", "deps": ["v1fz"], "default_args": ["--execute", "--limit", "12", "--force", "--allow-cpu"], "multimodal": False},
    "v1gf": {"script": "revp_v1gf_dino_structural_evidence_index.py", "deps": ["v1ga", "v1gb", "v1gc", "v1gd"], "default_args": ["--force"], "multimodal": False},
    "v1gg": {"script": "revp_v1gg_dino_human_review_package.py", "deps": ["v1gf"], "default_args": ["--force"], "multimodal": False},
    "v1gh": {"script": "revp_v1gh_dino_longitudinal_structural_diagnostics.py", "deps": ["v1gg"], "default_args": ["--force"], "multimodal": False},
    "v1gi": {"script": "revp_v1gi_dino_structural_provenance_tracker.py", "deps": ["v1gg"], "default_args": ["--force"], "multimodal": False},
    "v1gj": {"script": "revp_v1gj_multimodal_readiness_audit.py", "deps": ["v1gi"], "default_args": ["--force"], "multimodal": False, "multimodal_execution_enabled": False},
    "v1gk": {"script": "revp_v1gk_dino_pipeline_reproducibility_audit.py", "deps": ["v1gj"], "default_args": ["--force"], "multimodal": False},
    "v1gn": {"script": "revp_v1gn_dino_execution_health_monitor.py", "deps": ["v1gk"], "default_args": ["--force"], "multimodal": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1go DINO lightweight pipeline orchestrator.")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


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


def selected_stages(stage: str) -> list[str]:
    if stage == "all":
        return list(STAGES)
    if stage not in STAGES:
        raise ValueError(f"Invalid stage: {stage}")
    return [stage]


def detect_cycle() -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in STAGES[node]["deps"]:
            if dep in STAGES and visit(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(stage) for stage in STAGES)


def dependency_rows() -> list[dict[str, object]]:
    rows = []
    for stage, meta in STAGES.items():
        for dep in meta["deps"] or [""]:
            rows.append({"stage": stage, "dependency": dep, "script": meta["script"], "multimodal_execution_enabled": str(bool(meta.get("multimodal_execution_enabled", False))).lower()})
    return rows


def validation(stage_list: list[str]) -> list[dict[str, object]]:
    rows = []
    cycle = detect_cycle()
    for stage in stage_list:
        meta = STAGES[stage]
        script = ROOT / "scripts" / "dino" / str(meta["script"])
        missing = [dep for dep in meta["deps"] if dep not in STAGES]
        upstream_missing = [dep for dep in meta["deps"] if not (ROOT / "local_runs" / "dino_embeddings" / dep).exists()]
        rows.append(
            {
                "stage": stage,
                "script": script.as_posix(),
                "script_exists": str(script.exists()).lower(),
                "dependencies": "|".join(meta["deps"]),
                "missing_dependency_definitions": "|".join(missing),
                "missing_upstream_outputs": "|".join(upstream_missing),
                "cycle_detected": str(cycle).lower(),
                "multimodal_disabled": str(not bool(meta.get("multimodal_execution_enabled", False))).lower(),
                "validation_status": "PASS" if script.exists() and not missing and not cycle and not bool(meta.get("multimodal_execution_enabled", False)) else "FAIL",
            }
        )
    return rows


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        stage_list = selected_stages(args.stage)
    except ValueError as exc:
        write_outputs(output_dir, [], [], "FAIL", str(exc), args)
        print(str(exc), file=sys.stderr)
        return 2
    validation_rows = validation(stage_list)
    execution_rows = []
    if not args.validate_only:
        for stage in stage_list:
            meta = STAGES[stage]
            command = [sys.executable, str(ROOT / "scripts" / "dino" / str(meta["script"])), *list(meta["default_args"])]
            if args.dry_run:
                execution_rows.append({"stage": stage, "command": " ".join(command), "execution_status": "DRY_RUN", "returncode": ""})
            else:
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
                execution_rows.append({"stage": stage, "command": " ".join(command), "execution_status": "PASS" if result.returncode == 0 else "FAIL", "returncode": result.returncode})
                if result.returncode != 0:
                    break
    status = "PASS" if all(row["validation_status"] == "PASS" for row in validation_rows) and all(row.get("execution_status") in {"DRY_RUN", "PASS"} for row in execution_rows) else "FAIL"
    write_outputs(output_dir, validation_rows, execution_rows, status, "", args)
    print(json.dumps({"phase": "v1go", "stage": args.stage, "dry_run": args.dry_run, "validate_only": args.validate_only, "orchestration_status": status, "multimodal_execution_enabled": False}, indent=2))
    return 0 if status == "PASS" else 2


def write_outputs(output_dir: Path, validation_rows: list[dict[str, object]], execution_rows: list[dict[str, object]], status: str, error: str, args: argparse.Namespace) -> None:
    registry = {
        "phase": "v1go",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_stage": args.stage,
        "dry_run": bool(args.dry_run),
        "validate_only": bool(args.validate_only),
        "orchestration_status": status,
        "error": error,
        "stages": STAGES,
        "review_only": True,
        "supervised_training": False,
        "labels_created": False,
        "targets_created": False,
        "predictive_claims": False,
        "multimodal_execution_enabled": False,
        "multimodal_training_enabled": False,
    }
    write_json(output_dir / "pipeline_execution_registry.json", registry)
    write_csv(output_dir / "pipeline_dependency_graph.csv", dependency_rows(), ["stage", "dependency", "script", "multimodal_execution_enabled"])
    write_csv(output_dir / "pipeline_validation_report.csv", validation_rows + execution_rows, ["stage", "script", "script_exists", "dependencies", "missing_dependency_definitions", "missing_upstream_outputs", "cycle_detected", "multimodal_disabled", "validation_status", "command", "execution_status", "returncode"])


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
