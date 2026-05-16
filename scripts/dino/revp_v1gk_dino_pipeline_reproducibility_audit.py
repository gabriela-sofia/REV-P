from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gk"
PHASES = ["v1fw", "v1fx", "v1fy", "v1fz", "v1ga", "v1gb", "v1gc", "v1gd", "v1ge", "v1gf", "v1gg", "v1gh", "v1gi", "v1gj", "v1gk"]
SCRIPT_BY_PHASE = {
    "v1fw": "revp_v1fw_dino_embedding_extraction_scaffold.py",
    "v1fx": "revp_v1fx_dino_smoke_embedding_execution.py",
    "v1fy": "revp_v1fy_dino_embedding_corpus_analysis.py",
    "v1fz": "revp_v1fz_dino_balanced_embedding_corpus.py",
    "v1ga": "revp_v1ga_dino_embedding_structural_consistency_analysis.py",
    "v1gb": "revp_v1gb_dino_embedding_local_visual_structural_review.py",
    "v1gc": "revp_v1gc_dino_embedding_geo_structural_diagnostics.py",
    "v1gd": "revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py",
    "v1ge": "revp_v1ge_dino_expanded_sentinel_embedding_corpus.py",
    "v1gf": "revp_v1gf_dino_structural_evidence_index.py",
    "v1gg": "revp_v1gg_dino_human_review_package.py",
    "v1gh": "revp_v1gh_dino_longitudinal_structural_diagnostics.py",
    "v1gi": "revp_v1gi_dino_structural_provenance_tracker.py",
    "v1gj": "revp_v1gj_multimodal_readiness_audit.py",
    "v1gk": "revp_v1gk_dino_pipeline_reproducibility_audit.py",
}
REQUIRED_CONFIGS = [
    "configs/dino_embedding_extraction.example.yaml",
    "configs/dino_review_only.yaml",
    "configs/project_state.yaml",
]
REQUIRED_DOCS = [
    "docs/dino_sentinel_embedding_protocol.md",
    "docs/dino_sentinel_scientific_evidence_summary.md",
    "docs/dino_command_registry.md",
]
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index", ".tif", ".tiff"}
GUARDRAIL_TERMS = [
    "review_only",
    "supervised_training=false",
    "labels_created=false",
    "targets_created=false",
    "predictive_claims=false",
    "multimodal_execution_enabled=false",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gk DINO pipeline reproducibility audit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
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


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    return gitignore.exists() and any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def forbidden_versioned_artifacts() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "local_runs" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_VERSIONED_EXTENSIONS:
            found.append(path.as_posix())
        if path.is_dir() and path.name in {"data", "outputs"}:
            found.append(path.as_posix())
    return sorted(found)


def file_has_any(path: Path, terms: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return all(term.lower() in text for term in terms)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    rows: list[dict[str, object]] = []
    for phase in PHASES:
        script = ROOT / "scripts" / "dino" / SCRIPT_BY_PHASE[phase]
        local_output = ROOT / "local_runs" / "dino_embeddings" / phase
        rows.append(
            {
                "audit_item": phase,
                "item_type": "script_and_local_output",
                "path": script.as_posix(),
                "exists": str(script.exists()).lower(),
                "local_output_path": local_output.as_posix(),
                "local_output_exists": str(local_output.exists()).lower(),
                "status": "PASS" if script.exists() else "FAIL",
                "details": "local output optional for reproducibility audit",
            }
        )
    for config in REQUIRED_CONFIGS:
        path = ROOT / config
        rows.append({"audit_item": Path(config).name, "item_type": "config", "path": config, "exists": str(path.exists()).lower(), "local_output_path": "", "local_output_exists": "", "status": "PASS" if path.exists() else "FAIL", "details": "required config"})
    for doc in REQUIRED_DOCS:
        path = ROOT / doc
        rows.append({"audit_item": Path(doc).name, "item_type": "doc", "path": doc, "exists": str(path.exists()).lower(), "local_output_path": "", "local_output_exists": "", "status": "PASS" if path.exists() else "FAIL", "details": "required versioned doc"})
    protocol = ROOT / "docs" / "dino_sentinel_embedding_protocol.md"
    guardrails_present = file_has_any(protocol, ["review-only", "not a supervised", "multimodal_execution_enabled=false"])
    rows.append({"audit_item": "methodological_guardrails", "item_type": "guardrails", "path": protocol.as_posix(), "exists": str(protocol.exists()).lower(), "local_output_path": "", "local_output_exists": "", "status": "PASS" if guardrails_present else "FAIL", "details": "review-only/no supervised/no multimodal execution language"})
    seeds_flags_present = all((ROOT / "scripts" / "dino" / SCRIPT_BY_PHASE[phase]).exists() and "--seed" in (ROOT / "scripts" / "dino" / SCRIPT_BY_PHASE[phase]).read_text(encoding="utf-8", errors="ignore") for phase in ["v1fw", "v1fy", "v1fz", "v1ga", "v1gb", "v1gc", "v1gd", "v1ge"])
    rows.append({"audit_item": "seeds_and_flags", "item_type": "reproducibility", "path": "scripts/dino", "exists": "true", "local_output_path": "", "local_output_exists": "", "status": "PASS" if seeds_flags_present else "FAIL", "details": "--seed or equivalent deterministic flags present in expected scripts"})
    forbidden = forbidden_versioned_artifacts()
    qa = make_qa(rows, forbidden)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"
    summary = {
        "phase": "v1gk",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_rows": len(rows),
        "script_count": len(PHASES),
        "docs_checked": len(REQUIRED_DOCS),
        "configs_checked": len(REQUIRED_CONFIGS),
        "local_runs_ignored": local_runs_ignored(),
        "forbidden_versioned_artifacts": forbidden,
        "qa_status": qa_status,
        "review_only": True,
        "supervised_training": False,
        "labels_created": False,
        "targets_created": False,
        "predictive_claims": False,
        "multimodal_execution_enabled": False,
        "multimodal_training_enabled": False,
        "sentinel_first": True,
    }
    write_csv(output_dir / "reproducibility_audit.csv", rows, ["audit_item", "item_type", "path", "exists", "local_output_path", "local_output_exists", "status", "details"])
    write_csv(output_dir / "reproducibility_qa.csv", qa, ["check", "status", "details"])
    write_json(output_dir / "reproducibility_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def make_qa(rows: list[dict[str, object]], forbidden: list[str]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    add("scripts expected exist", all(row["status"] == "PASS" for row in rows if row["item_type"] == "script_and_local_output"), "v1fw-v1gk scripts")
    add("configs expected exist", all(row["status"] == "PASS" for row in rows if row["item_type"] == "config"), "required configs")
    add("docs expected exist", all(row["status"] == "PASS" for row in rows if row["item_type"] == "doc"), "protocol summary registry")
    add("local_runs gitignored", local_runs_ignored(), ".gitignore protects runtime outputs")
    add("no forbidden heavy artifacts versioned", not forbidden, "|".join(forbidden) if forbidden else "none")
    add("seeds flags documented", any(row["audit_item"] == "seeds_and_flags" and row["status"] == "PASS" for row in rows), "--seed and execution flags")
    add("guardrails methodological present", any(row["audit_item"] == "methodological_guardrails" and row["status"] == "PASS" for row in rows), "review-only guardrails")
    return qa


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
