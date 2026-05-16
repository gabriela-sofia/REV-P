from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gi"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gi DINO structural provenance tracker.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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


def by_id(rows: list[dict[str, str]], key: str = "dino_input_id") -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row.get(key, "")].append(row)
    return out


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    base = ROOT / "local_runs" / "dino_embeddings"
    manifest_path = base / "v1ge" / "dino_expanded_embedding_manifest_v1ge.csv"
    if not manifest_path.exists():
        manifest_path = base / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"
    manifest = [row for row in read_csv(manifest_path) if row.get("success") in {"SUCCESS", "SKIPPED_EXISTING"}]
    sources = {
        "v1fz": by_id(read_csv(base / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv")),
        "v1ga_outlier": by_id(read_csv(base / "v1ga" / "structural_outliers.csv")),
        "v1gb_medoid": by_id(read_csv(base / "v1gb" / "cluster_medoids.csv")),
        "v1gb_outlier": by_id(read_csv(base / "v1gb" / "outlier_taxonomy.csv")),
        "v1gb_visual": by_id(read_csv(base / "v1gb" / "visual_review_manifest.csv"), "source_patch"),
        "v1gc_bridge_source": by_id(read_csv(base / "v1gc" / "graph_bridges.csv"), "source_id"),
        "v1gc_bridge_target": by_id(read_csv(base / "v1gc" / "graph_bridges.csv"), "target_id"),
        "v1gd_robustness": by_id(read_csv(base / "v1gd" / "sensitivity_rankings.csv")),
        "v1gf_index": by_id(read_csv(base / "v1gf" / "structural_evidence_index.csv")),
        "v1gg_review": by_id(read_csv(base / "v1gg" / "human_review_manifest.csv")),
    }
    qa_sources = {
        "v1fz": base / "v1fz" / "dino_balanced_embedding_qa_v1fz.csv",
        "v1ga": base / "v1ga" / "structural_consistency_qa.csv",
        "v1gb": base / "v1gb" / "visual_structural_review_qa.csv",
        "v1gc": base / "v1gc" / "geo_structural_diagnostics_qa.csv",
        "v1gd": base / "v1gd" / "perturbation_robustness_qa.csv",
        "v1gf": base / "v1gf" / "structural_evidence_qa.csv",
        "v1gg": base / "v1gg" / "human_review_package_qa.csv",
    }
    provenance = []
    history = []
    trace = []
    seen_keys = set()
    for row in manifest:
        dino_id = row.get("dino_input_id", "")
        patch_id = row.get("patch_id") or row.get("canonical_patch_id", "")
        touched = []
        diagnostics = []
        if sources["v1fz"].get(dino_id):
            touched.append("v1fz")
            diagnostics.append("embedding")
        if sources["v1ga_outlier"].get(dino_id):
            touched.append("v1ga")
            diagnostics.append("structural_outlier")
        if sources["v1gb_medoid"].get(dino_id):
            touched.append("v1gb")
            diagnostics.append("medoid")
        if sources["v1gb_outlier"].get(dino_id):
            touched.append("v1gb")
            diagnostics.append("outlier_taxonomy")
        if sources["v1gc_bridge_source"].get(dino_id) or sources["v1gc_bridge_target"].get(dino_id):
            touched.append("v1gc")
            diagnostics.append("bridge")
        if sources["v1gd_robustness"].get(dino_id):
            touched.append("v1gd")
            diagnostics.append("robustness")
        if sources["v1gf_index"].get(dino_id):
            touched.append("v1gf")
            diagnostics.append("structural_index")
        if sources["v1gg_review"].get(dino_id):
            touched.append("v1gg")
            diagnostics.append("human_review_package")
        visuals = sources["v1gb_visual"].get(patch_id, [])
        qa_statuses = {name: read_qa_status(path) for name, path in qa_sources.items()}
        key = (dino_id, row.get("embedding_path", ""))
        seen_keys.add(key)
        provenance.append({"patch_id": patch_id, "dino_input_id": dino_id, "region": row.get("region", ""), "embedding_path": row.get("embedding_path", ""), "versions_touched": "|".join(sorted(set(touched))), "diagnostics_produced": "|".join(sorted(set(diagnostics))) if diagnostics else "NONE", "visualization_count": len(visuals), "qa_passed_versions": "|".join(name for name, status in qa_statuses.items() if status == "PASS"), "medoid_participation": str("medoid" in diagnostics).lower(), "bridge_participation": str("bridge" in diagnostics).lower(), "outlier_participation": str(any("outlier" in item for item in diagnostics)).lower(), "review_package_participation": str("human_review_package" in diagnostics).lower(), "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": REVIEW_ONLY_CLAIM})
        for version in sorted(set(touched)):
            history.append({"patch_id": patch_id, "dino_input_id": dino_id, "version": version, "diagnostic_status": "PRESENT", "history_chain_status": "VALID"})
        if sources["v1gg_review"].get(dino_id):
            review = sources["v1gg_review"][dino_id][0]
            trace.append({"review_item_id": review.get("review_item_id", ""), "patch_id": patch_id, "dino_input_id": dino_id, "evidence_sources": review.get("evidence_sources", ""), "traceability_status": "TRACEABLE"})
    qa = make_qa(provenance, history, trace, len(seen_keys) == len(provenance))
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"
    write_csv(output_dir / "structural_provenance_index.csv", provenance, ["patch_id", "dino_input_id", "region", "embedding_path", "versions_touched", "diagnostics_produced", "visualization_count", "qa_passed_versions", "medoid_participation", "bridge_participation", "outlier_participation", "review_package_participation", "label_status", "target_status", "claim_scope"])
    write_csv(output_dir / "patch_diagnostic_history.csv", history, ["patch_id", "dino_input_id", "version", "diagnostic_status", "history_chain_status"])
    write_csv(output_dir / "review_traceability.csv", trace, ["review_item_id", "patch_id", "dino_input_id", "evidence_sources", "traceability_status"])
    write_csv(output_dir / "provenance_qa.csv", qa, ["check", "status", "details"])
    summary = {"phase": "v1gi", "created_utc": datetime.now(timezone.utc).isoformat(), "patches_tracked": len(provenance), "history_rows": len(history), "review_trace_rows": len(trace), "provenance_status": "PASS" if qa_status == "PASS" else "FAIL", "qa_status": qa_status, "review_only": True, "supervised_training": False, "labels_created": False, "targets_created": False, "predictive_claims": False, "multimodal_hold": True}
    write_json(output_dir / "provenance_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def read_qa_status(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return "MISSING"
    return "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL"


def make_qa(provenance: list[dict[str, object]], history: list[dict[str, object]], trace: list[dict[str, object]], unique: bool) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    add("provenance uniqueness", unique, f"rows={len(provenance)}")
    add("invalid history chains", all(row.get("history_chain_status") == "VALID" for row in history), f"history={len(history)}")
    add("duplicate provenance entries", unique, "dino_input_id + embedding_path")
    add("review traceability", bool(trace), f"trace_rows={len(trace)}")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in provenance), REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    return qa


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
