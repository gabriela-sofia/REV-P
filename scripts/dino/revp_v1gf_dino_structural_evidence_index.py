from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gf"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gf integrated DINO structural evidence index.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedding-manifest", default="")
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


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def default_manifest() -> Path:
    v1ge = ROOT / "local_runs" / "dino_embeddings" / "v1ge" / "dino_expanded_embedding_manifest_v1ge.csv"
    if v1ge.exists():
        return v1ge
    return ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    return "local_runs/" in lines or "local_runs" in lines


def by_id(rows: list[dict[str, str]], id_col: str = "dino_input_id") -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(id_col, "")].append(row)
    return grouped


def load_support() -> dict[str, dict[str, list[dict[str, str]]]]:
    base = ROOT / "local_runs" / "dino_embeddings"
    return {
        "v1ga_outliers": by_id(read_csv(base / "v1ga" / "structural_outliers.csv")),
        "v1ga_neighbors": by_id(read_csv(base / "v1ga" / "neighbor_persistence.csv")),
        "v1gb_medoids": by_id(read_csv(base / "v1gb" / "cluster_medoids.csv")),
        "v1gb_outliers": by_id(read_csv(base / "v1gb" / "outlier_taxonomy.csv")),
        "v1gb_visuals": by_id(read_csv(base / "v1gb" / "visual_review_manifest.csv"), "source_patch"),
        "v1gc_bridges_source": by_id(read_csv(base / "v1gc" / "graph_bridges.csv"), "source_id"),
        "v1gc_bridges_target": by_id(read_csv(base / "v1gc" / "graph_bridges.csv"), "target_id"),
        "v1gc_transition": by_id(read_csv(base / "v1gc" / "transitional_embeddings.csv")),
        "v1gd_rankings": by_id(read_csv(base / "v1gd" / "sensitivity_rankings.csv")),
    }


def first_value(rows: list[dict[str, str]], key: str, default: str = "") -> str:
    for row in rows:
        value = row.get(key, "")
        if value:
            return value
    return default


def review_priority(flags: list[str]) -> str:
    high = {"STRUCTURAL_OUTLIER", "OUTLIER", "CROSS_REGION_BRIDGE", "TRANSITIONAL", "UNSTABLE"}
    medium = {"MEDOID", "VISUAL_PANEL", "NEIGHBOR_STABILITY_REVIEW"}
    if any(flag in high for flag in flags):
        return "HIGH_REVIEW"
    if any(flag in medium for flag in flags):
        return "MEDIUM_REVIEW"
    return "LOW_REVIEW"


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    manifest = Path(args.embedding_manifest) if args.embedding_manifest else default_manifest()
    embeddings = [row for row in read_csv(manifest) if row.get("success") in {"SUCCESS", "SKIPPED_EXISTING"}]
    support = load_support()
    index_rows: list[dict[str, object]] = []
    for row in embeddings:
        dino_id = row.get("dino_input_id", "")
        patch_id = row.get("patch_id") or row.get("canonical_patch_id", "")
        flags: list[str] = []
        medoid_status = "MEDOID" if support["v1gb_medoids"].get(dino_id) else ""
        if medoid_status:
            flags.append("MEDOID")
        outlier_category = first_value(support["v1gb_outliers"].get(dino_id, []), "outlier_categories", "")
        if outlier_category and outlier_category != "NONE":
            flags.append("OUTLIER")
        bridge_role = "CROSS_REGION_BRIDGE" if support["v1gc_bridges_source"].get(dino_id) or support["v1gc_bridges_target"].get(dino_id) else ""
        if bridge_role:
            flags.append("CROSS_REGION_BRIDGE")
        transition = "TRANSITIONAL" if support["v1gc_transition"].get(dino_id) else ""
        if transition:
            flags.append("TRANSITIONAL")
        robustness = first_value(support["v1gd_rankings"].get(dino_id, []), "sensitivity_status", "")
        if robustness == "UNSTABLE":
            flags.append("UNSTABLE")
        visual_available = "YES" if support["v1gb_visuals"].get(patch_id) else "NO"
        if visual_available == "YES":
            flags.append("VISUAL_PANEL")
        priority = review_priority(flags)
        index_rows.append(
            {
                "patch_id": patch_id,
                "dino_input_id": dino_id,
                "region": row.get("region", ""),
                "embedding_path": row.get("embedding_path", ""),
                "embedding_dim": row.get("embedding_dim", ""),
                "cluster_diagnostics": medoid_status or "NON_MEDOID",
                "neighbor_stability": first_value(support["v1ga_neighbors"].get(dino_id, []), "neighbor_persistence_status", "NOT_AVAILABLE"),
                "medoid_status": medoid_status or "NOT_MEDOID",
                "outlier_category": outlier_category or "NOT_AVAILABLE",
                "geo_structural_role": "|".join(flag for flag in [bridge_role, transition] if flag) or "LOCAL_NODE",
                "perturbation_robustness": robustness or "NOT_AVAILABLE",
                "visual_panel_availability": visual_available,
                "qa_flags": "|".join(flags) if flags else "NONE",
                "review_priority": priority,
                "review_priority_is_not_label": "true",
                "human_review_required": "true",
                "label_status": "NO_LABEL",
                "target_status": "NO_TARGET",
                "claim_scope": REVIEW_ONLY_CLAIM,
            }
        )
    summary_rows = [{"review_priority": key, "count": value, "meaning": "triage_for_human_review_only"} for key, value in sorted(Counter(row["review_priority"] for row in index_rows).items())]
    guardrails = {"review_only": True, "supervised_training": False, "labels_created": False, "targets_created": False, "predictive_claims": False, "review_priority_is_not_label": True, "human_review_required": True, "clusters_are_classes": False, "multimodal_hold": True, "outputs_local_only": True}
    qa = make_qa(index_rows, manifest)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"
    write_csv(output_dir / "structural_evidence_index.csv", index_rows, ["patch_id", "dino_input_id", "region", "embedding_path", "embedding_dim", "cluster_diagnostics", "neighbor_stability", "medoid_status", "outlier_category", "geo_structural_role", "perturbation_robustness", "visual_panel_availability", "qa_flags", "review_priority", "review_priority_is_not_label", "human_review_required", "label_status", "target_status", "claim_scope"])
    write_csv(output_dir / "review_priority_summary.csv", summary_rows, ["review_priority", "count", "meaning"])
    write_json(output_dir / "methodological_guardrails.json", guardrails)
    write_csv(output_dir / "structural_evidence_qa.csv", qa, ["check", "status", "details"])
    write_json(output_dir / "structural_evidence_summary.json", {"phase": "v1gf", "created_utc": datetime.now(timezone.utc).isoformat(), "input_manifest": str(manifest), "indexed_patches": len(index_rows), "qa_status": qa_status, **guardrails})
    print(json.dumps({"phase": "v1gf", "indexed_patches": len(index_rows), "qa_status": qa_status, "review_priority_counts": Counter(row["review_priority"] for row in index_rows)}, ensure_ascii=False, indent=2, default=dict))
    return 0 if qa_status == "PASS" else 2


def make_qa(rows: list[dict[str, object]], manifest: Path) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    add("input manifest exists", manifest.exists(), str(manifest))
    add("index rows created", bool(rows), f"rows={len(rows)}")
    add("review priority deterministic", all(row.get("review_priority") in {"HIGH_REVIEW", "MEDIUM_REVIEW", "LOW_REVIEW"} for row in rows), "fixed vocabulary")
    add("review priority is not label", all(row.get("review_priority_is_not_label") == "true" for row in rows), "triage only")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    return qa


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    return gitignore.exists() and any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
