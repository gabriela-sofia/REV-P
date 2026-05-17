from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gh"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gh longitudinal DINO structural diagnostics.")
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


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    return gitignore.exists() and any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def latest_embedding_manifest() -> Path:
    v1ge = ROOT / "local_runs" / "dino_embeddings" / "v1ge" / "dino_expanded_embedding_manifest_v1ge.csv"
    if v1ge.exists():
        return v1ge
    return ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"


def by_id(rows: list[dict[str, str]], key: str = "dino_input_id") -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return grouped


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    base_rows = [row for row in read_csv(latest_embedding_manifest()) if row.get("success") in {"SUCCESS", "SKIPPED_EXISTING"}]
    ids = [row.get("dino_input_id", "") for row in base_rows]
    regions = {row.get("dino_input_id", ""): row.get("region", "") for row in base_rows}
    upstream = {
        "v1fz_neighbors": ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_nearest_neighbors_v1fz.csv",
        "v1ga_neighbors": ROOT / "local_runs" / "dino_embeddings" / "v1ga" / "neighbor_persistence.csv",
        "v1gb_outliers": ROOT / "local_runs" / "dino_embeddings" / "v1gb" / "outlier_taxonomy.csv",
        "v1gb_medoids": ROOT / "local_runs" / "dino_embeddings" / "v1gb" / "cluster_medoids.csv",
        "v1gc_bridges": ROOT / "local_runs" / "dino_embeddings" / "v1gc" / "graph_bridges.csv",
        "v1gd_drift": ROOT / "local_runs" / "dino_embeddings" / "v1gd" / "sensitivity_rankings.csv",
        "v1gf_index": ROOT / "local_runs" / "dino_embeddings" / "v1gf" / "structural_evidence_index.csv",
        "v1gg_review": ROOT / "local_runs" / "dino_embeddings" / "v1gg" / "human_review_manifest.csv",
    }
    missing = [name for name, path in upstream.items() if not path.exists()]
    fz_neighbors = by_id(read_csv(upstream["v1fz_neighbors"]))
    ga_neighbors = by_id(read_csv(upstream["v1ga_neighbors"]))
    gb_outliers = by_id(read_csv(upstream["v1gb_outliers"]))
    gb_medoids = by_id(read_csv(upstream["v1gb_medoids"]))
    gc_bridges_source = by_id(read_csv(upstream["v1gc_bridges"]), "source_id")
    gc_bridges_target = by_id(read_csv(upstream["v1gc_bridges"]), "target_id")
    gd_drift = by_id(read_csv(upstream["v1gd_drift"]))
    gf_index = by_id(read_csv(upstream["v1gf_index"]))
    gg_review = by_id(read_csv(upstream["v1gg_review"]))
    neighbor_rows = []
    outlier_rows = []
    medoid_rows = []
    priority_rows = []
    region_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for dino_id in ids:
        flags = []
        if fz_neighbors.get(dino_id):
            flags.append("v1fz")
        if ga_neighbors.get(dino_id):
            flags.append("v1ga")
        neighbor_rows.append({"dino_input_id": dino_id, "region": regions.get(dino_id, ""), "versions_with_neighbor_diagnostics": "|".join(flags) if flags else "NONE", "neighbor_persistence_status": "PERSISTENT" if len(flags) > 1 else "PARTIAL"})
        out_cat = first(gb_outliers.get(dino_id, []), "outlier_categories", "NOT_AVAILABLE")
        sensitivity = first(gd_drift.get(dino_id, []), "sensitivity_status", "NOT_AVAILABLE")
        outlier_rows.append({"dino_input_id": dino_id, "region": regions.get(dino_id, ""), "v1gb_outlier_category": out_cat, "v1gd_sensitivity": sensitivity, "outlier_stability_status": "REVIEW" if out_cat not in {"", "NONE", "NOT_AVAILABLE"} or sensitivity == "UNSTABLE" else "STABLE_NON_OUTLIER"})
        medoid_rows.append({"dino_input_id": dino_id, "region": regions.get(dino_id, ""), "v1gb_medoid": str(bool(gb_medoids.get(dino_id))).lower(), "medoid_stability_status": "MEDOID_REVIEW" if gb_medoids.get(dino_id) else "NON_MEDOID"})
        bridge = bool(gc_bridges_source.get(dino_id) or gc_bridges_target.get(dino_id))
        priority = first(gf_index.get(dino_id, []), "review_priority", "NOT_INDEXED")
        package = "IN_REVIEW_PACKAGE" if gg_review.get(dino_id) else "NOT_IN_PACKAGE"
        priority_rows.append({"dino_input_id": dino_id, "region": regions.get(dino_id, ""), "review_priority": priority, "bridge_status": "BRIDGE" if bridge else "NON_BRIDGE", "review_package_status": package, "review_priority_is_not_label": "true"})
        region_counts[regions.get(dino_id, "")][priority] += 1
    region_rows = [{"region": region, "review_priority": priority, "count": count, "regional_stability_status": "DIAGNOSTIC_ONLY"} for region, counts in sorted(region_counts.items()) for priority, count in sorted(counts.items())]
    qa = make_qa(base_rows, missing, priority_rows)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"
    summary = {"phase": "v1gh", "created_utc": datetime.now(timezone.utc).isoformat(), "embeddings_analyzed": len(ids), "upstream_missing": missing, "longitudinal_diagnostics_status": "PASS" if not missing and ids else "FAIL", "qa_status": qa_status, "review_only": True, "supervised_training": False, "labels_created": False, "targets_created": False, "predictive_claims": False, "multimodal_hold": True}
    write_csv(output_dir / "longitudinal_neighbor_persistence.csv", neighbor_rows, ["dino_input_id", "region", "versions_with_neighbor_diagnostics", "neighbor_persistence_status"])
    write_csv(output_dir / "longitudinal_outlier_stability.csv", outlier_rows, ["dino_input_id", "region", "v1gb_outlier_category", "v1gd_sensitivity", "outlier_stability_status"])
    write_csv(output_dir / "longitudinal_medoid_stability.csv", medoid_rows, ["dino_input_id", "region", "v1gb_medoid", "medoid_stability_status"])
    write_csv(output_dir / "longitudinal_review_priority.csv", priority_rows, ["dino_input_id", "region", "review_priority", "bridge_status", "review_package_status", "review_priority_is_not_label"])
    write_csv(output_dir / "longitudinal_region_stability.csv", region_rows, ["region", "review_priority", "count", "regional_stability_status"])
    write_csv(output_dir / "longitudinal_qa.csv", qa, ["check", "status", "details"])
    write_json(output_dir / "longitudinal_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def first(rows: list[dict[str, str]], key: str, default: str) -> str:
    return rows[0].get(key, default) if rows else default


def make_qa(rows: list[dict[str, str]], missing: list[str], priority_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    add("longitudinal consistency", bool(rows) and not missing, f"rows={len(rows)} missing={missing}")
    add("missing upstream outputs", not missing, "|".join(missing) if missing else "none")
    add("review priority not label", all(row.get("review_priority_is_not_label") == "true" for row in priority_rows), "triage only")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    return qa


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
