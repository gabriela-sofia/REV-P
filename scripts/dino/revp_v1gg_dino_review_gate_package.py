from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "local_runs" / "dino_embeddings" / "v1gf" / "structural_evidence_index.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gg"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gg local-only DINO human review package.")
    parser.add_argument("--structural-index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
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


def visual_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path in [
        ROOT / "local_runs" / "dino_embeddings" / "v1gb" / "visual_review_manifest.csv",
        ROOT / "local_runs" / "dino_embeddings" / "v1gc" / "visual_review_manifest.csv",
        ROOT / "local_runs" / "dino_embeddings" / "v1gd" / "visual_review_manifest.csv",
    ]:
        for row in read_csv(path):
            key = row.get("source_patch") or row.get("dino_input_id") or row.get("panel_type", "")
            if key and row.get("image_path"):
                lookup.setdefault(key, row["image_path"])
    return lookup


def reason(row: dict[str, str]) -> tuple[str, str]:
    flags = row.get("qa_flags", "")
    if "OUTLIER" in flags:
        return "structural_outlier", "Inspect whether the outlier pattern is visually coherent or a local artifact."
    if "CROSS_REGION_BRIDGE" in flags:
        return "cross_region_bridge", "Inspect whether this cross-region neighbor relation is structurally plausible."
    if "TRANSITIONAL" in flags:
        return "transition_candidate", "Inspect whether the transition role is stable enough for later manual discussion."
    if "UNSTABLE" in flags:
        return "perturbation_sensitive", "Inspect whether perturbation sensitivity reflects image quality or structural ambiguity."
    if "MEDOID" in flags:
        return "cluster_medoid", "Inspect whether the structural representative is suitable as a review anchor."
    if row.get("visual_panel_availability") == "YES":
        return "visual_neighbor_review", "Inspect nearest-neighbor panels for structural consistency."
    return "region_representative", "Inspect as a baseline regional example."


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    index_path = Path(args.structural_index)
    index_rows = read_csv(index_path)
    visuals = visual_lookup()
    ordered = sorted(index_rows, key=lambda row: {"HIGH_REVIEW": 0, "MEDIUM_REVIEW": 1, "LOW_REVIEW": 2}.get(row.get("review_priority", ""), 9))
    manifest: list[dict[str, object]] = []
    for idx, row in enumerate(ordered, start=1):
        review_reason, question = reason(row)
        visual_path = visuals.get(row.get("patch_id", ""), "") or visuals.get(row.get("dino_input_id", ""), "")
        manifest.append(
            {
                "review_item_id": f"V1GG_REVIEW_{idx:04d}",
                "patch_id": row.get("patch_id", ""),
                "dino_input_id": row.get("dino_input_id", ""),
                "region": row.get("region", ""),
                "review_reason": review_reason,
                "evidence_sources": "v1gf_structural_index|v1gb_visual|v1gc_geo_structural|v1gd_robustness",
                "local_visual_path": visual_path,
                "local_embedding_path": row.get("embedding_path", ""),
                "notes_empty_for_human": "",
                "suggested_review_question": question,
                "review_priority": row.get("review_priority", ""),
                "review_priority_is_not_label": "true",
                "human_review_required": "true",
                "label_status": "NO_LABEL",
                "target_status": "NO_TARGET",
                "claim_scope": REVIEW_ONLY_CLAIM,
            }
        )
    batches = []
    for start in range(0, len(manifest), max(args.batch_size, 1)):
        batch = manifest[start : start + max(args.batch_size, 1)]
        batches.append({"batch_id": f"V1GG_BATCH_{len(batches)+1:03d}", "item_count": len(batch), "review_item_ids": "|".join(str(row["review_item_id"]) for row in batch), "batch_status": "READY_FOR_MANUAL_REVIEW"})
    qa = make_qa(manifest, index_path)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"
    write_csv(output_dir / "human_review_manifest.csv", manifest, ["review_item_id", "patch_id", "dino_input_id", "region", "review_reason", "evidence_sources", "local_visual_path", "local_embedding_path", "notes_empty_for_human", "suggested_review_question", "review_priority", "review_priority_is_not_label", "human_review_required", "label_status", "target_status", "claim_scope"])
    write_csv(output_dir / "review_batches.csv", batches, ["batch_id", "item_count", "review_item_ids", "batch_status"])
    write_csv(output_dir / "human_review_package_qa.csv", qa, ["check", "status", "details"])
    readme = """# DINO Sentinel-first human review package

This local package supports manual review of structural DINO embedding diagnostics. It is review-only.

- `review_priority` is triage for human inspection, not a scientific label.
- No labels, targets, supervised training, or predictive claims are created.
- Visual paths are local references only; raw rasters are not copied.
- Human notes should be added only after manual inspection.
"""
    (output_dir / "review_readme.md").write_text(readme, encoding="utf-8")
    summary = {"phase": "v1gg", "created_utc": datetime.now(timezone.utc).isoformat(), "review_items": len(manifest), "review_batches": len(batches), "reason_counts": dict(Counter(row["review_reason"] for row in manifest)), "qa_status": qa_status, "review_only": True, "supervised_training": False, "labels_created": False, "targets_created": False, "predictive_claims": False, "review_priority_is_not_label": True, "human_review_required": True, "outputs_local_only": True}
    write_json(output_dir / "human_review_package_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def make_qa(rows: list[dict[str, object]], index_path: Path) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    add("structural index exists", index_path.exists(), str(index_path))
    add("review manifest created", bool(rows), f"rows={len(rows)}")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("review priority is not label", all(row.get("review_priority_is_not_label") == "true" for row in rows), "triage only")
    add("human review required", all(row.get("human_review_required") == "true" for row in rows), "manual review required")
    add("raw rasters not copied", all(not str(row.get("local_visual_path", "")).lower().endswith((".tif", ".tiff")) for row in rows), "visual references only")
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    return qa


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
