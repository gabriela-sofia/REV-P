from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1GE = ROOT / "scripts" / "dino" / "revp_v1ge_dino_expanded_sentinel_embedding_corpus.py"
V1GF = ROOT / "scripts" / "dino" / "revp_v1gf_dino_structural_evidence_index.py"
V1GG = ROOT / "scripts" / "dino" / "revp_v1gg_dino_human_review_package.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest_rows = []
    preflight_rows = []
    regions = ["Curitiba", "Curitiba", "Recife", "Recife", "Petrópolis", "Petrópolis"]
    for idx, region in enumerate(regions):
        dino_id = f"DINO_{idx}"
        manifest_rows.append({"dino_input_id": dino_id, "canonical_patch_id": f"PATCH_{idx}", "region": region, "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"})
        preflight_rows.append({"dino_input_id": dino_id, "resolved_status": "FOUND", "resolved_path_private": str(tmp_path / "private" / f"{dino_id}.tif")})
    manifest = tmp_path / "manifest.csv"
    preflight = tmp_path / "preflight.csv"
    write_csv(manifest, manifest_rows, ["dino_input_id", "canonical_patch_id", "region", "label_status", "target_status", "claim_scope"])
    write_csv(preflight, preflight_rows, ["dino_input_id", "resolved_status", "resolved_path_private"])
    return manifest, preflight


def test_v1ge_expanded_selection_resume_skip_existing(tmp_path: Path) -> None:
    manifest, preflight = inputs(tmp_path)
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1ge"
    cmd = [sys.executable, str(V1GE), "--execute", "--input-manifest", str(manifest), "--asset-preflight", str(preflight), "--output-dir", str(out), "--per-region-limit", "1", "--force", "--embedding-proxy-for-tests"]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((out / "dino_expanded_embedding_summary_v1ge.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    assert summary["success_count"] == 3
    assert summary["embedding_dim"] == 32
    assert len(list((out / "embeddings").glob("*.npz"))) == 3
    result2 = subprocess.run([sys.executable, str(V1GE), "--execute", "--input-manifest", str(manifest), "--asset-preflight", str(preflight), "--output-dir", str(out), "--per-region-limit", "1", "--resume", "--skip-existing", "--embedding-proxy-for-tests"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result2.returncode == 0, result2.stderr + result2.stdout
    summary2 = json.loads((out / "dino_expanded_embedding_summary_v1ge.json").read_text(encoding="utf-8"))
    assert summary2["skipped_existing_count"] >= 3


def test_v1gf_index_and_v1gg_human_review_package(tmp_path: Path) -> None:
    manifest, preflight = inputs(tmp_path)
    v1ge_out = tmp_path / "local_runs" / "dino_embeddings" / "v1ge"
    subprocess.run([sys.executable, str(V1GE), "--execute", "--input-manifest", str(manifest), "--asset-preflight", str(preflight), "--output-dir", str(v1ge_out), "--per-region-limit", "1", "--force", "--embedding-proxy-for-tests"], cwd=ROOT, text=True, capture_output=True, check=True)
    v1gf_out = tmp_path / "local_runs" / "dino_embeddings" / "v1gf"
    result = subprocess.run([sys.executable, str(V1GF), "--embedding-manifest", str(v1ge_out / "dino_expanded_embedding_manifest_v1ge.csv"), "--output-dir", str(v1gf_out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    index = read_csv(v1gf_out / "structural_evidence_index.csv")
    assert index
    assert {row["label_status"] for row in index} == {"NO_LABEL"}
    assert {row["target_status"] for row in index} == {"NO_TARGET"}
    assert {row["review_priority_is_not_label"] for row in index} == {"true"}
    v1gg_out = tmp_path / "local_runs" / "dino_embeddings" / "v1gg"
    result2 = subprocess.run([sys.executable, str(V1GG), "--structural-index", str(v1gf_out / "structural_evidence_index.csv"), "--output-dir", str(v1gg_out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result2.returncode == 0, result2.stderr + result2.stdout
    review = read_csv(v1gg_out / "human_review_manifest.csv")
    assert review
    assert (v1gg_out / "review_readme.md").exists()
    assert {row["human_review_required"] for row in review} == {"true"}
    assert not any(row["local_visual_path"].lower().endswith((".tif", ".tiff")) for row in review)
