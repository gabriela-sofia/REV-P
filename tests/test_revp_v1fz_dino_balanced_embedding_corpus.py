from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1fz_dino_balanced_embedding_corpus.py"
CONFIG = ROOT / "configs" / "dino_embedding_extraction.example.yaml"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ppm(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P6\n12 12\n255\n" + bytes(color) * (12 * 12))


def build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    private = tmp_path / "PROJETO"
    fields = ["dino_input_id", "canonical_patch_id", "region", "asset_path_reference", "label_status", "target_status", "claim_scope"]
    pre_fields = ["dino_input_id", "resolved_status", "resolved_path_private"]
    manifest_rows = []
    preflight_rows = []
    colors = {
        "Curitiba": [(10, 20, 30), (20, 30, 40)],
        "Petrópolis": [(40, 50, 60)],
        "Recife": [(70, 80, 90), (80, 90, 100)],
    }
    idx = 1
    for region, region_colors in colors.items():
        for color in region_colors:
            dino_id = f"DINO_{idx:03d}"
            image = private / region / f"{dino_id}.ppm"
            ppm(image, color)
            manifest_rows.append({"dino_input_id": dino_id, "canonical_patch_id": f"PATCH_{idx}", "region": region, "asset_path_reference": str(image), "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"})
            preflight_rows.append({"dino_input_id": dino_id, "resolved_status": "FOUND", "resolved_path_private": str(image)})
            idx += 1
    manifest = tmp_path / "manifest.csv"
    preflight = tmp_path / "preflight.csv"
    write_csv(manifest, manifest_rows, fields)
    write_csv(preflight, preflight_rows, pre_fields)
    return manifest, preflight, tmp_path / "local_runs" / "dino_embeddings" / "v1fz"


def test_v1fz_balanced_selection_multi_region_outputs(tmp_path: Path) -> None:
    manifest, preflight, output_dir = build_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--execute",
            "--input-manifest",
            str(manifest),
            "--asset-preflight",
            str(preflight),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output_dir),
            "--regions",
            "Curitiba",
            "Petropolis",
            "Recife",
            "--per-region-limit",
            "2",
            "--backbone",
            "fake_smoke_encoder",
            "--allow-cpu",
            "--force",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    summary = json.loads((output_dir / "dino_balanced_embedding_summary_v1fz.json").read_text(encoding="utf-8"))
    assert summary["success_count"] == 5
    assert summary["embeddings_by_region"] == {"Curitiba": 2, "Petrópolis": 1, "Recife": 2}
    assert summary["qa_status"] == "PASS"
    assert summary["review_only"] is True
    assert summary["supervised_training"] is False
    assert summary["labels_created"] is False
    assert summary["predictive_claims"] is False
    assert summary["multimodal_hold"] is True

    audit = read_csv(output_dir / "dino_balanced_selection_audit_v1fz.csv")
    petro = [row for row in audit if row["matched_region"] == "Petrópolis"][0]
    assert petro["selected_count"] == "1"
    manifest_rows = read_csv(output_dir / "dino_balanced_embedding_manifest_v1fz.csv")
    assert {row["label_status"] for row in manifest_rows} == {"NO_LABEL"}
    assert {row["target_status"] for row in manifest_rows} == {"NO_TARGET"}
    assert list((output_dir / "embeddings").glob("*.npz"))
    assert read_csv(output_dir / "dino_balanced_cluster_metrics_v1fz.csv")
    assert read_csv(output_dir / "dino_balanced_region_diagnostics_v1fz.csv")
    assert "local_runs/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_v1fz_failure_isolation(tmp_path: Path) -> None:
    manifest, preflight, output_dir = build_fixture(tmp_path)
    rows = read_csv(preflight)
    rows[0]["resolved_path_private"] = str(tmp_path / "missing.ppm")
    write_csv(preflight, rows, ["dino_input_id", "resolved_status", "resolved_path_private"])
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--execute",
            "--input-manifest",
            str(manifest),
            "--asset-preflight",
            str(preflight),
            "--output-dir",
            str(output_dir),
            "--regions",
            "Curitiba",
            "Recife",
            "--per-region-limit",
            "1",
            "--backbone",
            "fake_smoke_encoder",
            "--allow-cpu",
            "--force",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    summary = json.loads((output_dir / "dino_balanced_embedding_summary_v1fz.json").read_text(encoding="utf-8"))
    assert summary["failed_count"] == 1
    assert summary["success_count"] == 1
    failures = read_csv(output_dir / "dino_balanced_embedding_failures_v1fz.csv")
    assert failures
