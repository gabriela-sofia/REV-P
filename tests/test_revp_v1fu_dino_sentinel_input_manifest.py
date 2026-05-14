from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1fu_dino_sentinel_input_manifest.py"
OUT_DIR = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest"
MANIFEST = OUT_DIR / "dino_sentinel_input_manifest_v1fu.csv"
SUMMARY = OUT_DIR / "dino_sentinel_input_summary_v1fu.json"
QA = OUT_DIR / "dino_sentinel_input_qa_v1fu.csv"
STATUS = OUT_DIR / "dino_sentinel_input_status_v1fu.csv"
V1FT_READY = ROOT / "manifests" / "training_readiness" / "revp_v1ft_embedding_config_and_recife_balance_audit" / "embedding_ready_assets_v1ft.csv"

FORBIDDEN_DIRS = {"data", "outputs", "docs"}
FORBIDDEN_EXTENSIONS = {".tif", ".tiff", ".zip", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".parquet"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_v1fu_script_creates_review_only_dino_manifest() -> None:
    tracked_outputs = [MANIFEST, SUMMARY, QA, STATUS]
    snapshots = {path: path.read_bytes() for path in tracked_outputs if path.exists()}
    try:
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)

        for path in tracked_outputs:
            assert path.exists(), path

        for dirname in FORBIDDEN_DIRS:
            assert not (ROOT / dirname).exists(), dirname

        forbidden_files = [
            path
            for path in ROOT.rglob("*")
            if ".git" not in path.parts and path.is_file() and path.suffix.lower() in FORBIDDEN_EXTENSIONS
        ]
        assert forbidden_files == []

        v1ft_ready = read_csv(V1FT_READY)
        expected_sentinel = [
            row
            for row in v1ft_ready
            if row.get("modality") == "sentinel_raster_path_only"
            and row.get("config_status") == "READY_SENTINEL_FIRST_REVIEW_ONLY"
        ]
        manifest = read_csv(MANIFEST)

        if len(expected_sentinel) == 128:
            assert len(manifest) == 128

        assert {row["label_status"] for row in manifest} == {"NO_LABEL"}
        assert {row["target_status"] for row in manifest} == {"NO_TARGET"}
        assert {row["pixel_read_status"] for row in manifest} == {"NOT_READ__FUTURE_DINO_ENCODING_ONLY"}
        assert {row["encoder_mode"] for row in manifest} == {"frozen_encoder"}
        assert {row["dino_scope"] for row in manifest} == {"SENTINEL_FIRST_EMBEDDING_INPUT"}
        assert all(row["claim_scope"] == "REVIEW_ONLY_NO_PREDICTIVE_CLAIM" for row in manifest)
        assert all("classification" not in row["claim_scope"].lower() for row in manifest)
        assert all("performance" not in row["claim_scope"].lower() for row in manifest)

        qa_rows = read_csv(QA)
        assert qa_rows
        assert {row["status"] for row in qa_rows} == {"PASS"}

        status_rows = {row["field"]: row["value"] for row in read_csv(STATUS)}
        assert status_rows["status"] == "PASS"
        assert status_rows["actual_sentinel_count"] == "128"
    finally:
        for path, content in snapshots.items():
            path.write_bytes(content)
