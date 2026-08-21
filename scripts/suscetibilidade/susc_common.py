"""
SUSC common aggregator: shared paths, simple logging, re-exports.

Imports the focused helper modules so downstream scripts can `from susc_common
import ...` a single surface. Pure Python; no third-party requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import susc_geometry as geometry  # noqa: E402,F401
import susc_governance as governance  # noqa: E402,F401
import susc_io as io  # noqa: E402,F401

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SUSC_SCHEMAS = ROOT / "schemas" / "suscetibilidade"
SUSC_SCRIPTS = ROOT / "scripts" / "suscetibilidade"
SUSC_DATASETS = ROOT / "datasets" / "suscetibilidade"
SUSC_MANIFESTS = ROOT / "manifests" / "suscetibilidade"
SUSC_OUTPUTS = ROOT / "outputs_public" / "suscetibilidade"
SUSC_MATRIX = SUSC_DATASETS / "susc_features_by_patch_v1.csv"
SUSC_MATRIX_MANIFEST = SUSC_MANIFESTS / "susc_features_by_patch_v1_artifact_manifest.json"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}


def log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}")


def matrix_sha256_ok() -> bool:
    """True if the SUSC-03 matrix still matches its recorded SHA256."""
    if not SUSC_MATRIX.exists() or not SUSC_MATRIX_MANIFEST.exists():
        return False
    expected = io.read_json(SUSC_MATRIX_MANIFEST).get("sha256")
    return io.sha256_file(SUSC_MATRIX) == expected


def find_model_artifacts() -> list[str]:
    found = []
    for base in (SUSC_OUTPUTS, SUSC_SCRIPTS, SUSC_DATASETS):
        if base.exists():
            found += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    return found
