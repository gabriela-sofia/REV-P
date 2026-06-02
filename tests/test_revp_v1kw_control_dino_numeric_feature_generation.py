"""Tests for v1kw control DINO and numeric feature generation."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1kw_control_dino_numeric_feature_generation.py"
OUT = ROOT / "datasets/control_numeric_feature_registry.csv"
DINO = ROOT / "datasets/control_dino_readiness_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_outputs_exist_or_run_existing() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert DINO.exists()


def test_dino_is_frozen_and_embeddings_are_not_saved() -> None:
    dino = rows(DINO)
    assert len([r for r in dino if r["dino_feature_status"] == "PASS"]) >= 9
    assert all(r["dino_frozen"] == "true" for r in dino)
    assert all(r["embedding_saved"] == "false" for r in dino)


def test_real_features_do_not_create_formal_negative_or_label() -> None:
    features = rows(OUT)
    assert len([r for r in features if r["numeric_feature_status"] == "PASS"]) >= 9
    assert all(r["can_be_formal_negative"] == "false" for r in features)
    assert all(r["can_create_operational_label"] == "false" for r in features)
    assert all(r["can_train_model"] == "false" for r in features)
