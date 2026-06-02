"""Tests for v1km positive/control numeric feature table."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/positive_control_numeric_feature_registry.csv"
AVAIL = ROOT / "datasets/positive_control_feature_availability_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    chain = [
        ("revp_v1kk_control_candidate_pool_expansion.py", ["--force", "--limit", "50", "--emit-pool"]),
        ("revp_v1kl_control_multimodal_patch_acquisition.py", ["--force", "--emit-patch-qa"]),
        ("revp_v1km_positive_control_numeric_feature_table.py", ["--force", "--emit-feature-table"]),
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert AVAIL.exists()


def test_positive_features_do_not_create_control_labels() -> None:
    all_rows = rows(OUT)
    assert any(r["sample_role"].startswith("OFFICIAL") for r in all_rows)
    assert all(r["can_create_operational_label"] == "false" for r in all_rows)
    assert all(r["can_train_model"] == "false" for r in all_rows)


def test_control_feature_blocker_is_explicit_when_controls_are_not_ready() -> None:
    matrix = rows(AVAIL)[0]
    if int(matrix["control_ready_count"]) < 9:
        assert matrix["blocking_reason"] == "CONTROL_NUMERIC_PATCH_FEATURES_BLOCKED"


def test_public_outputs_have_no_private_paths_or_raw_embeddings() -> None:
    text = OUT.read_text(encoding="utf-8") + AVAIL.read_text(encoding="utf-8")
    assert "C:\\" not in text and "C:/" not in text
    assert "gabriela" not in text.lower()
    assert "embedding_vector" not in text.lower()
