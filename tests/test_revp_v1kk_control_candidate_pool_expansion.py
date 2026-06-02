"""Tests for v1kk control candidate pool expansion."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1kk_control_candidate_pool_expansion.py"
OUT = ROOT / "datasets/expanded_conservative_control_pool_registry.csv"
DESIGN = ROOT / "datasets/expanded_control_sampling_design_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--limit", "50", "--emit-pool"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert DESIGN.exists()


def test_expanded_controls_are_not_negatives_or_labels() -> None:
    accepted = [r for r in rows(OUT) if r["classification"] == "EXPANDED_CONSERVATIVE_CONTROL_CANDIDATE"]
    assert accepted
    assert all(r["can_be_formal_negative"] == "false" for r in accepted)
    assert all(r["can_create_operational_label"] == "false" for r in accepted)
    assert all(r["can_train_model"] == "false" for r in accepted)


def test_positive_and_feature_buffers_are_enforced_for_accepted_controls() -> None:
    accepted = [r for r in rows(OUT) if r["classification"] == "EXPANDED_CONSERVATIVE_CONTROL_CANDIDATE"]
    assert all(r["outside_positive_1000m"] == "true" for r in accepted)
    assert all(r["outside_landslide_feature_100m"] == "true" for r in accepted)


def test_public_outputs_have_no_private_paths() -> None:
    text = OUT.read_text(encoding="utf-8") + DESIGN.read_text(encoding="utf-8")
    assert "C:\\" not in text and "C:/" not in text
    assert "gabriela" not in text.lower()
