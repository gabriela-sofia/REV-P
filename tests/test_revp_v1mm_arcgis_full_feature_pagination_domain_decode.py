"""Tests for v1mm ArcGIS pagination and domain decode."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mm_arcgis_full_feature_pagination_domain_decode.py"
HARVEST = ROOT / "datasets/arcgis_full_feature_harvest_registry.csv"
DOMAIN = ROOT / "datasets/arcgis_field_domain_decode_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert HARVEST.exists() and DOMAIN.exists()


def test_domain_decode_preserves_original_and_label_columns() -> None:
    assert {"original_code", "decoded_label", "field_name"}.issubset(set(rows(DOMAIN)[0].keys()))
    assert all(r["private_path_removed"] == "true" for r in rows(DOMAIN))


def test_raw_policy_and_public_sanitization() -> None:
    low = (HARVEST.read_text(encoding="utf-8") + DOMAIN.read_text(encoding="utf-8")).lower()
    assert "raw_only_local_runs" in low
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
