"""Tests for v1og REC patch provenance graph builder.

All I/O is redirected to tmp_path — datasets/ is never touched.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1og_rec_patch_provenance_graph_builder.py"
DATASETS = ROOT / "datasets"


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_v1og_builds_provenance_graph_and_sanitizes_local_runs(tmp_path: Path) -> None:
    manifest = tmp_path / "fixture.csv"
    _write(manifest, [{"canonical_patch_id": "RECIFE_00001", "region": "Recife",
                       "asset_path_reference": "local_runs/protocolo_c/raw/S2A_MSIL2A_20220525T131241_T25LDD_patch.tif"}])

    out_graph = tmp_path / "graph.csv"
    out_alias = tmp_path / "alias.csv"
    out_break = tmp_path / "break.csv"

    env = {**os.environ,
           "REVP_RECIFE_PROVENANCE_CSV_PATHS": str(manifest),
           "REVP_V1OG_OUT_GRAPH": str(out_graph),
           "REVP_V1OG_OUT_ALIAS": str(out_alias),
           "REVP_V1OG_OUT_BREAK": str(out_break),
           "REVP_V1OG_SCHEMA_GRAPH": str(tmp_path / "s_graph.csv"),
           "REVP_V1OG_SCHEMA_ALIAS": str(tmp_path / "s_alias.csv"),
           "REVP_V1OG_SCHEMA_BREAK": str(tmp_path / "s_break.csv"),
           "REVP_V1OG_DOC": str(tmp_path / "doc.md")}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "--emit-evidence"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    row = _read(out_graph)[0]
    assert row["canonical_patch_id"] == "REC_00001"
    assert row["product_date_status"] == "PRODUCT_DATE_CONFIRMED"
    assert "local_runs" not in out_graph.read_text(encoding="utf-8")
    assert _read(out_alias)[0]["alias_status"] == "NORMALIZED"
    assert _read(out_break)[0]["can_recover_scene_date_now"] == "true"

    # Verify the test outputs are in tmp_path, not in real datasets/
    assert out_graph.exists(), "Output must be in tmp_path"
    assert not (DATASETS / "recife_patch_provenance_graph_registry.csv").samefile(out_graph), \
        "Test must not write to real datasets/"
