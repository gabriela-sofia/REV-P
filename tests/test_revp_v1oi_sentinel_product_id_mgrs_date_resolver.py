"""Tests for v1oi Sentinel product id / MGRS date resolver.

All I/O is redirected to tmp_path — datasets/ is never touched.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1oi_sentinel_product_id_mgrs_date_resolver.py"
DATASETS = ROOT / "datasets"


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fnames = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_v1oi_confirms_only_recognized_sentinel_product_dates(tmp_path: Path) -> None:
    in_graph = tmp_path / "graph.csv"
    in_local = tmp_path / "local.csv"
    out_res = tmp_path / "resolution.csv"
    out_conf = tmp_path / "confidence.csv"

    _write(in_graph, [
        {"provenance_id": "P1", "canonical_patch_id": "RECIFE_PATCH_SENTINEL_001",
         "asset_path_sanitized": "S2A_MSIL2A_20220525T131241_T25LDD_patch.tif",
         "source_manifest": "fixture.csv", "candidate_date": ""},
        {"provenance_id": "P2", "canonical_patch_id": "RECIFE_PATCH_NAME_ONLY_002",
         "asset_path_sanitized": "RECIFE_PATCH_NAME_ONLY_002", "source_manifest": "fixture.csv", "candidate_date": ""},
    ])
    _write(in_local, [],
           ["date_candidate_id", "file_name", "date_candidate", "product_date_status", "confidence", "pattern_used"])

    env = {**os.environ,
           "REVP_V1OI_IN_GRAPH": str(in_graph),
           "REVP_V1OI_IN_LOCAL": str(in_local),
           "REVP_V1OI_OUT_RES": str(out_res),
           "REVP_V1OI_OUT_CONF": str(out_conf),
           "REVP_V1OI_SCHEMA_RES": str(tmp_path / "s_res.csv"),
           "REVP_V1OI_SCHEMA_CONF": str(tmp_path / "s_conf.csv"),
           "REVP_V1OI_DOC": str(tmp_path / "doc.md")}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "--emit-evidence"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    rows = _read(out_res)
    # Row with real Sentinel product name should get PRODUCT_DATE_CONFIRMED
    confirmed = [r for r in rows if r.get("resolution_status") == "PRODUCT_DATE_CONFIRMED"]
    assert confirmed, "Expected at least one PRODUCT_DATE_CONFIRMED row"
    assert confirmed[0]["resolved_scene_date"] == "2022-05-25"
    # Row with patch name only should NOT be confirmed
    not_confirmed = [r for r in rows if r.get("canonical_patch_id") == "RECIFE_PATCH_NAME_ONLY_002"]
    if not_confirmed:
        assert not_confirmed[0]["resolved_scene_date"] == ""
    assert _read(out_conf)[0]["can_train_model"] == "false"

    # Verify the test outputs are in tmp_path, not in real datasets/
    assert out_res.exists(), "Output must be in tmp_path"
    assert not (DATASETS / "recife_sentinel_product_date_resolution_registry.csv").samefile(out_res), \
        "Test must not write to real datasets/"
