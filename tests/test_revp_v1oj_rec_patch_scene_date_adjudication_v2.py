"""Tests for v1oj REC patch scene_date adjudication v2.

All I/O is redirected to tmp_path — datasets/ is never touched.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1oj_rec_patch_scene_date_adjudication_v2.py"
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


def test_v1oj_confirms_only_patch_asset_product_date_linkage(tmp_path: Path) -> None:
    in_graph = tmp_path / "graph.csv"
    in_product = tmp_path / "product.csv"
    in_v1od = tmp_path / "v1od.csv"
    out_adj = tmp_path / "adj.csv"
    out_confirmed = tmp_path / "confirmed.csv"
    out_unresolved = tmp_path / "unresolved.csv"

    # Use realistic-looking IDs (not REC_00001 pattern)
    _write(in_graph, [{"canonical_patch_id": "RECIFE_SENTINEL_PATCH_S2A_001"},
                      {"canonical_patch_id": "RECIFE_NAME_ONLY_PATCH_002"}])
    _write(in_product, [
        {"resolution_id": "RESOLUTION_V1OI_S2A_001",
         "patch_id": "RECIFE_SENTINEL_PATCH_S2A_001",
         "resolved_scene_date": "2022-05-25",
         "resolution_status": "PRODUCT_DATE_CONFIRMED",
         "has_patch_association": "true",
         "confidence": "HIGH"},
        {"resolution_id": "RESOLUTION_V1OI_NAME_002",
         "patch_id": "RECIFE_NAME_ONLY_PATCH_002",
         "resolved_scene_date": "",
         "resolution_status": "FILENAME_DATE_CANDIDATE_ONLY",
         "has_patch_association": "true",
         "confidence": "MEDIUM"},
    ])
    _write(in_v1od, [], ["patch_id"])

    env = {**os.environ,
           "REVP_V1OJ_IN_GRAPH": str(in_graph),
           "REVP_V1OJ_IN_PRODUCT": str(in_product),
           "REVP_V1OJ_IN_V1OD": str(in_v1od),
           "REVP_V1OJ_OUT_ADJ": str(out_adj),
           "REVP_V1OJ_OUT_CONFIRMED": str(out_confirmed),
           "REVP_V1OJ_OUT_UNRESOLVED": str(out_unresolved),
           "REVP_V1OJ_SCHEMA_ADJ": str(tmp_path / "s_adj.csv"),
           "REVP_V1OJ_SCHEMA_CONFIRMED": str(tmp_path / "s_confirmed.csv"),
           "REVP_V1OJ_SCHEMA_UNRESOLVED": str(tmp_path / "s_unresolved.csv"),
           "REVP_V1OJ_DOC": str(tmp_path / "doc.md")}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "--emit-evidence"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    rows = {r["patch_id"]: r for r in _read(out_adj)}
    assert rows["RECIFE_SENTINEL_PATCH_S2A_001"]["scene_date_status"] == "SCENE_DATE_CONFIRMED"
    assert rows["RECIFE_NAME_ONLY_PATCH_002"]["scene_date_status"] != "SCENE_DATE_CONFIRMED"
    conf_rows = _read(out_confirmed)
    assert conf_rows, "Expected at least one confirmed row"
    assert conf_rows[0]["can_support_temporal_matching"] == "true"

    # Verify the test outputs are in tmp_path, not in real datasets/
    assert out_adj.exists(), "Output must be in tmp_path"
    assert not (DATASETS / "recife_patch_scene_date_adjudication_v2.csv").samefile(out_adj), \
        "Test must not write to real datasets/"
