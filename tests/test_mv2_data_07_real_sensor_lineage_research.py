from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_07_real_sensor_lineage_research as r07


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _synthetic(tmp_path: Path, sensor: str, asset_type: str = "SENTINEL_TIF_ASSET", path_ref: str = "data/sentinel/patch_recife_00019.tif"):
    queue = tmp_path / "q.csv"
    pkg = tmp_path / "pkg.csv"
    v1fu = tmp_path / "v1fu.csv"
    v2bd = tmp_path / "v2bd.csv"
    _write(queue, ["target_rank", "patch_id", "asset_id"], [{"target_rank": "1", "patch_id": "REC_00019", "asset_id": "a1"}])
    _write(pkg, ["patch_id", "sentinel_sensor_family", "sentinel_observation_date"], [{"patch_id": "REC_00019", "sentinel_sensor_family": sensor, "sentinel_observation_date": "2022-05-24"}])
    _write(v1fu, ["source_asset_id", "source_asset_type", "asset_path_reference", "dino_input_id"], [{"source_asset_id": "a1", "source_asset_type": asset_type, "asset_path_reference": path_ref, "dino_input_id": "DINO_X"}])
    _write(v2bd, ["candidate_asset_id", "asset_type", "asset_file"], [{"candidate_asset_id": "a1", "asset_type": asset_type, "asset_file": path_ref}])
    return queue, pkg, v1fu, v2bd


def test_normalize_sensor_family() -> None:
    assert r07.normalize_sensor_family("SENTINEL2_MSI") == "SENTINEL_2"
    assert r07.normalize_sensor_family("Sentinel-2") == "SENTINEL_2"
    assert r07.normalize_sensor_family("SENTINEL1") == "SENTINEL_1"
    assert r07.normalize_sensor_family("UNKNOWN") == "UNKNOWN"
    assert r07.normalize_sensor_family("") == "UNKNOWN"


def test_sentinel_2_family_is_spectral_but_needs_product_id(tmp_path: Path) -> None:
    q, p, f, b = _synthetic(tmp_path, "SENTINEL2_MSI")
    rows = r07.research_targets(q, p, f, b)
    row = rows[0]
    assert row["sensor_family"] == "SENTINEL_2"
    assert row["spectral_eligible"] == "true"
    assert row["evidence_strength"] == "MEDIUM"
    assert row["review_status"] == "NEEDS_REVIEW"
    assert row["blocked_reason"] == "EXPLICIT_S2_PRODUCT_ID_ABSENT_NEEDS_HUMAN_CONFIRM"


def test_unknown_sensor_is_blocked_not_inferred(tmp_path: Path) -> None:
    q, p, f, b = _synthetic(tmp_path, "UNKNOWN")  # SENTINEL_TIF_ASSET but no S2 declaration
    rows = r07.research_targets(q, p, f, b)
    row = rows[0]
    assert row["sensor_family"] == "UNKNOWN"
    assert row["spectral_eligible"] == "false"
    assert row["review_status"] == "UNKNOWN_BLOCKED"
    assert row["blocked_reason"] == "NO_SENTINEL_2_DECLARATION"


def test_sentinel_1_is_support_only(tmp_path: Path) -> None:
    q, p, f, b = _synthetic(tmp_path, "SENTINEL1")
    rows = r07.research_targets(q, p, f, b)
    row = rows[0]
    assert row["sensor_family"] == "SENTINEL_1"
    assert row["support_only"] == "true"
    assert row["spectral_eligible"] == "false"


def test_absolute_path_is_redacted(tmp_path: Path) -> None:
    q, p, f, b = _synthetic(tmp_path, "SENTINEL2_MSI", path_ref="C:/Users/secret/data/sentinel/patch_recife_00019.tif")
    rows = r07.research_targets(q, p, f, b)
    assert not rows[0]["source_asset_ref"].startswith("C:")
    assert "<repo_local>/data/sentinel/" in rows[0]["source_asset_ref"]


def test_no_local_candidate_when_not_strong(tmp_path: Path) -> None:
    q, p, f, b = _synthetic(tmp_path, "SENTINEL2_MSI")
    rows = r07.research_targets(q, p, f, b)
    dest = tmp_path / "inputs_local" / "data_07.csv"
    assert r07.write_local_candidate_if_strong(rows, dest) is False
    assert not dest.exists()


def test_real_run_documents_sentinel_2_family() -> None:
    rows = r07.research_targets()
    assert len(rows) == 10
    s2 = sum(1 for r in rows if r["sensor_family"] == "SENTINEL_2")
    assert s2 == 5  # the 5 Recife patches
    assert all(r["evidence_strength"] != "STRONG" for r in rows)  # product id absent everywhere


def test_no_side_effects() -> None:
    summary = r07.summarize(r07.research_targets(), local_created=False)
    assert summary["live_calls"] == 0
    assert summary["downloads"] == 0
    assert summary["rasters"] == 0
    assert summary["crops"] == 0
