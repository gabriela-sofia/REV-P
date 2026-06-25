from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_09_real_acquisition_queue as q


def test_city_mapping() -> None:
    assert q.city_for_patch("REC_00019") == "Recife"
    assert q.city_for_patch("PET_00016") == "Petropolis"
    assert q.city_for_patch("CUR_00001") == "Curitiba"
    assert q.city_for_patch("XXX_00001") == ""


def test_data06_queue_has_ten_targets_with_city() -> None:
    rows = q.build_data06_queue(10)
    assert len(rows) == 10
    assert all(row["city"] for row in rows)
    assert all(row["status"] == "PENDING_REAL_ACQUISITION" for row in rows)
    assert all(row["source_ref_required"] == "sim" for row in rows)


def test_data06_queue_lists_accepted_sources() -> None:
    rows = q.build_data06_queue(10)
    families = rows[0]["accepted_source_family"]
    assert "CEMADEN" in families
    assert "Copernicus EMS/CEMS" in families
    assert "ANA" in families


def test_data07_queue_requires_sensor_source() -> None:
    rows = q.build_data07_queue(10)
    assert len(rows) == 10
    assert all("SENTINEL_2" in row["needed_field"] for row in rows)
    assert all(row["sensor_source_ref_required"] == "sim" for row in rows)
    assert "historico/export GEE" in rows[0]["accepted_source_family"]


def test_no_auto_resolution_flags() -> None:
    summary = q.build_summary(q.build_data06_queue(10), q.build_data07_queue(10))
    assert summary["auto_resolution_by_bbox"] is False
    assert summary["auto_resolution_by_filename"] is False
    assert summary["live_calls"] == 0


def test_bbox_only_from_real_geometry(tmp_path: Path) -> None:
    # An invalid-geometry registry row must never yield an invented bbox.
    reg = tmp_path / "reg.csv"
    with reg.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patch_id", "crs", "is_valid_geometry", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"])
        writer.writeheader()
        writer.writerow({"patch_id": "REC_00019", "crs": "UNKNOWN", "is_valid_geometry": "false", "bbox_minx": "", "bbox_miny": "", "bbox_maxx": "", "bbox_maxy": ""})
    lookup = {r["patch_id"]: r for r in csv.DictReader(reg.open(encoding="utf-8"))}
    bbox, crs = q._bbox_crs_for("REC_00019", lookup)
    assert bbox == ""
    assert crs == ""


def test_valid_geometry_yields_bbox(tmp_path: Path) -> None:
    lookup = {
        "REC_00019": {
            "patch_id": "REC_00019",
            "crs": "EPSG:32725",
            "is_valid_geometry": "true",
            "bbox_minx": "-34.99",
            "bbox_miny": "-8.23",
            "bbox_maxx": "-34.98",
            "bbox_maxy": "-8.22",
        }
    }
    bbox, crs = q._bbox_crs_for("REC_00019", lookup)
    assert bbox == "-34.99,-8.23,-34.98,-8.22"
    assert crs == "EPSG:32725"


def test_write_outputs_creates_queues(tmp_path: Path) -> None:
    summary = q.write_outputs(tmp_path, 10)
    assert (tmp_path / "mv2_data_06_temporal_window_acquisition_queue.csv").exists()
    assert (tmp_path / "mv2_data_07_sensor_lineage_acquisition_queue.csv").exists()
    assert (tmp_path / "mv2_data_08_metadata_config_checklist.md").exists()
    assert summary["data_06_targets"] == 10
