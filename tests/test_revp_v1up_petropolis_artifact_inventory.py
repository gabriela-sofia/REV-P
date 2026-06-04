import json
import shutil

import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env, write_csv


def test_artifact_inventory_detects_geojson_and_csv_fields(tmp_path, monkeypatch):
    data, _, _, raw = set_env(tmp_path, monkeypatch)
    shutil.copyfile("tests/fixtures/v1up/landslide_scar.geojson", raw / "landslide.geojson")
    shutil.copyfile("tests/fixtures/v1up/coordinate_phenomenon.csv", raw / "coord.csv")
    write_csv(data / "v1up_petropolis_download_manifest.csv", common.DOWNLOAD_COLUMNS, [
        {"download_id": "DL_v1up_0000", "event_id": "PET_2022_02_15", "source_id": "SGB_RIGEO", "record_id": "r0", "url": "", "url_sha1_12": "", "safe_filename": "landslide.geojson", "basename": "", "format_hint": "geojson", "download_status": "DOWNLOADED", "downloaded": "true", "sha256": "", "size_bytes": "", "duplicate_of_sha256": "", "collision_status": "", "storage_scope": "", "blocking_reason": "", "notes": ""},
        {"download_id": "DL_v1up_0001", "event_id": "PET_2022_02_15", "source_id": "SGB_RIGEO", "record_id": "r1", "url": "", "url_sha1_12": "", "safe_filename": "coord.csv", "basename": "", "format_hint": "csv", "download_status": "DOWNLOADED", "downloaded": "true", "sha256": "", "size_bytes": "", "duplicate_of_sha256": "", "collision_status": "", "storage_scope": "", "blocking_reason": "", "notes": ""},
    ])
    rows = common.run_artifact_inventory()
    assert any(r["has_geodata"] == "true" for r in rows)
    assert any(r["has_coordinate_fields"] == "true" and r["has_phenomenon_field"] == "true" for r in rows)
