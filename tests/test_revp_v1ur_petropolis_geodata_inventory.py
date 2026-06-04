import zipfile, shutil
import scripts.protocolo_c.revp_v1ur_petropolis_common as common
from tests.test_revp_v1ur_petropolis_geodata_signal_seed_builder import set_env, write_csv

def test_inventory_detects_geodata_in_zip_geojson_and_csv(tmp_path, monkeypatch):
    data,_,_,raw=set_env(tmp_path,monkeypatch)
    z=raw/"pkg.zip"
    with zipfile.ZipFile(z,"w") as f:
        f.write("tests/fixtures/v1ur/occurrence.geojson","occurrence.geojson")
        f.write("tests/fixtures/v1ur/coordinate_phenomenon.csv","coord.csv")
    write_csv(data/"v1ur_petropolis_geodata_download_manifest.csv", common.DOWNLOAD_COLUMNS, [{"download_id":"d","event_id":"PET_2022_02_15","candidate_url_id":"c","url":"","safe_filename":"pkg.zip","local_path_hash":"h","sha256":"","file_size_bytes":"","mime_type":"","extension":"zip","download_status":"DOWNLOADED","duplicate_status":"","license_status":"","notes":""}])
    rows=common.run_geodata_inventory()
    assert any(r["has_geometry"]=="true" for r in rows)
    assert any(r["has_coordinate_fields"]=="true" for r in rows)
