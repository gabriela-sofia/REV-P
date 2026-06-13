"""v2bb download and raw inventory tests."""

from pathlib import Path
import scripts.v2bb_public_geometry_retrieval_feed_builder as engine

ROOT = Path(__file__).resolve().parents[1]


def test_public_downloads_are_audited_and_hashed():
    attempts = engine.load_csv(ROOT / "datasets" / "v2bb_public_download_attempts.csv")
    inventory = engine.load_csv(ROOT / "datasets" / "v2bb_public_raw_file_inventory.csv")
    assert len(attempts) == 4
    assert all(row["attempted"] == "true" for row in attempts)
    assert all(len(row["hash_sha256"]) == 64 for row in inventory)


def test_failed_download_is_recorded_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "DIRECT_SOURCES", [("event_context", "E", "bad", "context",
        "http://127.0.0.1:1/missing.geojson", "context", "polygon", "geojson")])
    monkeypatch.setattr(engine, "TEXT_SEARCHES", [])
    paths = {"dataset_dir": tmp_path/"datasets", "output_dir": tmp_path/"outputs", "config_dir": tmp_path/"configs",
             "external_dir": tmp_path/"external", "docs_dir": tmp_path/"docs"}
    code, _ = engine.run("download_public", **paths)
    assert code == 0
    assert engine.load_csv(paths["dataset_dir"]/"v2bb_public_download_attempts.csv")[0]["success"] == "false"
