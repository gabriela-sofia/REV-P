import csv
import os

import scripts.protocolo_c.revp_v1up_petropolis_common as common


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    configs = tmp_path / "configs" / "protocolo_c"
    raw = tmp_path / "raw"
    for path in [data, docs, configs, raw]:
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(configs))
    monkeypatch.setattr(common, "LOCAL_RAW_DIR", str(raw))
    monkeypatch.setattr(common, "LOCAL_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setattr(common, "LOCAL_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    monkeypatch.setattr(common, "LOCAL_REPORTS_DIR", str(tmp_path / "reports"))
    return data, docs, configs, raw


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_source_target_registry_created_for_two_petropolis_events(tmp_path, monkeypatch):
    data, _, _, _ = set_env(tmp_path, monkeypatch)
    rows = common.run_source_target_builder()
    assert os.path.exists(data / "v1up_petropolis_source_target_registry.csv")
    assert {r["event_id"] for r in rows} == {"PET_2022_02_15", "PET_2024_03_21_28"}
    assert any(r["source_id"] == "SGB_RIGEO" and r["can_contain_observed_geometry"] == "true" for r in rows)
