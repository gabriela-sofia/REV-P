import csv
import os
import shutil
from pathlib import Path

import scripts.protocolo_c.revp_v2aa_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v2aa"
SOURCE_FIXTURES = [
    "patch_registry_sentinel_filenames.csv",
    "sidecar_datetime.json",
    "sidecar_sensing_date.json",
    "ambiguous_dates_registry.csv",
    "conflict_source_a.csv",
    "conflict_source_b.csv",
    "created_modified_trap.csv",
    "no_date_registry.csv",
]


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    scan = tmp_path / "scan_root"
    staging = tmp_path / "local_only" / "v2aa" / "staging"
    reports = tmp_path / "local_only" / "v2aa" / "reports"
    for p in (data, docs, cfg, scan, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "SCAN_ROOTS", [str(scan)])
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data, scan


def install_base_inputs(data, scan):
    for name in SOURCE_FIXTURES:
        shutil.copy(FIXTURE_DIR / name, Path(scan) / name)
    # Event-patch candidate linkage (real namespace) with one matching recovered
    # patch (P_S2A) and one without a recoverable date (P_NODATE).
    write_csv(data / "v1us_event_temporal_window_linkage.csv", [
        "temporal_linkage_id", "event_patch_candidate_id", "event_id", "patch_id",
        "region", "event_start_date", "event_end_date", "sentinel_scene_date",
        "has_sentinel_date", "temporal_linkage_class", "notes",
    ], [
        {"temporal_linkage_id": "TWL0", "event_patch_candidate_id": "EPC0",
         "event_id": "REC_2022_05_24_30", "patch_id": "P_S2A", "region": "REC",
         "event_start_date": "2022-05-24", "event_end_date": "2022-05-25",
         "sentinel_scene_date": "", "has_sentinel_date": "false",
         "temporal_linkage_class": "SENTINEL_DATE_MISSING", "notes": "x"},
        {"temporal_linkage_id": "TWL1", "event_patch_candidate_id": "EPC1",
         "event_id": "REC_2022_05_24_30", "patch_id": "P_S2B", "region": "REC",
         "event_start_date": "2022-05-24", "event_end_date": "2022-05-25",
         "sentinel_scene_date": "", "has_sentinel_date": "false",
         "temporal_linkage_class": "SENTINEL_DATE_MISSING", "notes": "x"},
        {"temporal_linkage_id": "TWL2", "event_patch_candidate_id": "EPC2",
         "event_id": "REC_2022_05_24_30", "patch_id": "P_NODATE", "region": "REC",
         "event_start_date": "2022-05-24", "event_end_date": "2022-05-25",
         "sentinel_scene_date": "", "has_sentinel_date": "false",
         "temporal_linkage_class": "SENTINEL_DATE_MISSING", "notes": "x"},
    ])
    write_csv(data / "v1us_patch_registry_resolution.csv", [
        "patch_resolution_id", "patch_id", "region", "sentinel_scene_date",
    ], [
        {"patch_resolution_id": "PR0", "patch_id": "P_S2A", "region": "REC", "sentinel_scene_date": ""},
        {"patch_resolution_id": "PR1", "patch_id": "P_S2B", "region": "REC", "sentinel_scene_date": ""},
        {"patch_resolution_id": "PR2", "patch_id": "P_NODATE", "region": "REC", "sentinel_scene_date": ""},
    ])


def test_scanner_finds_candidates_without_scanning_local_only(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    # A local_only directory under the scan root must be skipped entirely.
    leak = Path(scan) / "local_only"
    leak.mkdir()
    write_csv(leak / "raw_patch.csv", ["patch_id", "scene_date"], [{"patch_id": "LEAK", "scene_date": "2022-01-01"}])
    rows = common.run_patch_source_registry_scanner(common.parse_args([]))
    paths = {r["registry_path"] for r in rows}
    assert not any("local_only" in p for p in paths)
    assert any(r["should_parse_for_dates"] == "true" for r in rows)
    # The Sentinel-filename registry is flagged as a patch+date source.
    sentinel = [r for r in rows if r["registry_path"].endswith("patch_registry_sentinel_filenames.csv")]
    assert sentinel and sentinel[0]["has_patch_id"] == "true" and sentinel[0]["has_filename"] == "true"
    assert all(r["notes"] for r in rows)
