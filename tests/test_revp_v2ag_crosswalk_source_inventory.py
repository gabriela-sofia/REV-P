import csv
import os
from pathlib import Path

import scripts.protocolo_c.revp_v2ag_common as common


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = Path("datasets") / "protocolo_c"
    docs = Path("docs") / "metodologia_cientifica"
    cfg = Path("configs") / "protocolo_c"
    for p in (data, docs, cfg):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "STAGING_DIR", str(Path("local_only") / "v2ag" / "staging"))
    monkeypatch.setattr(common, "REPORTS_DIR", str(Path("local_only") / "v2ag" / "reports"))
    monkeypatch.setattr(common, "ROOT_SCAN_DIRS", ["datasets", "configs", "docs"])
    return data


def install_packages(data):
    write_csv(
        data / "v2ac_event_patch_v2_package_registry.csv",
        [
            "event_patch_candidate_id", "event_id", "event_region", "patch_id",
            "date_linkability_status", "sentinel_date_status", "explicit_crosswalk_id",
            "crosswalk_status", "can_create_ground_reference",
            "can_create_training_label", "ground_truth_operational",
            "crosswalk_inferred", "sentinel_date_inferred",
        ],
        [
            {
                "event_patch_candidate_id": "EPC_SYN_001",
                "event_id": "REC_2022_05_24_30",
                "event_region": "REC",
                "patch_id": "REC_00001",
                "date_linkability_status": "UNLINKABLE_NAMESPACE",
                "sentinel_date_status": "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE",
                "explicit_crosswalk_id": "XW_DINO::REC_00001",
                "crosswalk_status": "EXPLICIT_DINO_CROSSWALK_NO_ANCHOR_CROSSWALK",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "ground_truth_operational": "false",
                "crosswalk_inferred": "false",
                "sentinel_date_inferred": "false",
            },
            {
                "event_patch_candidate_id": "EPC_SYN_002",
                "event_id": "PET_2022_02_15",
                "event_region": "PET",
                "patch_id": "PET_00002",
                "date_linkability_status": "UNLINKABLE_NAMESPACE",
                "sentinel_date_status": "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE",
                "explicit_crosswalk_id": "",
                "crosswalk_status": "NO_EXPLICIT_ANCHOR_CROSSWALK",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "ground_truth_operational": "false",
                "crosswalk_inferred": "false",
                "sentinel_date_inferred": "false",
            },
        ],
    )
    write_csv(
        data / "v2aa_patch_date_candidate_consolidation.csv",
        ["patch_date_id", "patch_id", "region", "selected_sentinel_date", "conflict_status", "sentinel_date_recovered", "sentinel_date_inferred"],
        [
            {"patch_date_id": "PD_SYN_001", "patch_id": "REFPATCH_SYN_DATE_001", "region": "REC", "selected_sentinel_date": "2022-05-25", "conflict_status": "NONE", "sentinel_date_recovered": "true", "sentinel_date_inferred": "false"},
            {"patch_date_id": "PD_SYN_002", "patch_id": "REFPATCH_UNLINKABLE_001", "region": "PET", "selected_sentinel_date": "2022-02-02", "conflict_status": "NONE", "sentinel_date_recovered": "true", "sentinel_date_inferred": "false"},
        ],
    )


def test_source_inventory_excludes_local_non_versionable_dirs(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    write_csv(data / "candidate_registry.csv", ["event_patch_candidate_id", "patch_id", "refpatch_id"], [{"event_patch_candidate_id": "EPC_SYN_001", "patch_id": "REC_00001", "refpatch_id": "REFPATCH_SYN_DATE_001"}])
    write_csv(Path("local_only") / "protocolo_c" / "bad.csv", ["patch_id"], [{"patch_id": "REC_99999"}])
    rows = common.run_crosswalk_source_inventory(common.parse_args([]))
    paths = [r["registry_path"] for r in rows]
    assert any(p.endswith("candidate_registry.csv") for p in paths)
    assert all("local_only" not in p for p in paths)
    assert any(r["should_extract_keys"] == "true" for r in rows)
