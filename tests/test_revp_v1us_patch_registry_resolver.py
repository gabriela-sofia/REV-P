import csv, os, shutil
import scripts.protocolo_c.revp_v1us_common as common

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "v1us")


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    root = tmp_path / "datasets"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    for p in (data, docs, cfg):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DATASETS_ROOT", str(root))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    return data, root, docs, cfg


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def install_patch_registry(root):
    shutil.copy(os.path.join(FIX, "patch_registry_multiregion.csv"),
                os.path.join(str(root), common.PATCH_REGISTRY_SOURCES[0]))


def install_event_registry(data):
    shutil.copy(os.path.join(FIX, "event_registry_multiregion.csv"),
                os.path.join(str(data), common.EVENT_REGISTRY))


def build_chain(data, root):
    """Install fixtures and run resolver + candidate builder."""
    install_patch_registry(root)
    install_event_registry(data)
    common.run_patch_registry_resolver()
    return common.run_event_patch_candidate_builder()


def test_resolver_reads_real_registry_dedup_normalized(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    install_patch_registry(root)
    rows = common.run_patch_registry_resolver()
    ids = {r["patch_id"] for r in rows}
    assert ids == {"REC_01", "REC_02", "PET_01", "PET_02", "CUR_01"}  # dup REC_01 collapsed
    assert {r["region"] for r in rows} == {"REC", "PET", "CUR"}  # CURITIBA/RECIFE normalized
    assert all(r["has_sentinel_date"] == "false" for r in rows)
    assert all(r["resolution_status"] == "RESOLVED_PATCH_SENTINEL_DATE_MISSING" for r in rows)


def test_resolver_does_not_invent_patch_id_when_missing(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    rows = common.run_patch_registry_resolver()  # no registry installed
    assert len(rows) == 1
    assert rows[0]["resolution_status"] == "PATCH_REGISTRY_MISSING"
    assert rows[0]["patch_id"] == ""
