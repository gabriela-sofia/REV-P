import csv, os, zipfile
import scripts.protocolo_c.revp_v1ur_petropolis_common as common

def set_env(tmp_path, monkeypatch):
    data=tmp_path/"datasets"/"protocolo_c"; docs=tmp_path/"docs"/"metodologia_cientifica"; cfg=tmp_path/"configs"/"protocolo_c"; raw=tmp_path/"raw"; st=tmp_path/"staging"
    for p in [data,docs,cfg,raw,st]: p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common,"DATASET_DIR",str(data)); monkeypatch.setattr(common,"DOCS_DIR",str(docs)); monkeypatch.setattr(common,"CONFIG_DIR",str(cfg)); monkeypatch.setattr(common,"LOCAL_RAW_DIR",str(raw)); monkeypatch.setattr(common,"LOCAL_STAGING_DIR",str(st)); monkeypatch.setattr(common,"LOCAL_QUARANTINE_DIR",str(tmp_path/"q")); monkeypatch.setattr(common,"LOCAL_REPORTS_DIR",str(tmp_path/"r"))
    return data,docs,cfg,raw

def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)

def test_seed_builder_uses_missing_geodata_signals(tmp_path, monkeypatch):
    data,_,_,_=set_env(tmp_path,monkeypatch)
    rows=[{"missing_geodata_id":f"m{i}","event_id":"PET_2022_02_15","asset_id":"a","page_number":str(i),"signal_class":"DIGITAL_GEODATA_REFERENCE","signal_strength":"STRONG","referenced_artifact_type":"geodata_or_sig_asset","public_path_hint":"SGB","can_be_resolved_by_public_search":"true","recommended_next_query":"Petropolis SGB shapefile SIG camada geodados","notes":""} for i in range(44)]
    write_csv(data/"v1uq_petropolis_missing_geodata_signal_audit.csv", common.MISSING_GEODATA_COLUMNS, rows)
    out=common.run_geodata_signal_seed_builder()
    assert len(out)==44
    assert all(r["can_create_ground_reference"]=="false" for r in out)
