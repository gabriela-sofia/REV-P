import csv, datetime as dt, os, shutil, zipfile
import pytest
import scripts.protocolo_c.revp_v2ay_common as common

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def install(tmp_path,monkeypatch):
    p=tmp_path/"datasets/protocolo_c";d=tmp_path/"docs/protocolo_c/v2ay_hydromet_series_ingestion";c=d/"evidence_cache";r=c/"raw"
    for x in (p,d,c,r):x.mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(common,"DATASET_DIR",str(p));monkeypatch.setattr(common,"DOCS_DIR",str(d));monkeypatch.setattr(common,"CACHE_DIR",str(c));monkeypatch.setattr(common,"RAW_DIR",str(r))
    data={"sources":[{"source_id":"INMET"}],"windows":[{"assertion_id":"A1","candidate_id":"REC_2023_02_05_06","event_id":"REC_2023_02_05_06","window_start":"2023-01-29","event_date":"2023-02-05","window_end":"2023-02-09","temporal_window_defined":"true"}],"acquisition":[{"source_id":"INMET"}],"linkage":[{"assertion_id":"A1","candidate_id":"REC_1"}],"readiness":[{"assertion_id":"A1"}],"provenance":[{"source_id":"INMET"}]}
    for k,n in common.INPUTS.items():write_csv(p/n,data[k])
    common.ensure_cache_policy();return p,d,c,r

@pytest.mark.parametrize("name,expected",[("inmet_2023.csv","INMET"),("cemaden.csv","CEMADEN"),("hidroweb.zip","ANA_HIDROWEB"),("ana_data.csv","ANA_HIDROWEB"),("other.bin","UNKNOWN")])
def test_infer_source(name,expected):assert common.infer_source(name)==expected

@pytest.mark.parametrize("name,expected",[("a.csv","SUPPORTED_RAW_CSV"),("a.zip","SUPPORTED_RAW_ZIP"),("a.txt","UNSUPPORTED_RAW_FILE")])
def test_inventory_status(name,expected):assert common.inventory_status(name)==expected

@pytest.mark.parametrize("field,expected",list(common.INVARIANTS.items()))
def test_invariants(field,expected):assert common.with_invariants({})[field]==expected

@pytest.mark.parametrize("parsed,window,missing,precip,signal,expected",[(True,True,.1,True,"PRECIPITATION_PRESENT","TEMPORAL_EVIDENCE_READY_FOR_REVIEW"),(False,True,.1,True,"PRECIPITATION_PRESENT","TEMPORAL_EVIDENCE_NOT_READY"),(True,False,.1,True,"PRECIPITATION_PRESENT","TEMPORAL_EVIDENCE_NOT_READY"),(True,True,.3,True,"PRECIPITATION_PRESENT","TEMPORAL_EVIDENCE_NOT_READY"),(True,True,.1,False,"UNKNOWN","TEMPORAL_EVIDENCE_NOT_READY"),(True,True,.1,True,"UNKNOWN","TEMPORAL_EVIDENCE_NOT_READY")])
def test_readiness(parsed,window,missing,precip,signal,expected):assert common.readiness(parsed,window,missing,precip,signal)==expected

@pytest.mark.parametrize("candidate,station",[("REC_2023_02_05_06","RECIFE"),("PET_2022_02_15","PICO DO COUTO"),("CTB_2024_02_18_20","CURITIBA"),("X","")])
def test_station_name_for_candidate(candidate,station):assert common.station_name_for_candidate(candidate)==station

@pytest.mark.parametrize("fixture,source,count",[("inmet_valid.csv","INMET",2),("cemaden_valid.csv","CEMADEN",1),("hidroweb_valid.csv","ANA_HIDROWEB",1)])
def test_valid_parsers(tmp_path,monkeypatch,fixture,source,count):
    p,d,c,r=install(tmp_path,monkeypatch);target=r/(source.lower()+".csv");shutil.copy("tests/fixtures/v2ay/"+fixture,target);rows=common.parse_raw_file(str(target));assert len(rows)==count and all(x["parse_status"]=="PARSED" for x in rows)

def test_zip_csv(tmp_path,monkeypatch):
    p,d,c,r=install(tmp_path,monkeypatch);z=r/"inmet.zip"
    with zipfile.ZipFile(z,"w") as a:a.write("tests/fixtures/v2ay/inmet_valid.csv","data.csv")
    assert len(common.parse_raw_file(str(z)))==2

def test_empty_cache(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch);assert common.run_discover_cached_timeseries()[0]["inventory_status"]=="NO_RAW_TIMESERIES_AVAILABLE"

def test_unknown_file(tmp_path,monkeypatch):
    p,d,c,r=install(tmp_path,monkeypatch);(r/"x.bin").write_bytes(b"x");assert common.run_discover_cached_timeseries()[0]["inventory_status"]=="UNSUPPORTED_RAW_FILE"

def test_missing_fields():
    row=common.normalize_row({"foo":"bar"},"INMET","h");assert row["parse_status"]=="MISSING_REQUIRED_FIELDS"

@pytest.mark.parametrize("row,value",[
    ({"timestamp":"2023-01-01","precipitation":"1"},"1.0"),
    ({"datahora":"2023-01-01","valor":"2,5"},"2.5"),
    ({"data":"2023-01-01","chuva":"0"},"0.0"),
    ({"date":"2023-01-01","observed_value":"3"},"3.0"),
])
def test_normalized_aliases(row,value):
    normalized=common.normalize_row(row,"INMET","hash");assert normalized["parse_status"]=="PARSED" and normalized["observed_value"]==value

@pytest.mark.parametrize("value,expected",[
    ("2023-02-05",dt.date(2023,2,5)),
    ("05/02/2023",dt.date(2023,2,5)),
    ("2023/02/05",dt.date(2023,2,5)),
    ("invalid",None),
])
def test_parse_date_formats(value,expected):
    assert common.parse_date(value)==expected

@pytest.mark.parametrize("available,expected_missing",[(24,"0.000"),(20,"0.167"),(0,"1.000")])
def test_window_missing_rates(available,expected_missing):
    rows=[{"timestamp":f"2023-02-05 {hour%24:02d}:00","observed_value":"1","parse_status":"PARSED"} for hour in range(available)]
    assert common.window_metrics(rows,{"window_start":"2023-02-05","window_end":"2023-02-05"})["missing_rate"]==expected_missing

def test_window_metrics_strong_precip():
    rows=[{"timestamp":"2023-02-05 00:00","observed_value":"30","parse_status":"PARSED"},{"timestamp":"2023-02-05 01:00","observed_value":"20","parse_status":"PARSED"}];m=common.window_metrics(rows,{"window_start":"2023-02-05","window_end":"2023-02-05"});assert m["precip_total_window"]=="50.000" and m["precip_max_24h"]=="50.000" and m["precip_signal_status"]=="PRECIPITATION_PRESENT"

def test_max_rolling_24h_excludes_older_observations():
    rows=[{"timestamp":"2023-02-05 00:00","observed_value":"30","parse_status":"PARSED"},{"timestamp":"2023-02-06 00:00","observed_value":"20","parse_status":"PARSED"}]
    assert common.max_rolling_24h(rows)==30.0

def test_absent_precip_no_negative():
    rows=[{"timestamp":"2023-02-05","observed_value":"0","parse_status":"PARSED"}];m=common.window_metrics(rows,{"window_start":"2023-02-05","window_end":"2023-02-05"});assert m["precip_signal_status"]=="NO_PRECIPITATION_OBSERVED"

def test_no_raw_orchestrator(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch);rows=common.run_orchestrator();assert len(rows)==8 and all(x["status"]=="OK" for x in rows);report=common.load_csv(common.dataset_path("v2ay_ingestion_gap_report.csv"));assert report[0]["value"]=="NO_RAW_TIMESERIES_AVAILABLE" and report[-1]["value"]=="MANUALLY_DOWNLOAD_PUBLIC_HYDROMET_TIMESERIES"

def test_ready_never_truth():
    row=common.with_invariants({"temporal_readiness_status":"TEMPORAL_EVIDENCE_READY_FOR_REVIEW"});assert row["can_create_ground_truth"]=="false" and row["can_create_patch_truth"]=="false"

def test_cache_markers(tmp_path,monkeypatch):
    p,d,c,r=install(tmp_path,monkeypatch);assert (c/".gitignore").read_text()=="*\n!.gitignore\n!raw/\n!raw/.gitignore\n" and (r/".gitignore").read_text()=="*\n!.gitignore\n"

def test_guardrail(tmp_path,monkeypatch):
    p,d,c,r=install(tmp_path,monkeypatch);write_csv(p/"v2ay_bad.csv",[{**common.INVARIANTS,"can_create_negative":"true"}])
    with pytest.raises(ValueError):common.run_guardrail_regression()
