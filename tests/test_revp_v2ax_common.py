import csv
import os
import pytest
import scripts.protocolo_c.revp_v2ax_common as common

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def install(tmp_path, monkeypatch):
    p=tmp_path/"datasets/protocolo_c"; d=tmp_path/"docs/protocolo_c/v2ax_hydrometeorological_temporal_evidence"; c=d/"evidence_cache"
    for x in (p,d,c): x.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common,"DATASET_DIR",str(p)); monkeypatch.setattr(common,"DOCS_DIR",str(d)); monkeypatch.setattr(common,"CACHE_DIR",str(c)); monkeypatch.delenv(common.NETWORK_ENV,raising=False)
    data={
      "targets":[{"source_id":"INMET","source_role":"OFFICIAL_OBSERVED_TIME_SERIES"}],
      "plan":[{"source_id":"INMET","official_url":"https://example.test","observation_type":"meteorological"}],
      "windows":[{"assertion_id":"A1","candidate_id":"REC_2023_02_05_06","event_id":"REC_2023_02_05_06"}],
      "packages":[{"assertion_id":"A1","candidate_id":"REC_2023_02_05_06","event_id":"REC_2023_02_05_06","patch_id":""}],
      "provenance":[{"source_id":"INMET","provenance_status":"PUBLIC_SOURCE_PROVENANCE_PENDING"}]}
    for k,n in common.INPUTS.items(): write_csv(p/n,data[k])
    return p,d,c

with open(os.path.join("tests","fixtures","v2ax","cases.csv"),encoding="utf-8",newline="") as f: CASES=list(csv.DictReader(f))

@pytest.mark.parametrize("case",CASES,ids=lambda c:f"{c['kind']}-{c['expected']}")
def test_cases(case):
    if case["kind"]=="window": result=common.parse_event_window(case["a"])["temporal_precision"]
    elif case["kind"]=="quality": result=common.quality_status(case["a"],case["b"],case["c"]=="true")[1]
    else: result=common.readiness(case["a"]=="true",case["b"]=="true",case["c"]=="true",float(case["d"]),True)
    assert result==case["expected"]

@pytest.mark.parametrize("field,expected",list(common.INVARIANTS.items()))
def test_invariants(field,expected): assert common.with_invariants({})[field]==expected

@pytest.mark.parametrize("expected,available,status",[(10,10,"QUALITY_ACCEPTABLE_FOR_REVIEW"),(10,8,"QUALITY_ACCEPTABLE_FOR_REVIEW"),(10,7,"QUALITY_INCOMPLETE"),(0,0,"QUALITY_INCOMPLETE")])
def test_quality_boundaries(expected,available,status): assert common.quality_status(expected,available,True)[1]==status

@pytest.mark.parametrize("source,expected", [("INMET","PUBLIC_TIMESERIES"),("CEMADEN","PUBLIC_TIMESERIES"),("ANA_HIDROWEB","DYNAMIC_MANUAL")])
def test_source_types(tmp_path,monkeypatch,source,expected):
    p,d,c=install(tmp_path,monkeypatch); rows=common.load_csv(common.dataset_path(common.INPUTS["plan"])); rows[0]["source_id"]=source; write_csv(p/common.INPUTS["plan"],rows)
    prov=common.load_csv(p/common.INPUTS["provenance"]); prov[0]["source_id"]=source; write_csv(p/common.INPUTS["provenance"],prov)
    assert common.run_build_hydromet_source_registry()[0]["source_type"]==expected

def test_offline_manifest(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch); common.run_build_hydromet_source_registry(); row=common.run_acquire_timeseries_manifest()[0]; assert row["acquisition_status"]=="NETWORK_DISABLED_DETERMINISTIC_RUN"

def test_window_minus7_plus3(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch); row=common.run_build_event_temporal_windows()[0]; assert row["window_start"]=="2023-01-29" and row["window_end"]=="2023-02-09"

@pytest.mark.parametrize("event_id,event_date,window_end",[
    ("PET_2022_02_15","2022-02-15","2022-02-18"),
    ("PET_2022_03_20_21","2022-03-20","2022-03-24"),
    ("REC_2022_05_24_30","2022-05-24","2022-06-02"),
])
def test_event_window_dates(event_id,event_date,window_end):
    row=common.parse_event_window(event_id); assert row["event_date"]==event_date and row["window_end"]==window_end

def test_unknown_window_incomplete():
    row=common.parse_event_window("UNKNOWN"); assert row["temporal_window_defined"]=="false" and row["temporal_precision"]=="UNKNOWN"

def test_readiness_requires_precipitation():
    assert common.readiness(True,True,True,0.0,False)=="TEMPORAL_EVIDENCE_NOT_READY"

def test_readiness_does_not_change_truth():
    row=common.with_invariants({"temporal_readiness_status":"TEMPORAL_EVIDENCE_READY_FOR_REVIEW"})
    assert row["can_create_ground_truth"]=="false" and row["can_create_patch_truth"]=="false"

def test_station_nearby_not_proof(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch); common.run_build_hydromet_source_registry(); assert common.run_build_station_candidates()[0]["nearby_station_proves_patch_event"]=="false"

def test_precip_absence_not_negative(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch); common.run_build_event_temporal_windows(); assert common.run_summarize_precipitation_events()[0]["absence_of_rain_creates_negative"]=="false"

def test_link_geometry_blocking(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch); common.run_build_event_temporal_windows(); row=common.run_link_event_patch_temporal_evidence()[0]; assert row["geometry_still_blocking_truth"]=="true" and row["can_create_patch_truth"]=="false"

def test_orchestrator(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch); rows=common.run_orchestrator(); assert len(rows)==10 and all(r["status"]=="OK" for r in rows)

def test_cache(tmp_path,monkeypatch):
    install(tmp_path,monkeypatch); assert open(common.ensure_cache_policy(),encoding="utf-8").read()=="*\n!.gitignore\n"

def test_guardrail_rejects_truth(tmp_path,monkeypatch):
    p,d,c=install(tmp_path,monkeypatch); write_csv(p/"v2ax_bad.csv",[{**common.INVARIANTS,"can_create_ground_truth":"true"}])
    with pytest.raises(ValueError): common.run_guardrail_regression()
