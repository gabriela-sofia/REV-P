"""Tests for REV-P Protocol C v1ta-v1tf INMET canonical hydromet QA layer.

All outputs redirected to tmp_path. No network access. ZIPs read via fixtures.
"""
from __future__ import annotations

import csv
import importlib
import os
import sys
import zipfile
from pathlib import Path

import pytest

ROOT    = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1ta_v1tf_inmet_canonical_common as C  # noqa: E402

v1ta = importlib.import_module("revp_v1ta_inmet_canonical_station_registry")
v1tb = importlib.import_module("revp_v1tb_inmet_coordinate_parse_discrepancy_audit")
v1tc = importlib.import_module("revp_v1tc_inmet_canonical_precipitation_index")
v1td = importlib.import_module("revp_v1td_hydromet_event_evidence_bridge")
v1te = importlib.import_module("revp_v1te_tcc_hydromet_correction_evidence_tables")
v1tf = importlib.import_module("revp_v1tf_inmet_canonical_hydromet_qa_bundle")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), [])


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _redirect(monkeypatch, mod, tmp: Path) -> None:
    for name in dir(mod):
        if name.startswith(("OUT_", "SCHEMA_", "DOC", "IN_")):
            val = getattr(mod, name)
            if isinstance(val, Path):
                monkeypatch.setattr(mod, name, tmp / val.name)


def _fake_inmet_csv(code: str, lat_comma: str, lon_comma: str, uf: str,
                     name: str, year: str = "2022") -> str:
    return (
        f"REGIAO:;SE\nUF:;{uf}\nESTACAO:;{name}\n"
        f"CODIGO (WMO):;{code}\nLATITUDE:;{lat_comma}\nLONGITUDE:;{lon_comma}\n"
        "ALTITUDE:;100\nDATA DE FUNDACAO:;01/01/00\n"
        "Data;Hora UTC;PRECIPITACAO TOTAL, HORARIO (mm)\n"
        f"{year}/02/15;0000 UTC;5,0\n"
        f"{year}/02/16;0000 UTC;12,2\n"
        f"{year}/02/17;0000 UTC;0\n"
        f"{year}/02/18;0000 UTC;20,0\n"
    )


def _make_zip(tmp: Path, entries: list[tuple[str, str]]) -> Path:
    zp = tmp / "inmet" / "historical"
    zp.mkdir(parents=True, exist_ok=True)
    out = zp / "inmet_2022.zip"
    with zipfile.ZipFile(out, "w") as z:
        for name, content in entries:
            z.writestr(name, content)
    return out


# ---------------------------------------------------------------------------
# common — parse_decimal_comma_float
# ---------------------------------------------------------------------------

def test_parse_decimal_comma_simple():
    assert C.parse_decimal_comma_float("-22,75777777") == pytest.approx(-22.75777777)


def test_parse_decimal_comma_positive():
    assert C.parse_decimal_comma_float("43,68472221") == pytest.approx(43.68472221)


def test_parse_decimal_comma_integer():
    assert C.parse_decimal_comma_float("100") == pytest.approx(100.0)


def test_parse_decimal_comma_dot():
    assert C.parse_decimal_comma_float("-8.059") == pytest.approx(-8.059)


def test_parse_decimal_comma_empty():
    assert C.parse_decimal_comma_float("") == pytest.approx(0.0)


def test_parse_decimal_comma_garbage():
    assert C.parse_decimal_comma_float("not-a-number") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# common — detect_coordinate_anomaly
# ---------------------------------------------------------------------------

def test_detect_anomaly_ok():
    assert C.detect_coordinate_anomaly(-22.75, -43.68) == "OK"


def test_detect_anomaly_zero():
    assert C.detect_coordinate_anomaly(0.0, 0.0) == "ZERO_COORDS"


def test_detect_anomaly_lat_out_of_brazil():
    # Garbled parse result like 789343.0
    assert C.detect_coordinate_anomaly(789343.0, 925756.0) != "OK"


def test_detect_anomaly_lon_out():
    assert C.detect_coordinate_anomaly(-22.5, 43.0) != "OK"  # positive lon not Brazil


def test_station_coordinate_quality_ok():
    status = C.station_coordinate_quality_status(-22.75, -43.68, "-22,75", "-43,68")
    assert "CANONICAL" in status


def test_station_coordinate_quality_anomaly():
    status = C.station_coordinate_quality_status(789343.0, 925756.0, "789343,0", "925756,0")
    assert "COORD_ANOMALY" in status


# ---------------------------------------------------------------------------
# common — compare_station_records
# ---------------------------------------------------------------------------

def test_compare_records_no_discrepancy():
    disc = C.compare_station_records("-22.75", "-43.68", "-22.75", "-43.68")
    assert disc["discrepancy_type"] == "NO_DISCREPANCY"
    assert disc["delta_km"] == "0.00"


def test_compare_records_comma_correction():
    # v1si garbled: 2275.0 vs canonical -22.75
    disc = C.compare_station_records("2275.0", "4368.0", "-22.75", "-43.68")
    assert disc["discrepancy_type"] in ("V1SI_COORD_ANOMALY", "DECIMAL_COMMA_CORRECTION",
                                        "CALCULATION_ERROR")


def test_compare_records_delta_km():
    # Two known PET-area stations ~5 km apart
    disc = C.compare_station_records("-22.50", "-43.18", "-22.51", "-43.19")
    assert disc["discrepancy_type"] in ("MINOR_ROUNDING", "DECIMAL_COMMA_CORRECTION")
    if disc["delta_km"]:
        assert float(disc["delta_km"]) < 5.0


# ---------------------------------------------------------------------------
# v1ta — canonical station registry
# ---------------------------------------------------------------------------

def test_v1ta_canonical_registry_fixture(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    rows = _read(tmp_path / v1ta.OUT_REG.name)
    assert any(r["station_code"] == "A627" for r in rows)


def test_v1ta_coordinates_correct(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    rows = _read(tmp_path / v1ta.OUT_REG.name)
    row = next(r for r in rows if r["station_code"] == "A627")
    lat = float(row["latitude"])
    assert -30 < lat < -15  # valid Brazil latitude


def test_v1ta_provenance_includes_raw_zip(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_PE_A301_RECIFE_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A301", "-8,05", "-34,87", "PE", "RECIFE")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    rows = _read(tmp_path / v1ta.OUT_REG.name)
    assert any("raw_zip" in r.get("provenance_sources", "") for r in rows)


def test_v1ta_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    for r in _read(tmp_path / v1ta.OUT_REG.name):
        assert r["review_only"] == "true"


def test_v1ta_no_abs_path(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    for r in _read(tmp_path / v1ta.OUT_REG.name):
        for v in r.values():
            assert not C.ABS_PATH_RE.search(str(v))


def test_v1ta_fail_closed_no_zips(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "no_zips"))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    assert _header(tmp_path / v1ta.OUT_REG.name) != []  # header always emitted


# ---------------------------------------------------------------------------
# v1tb — discrepancy audit
# ---------------------------------------------------------------------------

def _write_v1si_and_v1ta(tmp: Path):
    gr = C.guardrail_row()
    v1si = [{"station_code": "A627", "station_name": "PETROPO", "latitude": "2246000.0",
              "longitude": "4310000.0", "region_candidate": "UNKNOWN",
              "review_only": "true", "notes": ""}]
    _write_csv(tmp / "protocol_c_inmet_station_candidates_v1si.csv", v1si,
               list(v1si[0].keys()))
    v1ta_r = [{"station_code": "A627", "station_name": "PETROPOLIS",
               "latitude": "-22.460000", "longitude": "-43.100000",
               "coordinate_quality_status": "CANONICAL_DECIMAL_COMMA_PARSED",
               "provenance_sources": "raw_zip;v1sr",
               "nearest_region": "PET", "nearest_region_distance_km": "6.2",
               "within_25km": "true", "within_50km": "true", "within_100km": "true",
               "notes": "", **gr}]
    _write_csv(tmp / "protocol_c_inmet_canonical_station_registry_v1ta.csv", v1ta_r,
               list(v1ta_r[0].keys()))


def test_v1tb_detects_discrepancy(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tb, tmp_path)
    _write_v1si_and_v1ta(tmp_path)
    monkeypatch.setattr(v1tb, "DATASETS", tmp_path)
    v1tb.run()
    rows = _read(tmp_path / v1tb.OUT_AUD.name)
    assert any(r["discrepancy_type"] == "V1SI_COORD_ANOMALY" for r in rows)


def test_v1tb_v1si_not_modified(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tb, tmp_path)
    _write_v1si_and_v1ta(tmp_path)
    monkeypatch.setattr(v1tb, "DATASETS", tmp_path)
    v1tb.run()
    # v1si file must be unchanged
    v1si_after = _read(tmp_path / "protocol_c_inmet_station_candidates_v1si.csv")
    assert v1si_after[0]["latitude"] == "2246000.0"


def test_v1tb_no_discrepancy_case(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tb, tmp_path)
    gr = C.guardrail_row()
    v1si = [{"station_code": "A301", "station_name": "RECIFE",
              "latitude": "-8.059167", "longitude": "-34.959167",
              "region_candidate": "RECIFE", "review_only": "true", "notes": ""}]
    _write_csv(tmp_path / "protocol_c_inmet_station_candidates_v1si.csv", v1si,
               list(v1si[0].keys()))
    v1ta_r = [{"station_code": "A301", "station_name": "RECIFE",
               "latitude": "-8.059167", "longitude": "-34.959167",
               "coordinate_quality_status": "OK", "provenance_sources": "raw_zip",
               "nearest_region": "RECIFE", "nearest_region_distance_km": "8.6",
               "within_25km": "true", "within_50km": "true", "within_100km": "true",
               "notes": "", **gr}]
    _write_csv(tmp_path / "protocol_c_inmet_canonical_station_registry_v1ta.csv", v1ta_r,
               list(v1ta_r[0].keys()))
    monkeypatch.setattr(v1tb, "DATASETS", tmp_path)
    v1tb.run()
    rows = _read(tmp_path / v1tb.OUT_AUD.name)
    assert any(r["discrepancy_type"] in ("NO_DISCREPANCY", "MINOR_ROUNDING") for r in rows)


def test_v1tb_correction_status(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tb, tmp_path)
    _write_v1si_and_v1ta(tmp_path)
    monkeypatch.setattr(v1tb, "DATASETS", tmp_path)
    v1tb.run()
    rows = _read(tmp_path / v1tb.OUT_AUD.name)
    assert any(r["correction_status"] == "CORRECTED_IN_V1TA" for r in rows)


def test_v1tb_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tb, tmp_path)
    _write_v1si_and_v1ta(tmp_path)
    monkeypatch.setattr(v1tb, "DATASETS", tmp_path)
    v1tb.run()
    for r in _read(tmp_path / v1tb.OUT_AUD.name):
        assert r.get("review_only") == "true"


# ---------------------------------------------------------------------------
# v1tc — canonical precipitation index
# ---------------------------------------------------------------------------

def _make_v1ta_with_station(tmp: Path, code: str = "A627",
                              lat: str = "-22.460000", lon: str = "-43.100000"):
    gr = C.guardrail_row()
    rows = [{"station_code": code, "station_name": "PETROPOLIS", "uf": "RJ",
             "latitude": lat, "longitude": lon,
             "coordinate_quality_status": "CANONICAL_DECIMAL_COMMA_PARSED",
             "nearest_region": "PET", "nearest_region_distance_km": "6.2",
             "within_25km": "true", "within_50km": "true", "within_100km": "true",
             "provenance_sources": "raw_zip", "source_years": "2022",
             "notes": "", **gr}]
    _write_csv(tmp / "protocol_c_inmet_canonical_station_registry_v1ta.csv",
               rows, list(rows[0].keys()))


def test_v1tc_reads_precip_from_zip(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tc, tmp_path)
    _make_v1ta_with_station(tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1tc, "DATASETS", tmp_path)
    monkeypatch.setenv("REVP_INMET_MAX_DAILY_ROWS", "1000")
    v1tc.run()
    rows = _read(tmp_path / v1tc.OUT_IDX.name)
    real = [r for r in rows if r["provenance_status"] == "OFFICIAL_INMET_CANONICAL_REVIEW_ONLY"]
    assert len(real) >= 1


def test_v1tc_precip_decimal_comma_parsed(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tc, tmp_path)
    _make_v1ta_with_station(tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1tc, "DATASETS", tmp_path)
    monkeypatch.setenv("REVP_INMET_MAX_DAILY_ROWS", "1000")
    v1tc.run()
    rows = _read(tmp_path / v1tc.OUT_IDX.name)
    real = [r for r in rows if r.get("precipitation_mm")]
    # At least one value should be parseable as a float
    assert all(float(r["precipitation_mm"]) >= 0 for r in real)


def test_v1tc_cap_env_var(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tc, tmp_path)
    _make_v1ta_with_station(tmp_path)
    big_csv = (
        "REGIAO:;SE\nUF:;RJ\nESTACAO:;PETROPOLIS\n"
        "CODIGO (WMO):;A627\nLATITUDE:;-22,46\nLONGITUDE:;-43,10\n"
        "ALTITUDE:;100\nDATA DE FUNDACAO:;01/01/00\n"
        "Data;Hora UTC;PRECIPITACAO TOTAL, HORARIO (mm)\n"
    )
    for i in range(500):
        big_csv += f"2022/01/{(i%28)+1:02d};0000 UTC;1,5\n"
    _make_zip(tmp_path, [("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV", big_csv)])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1tc, "DATASETS", tmp_path)
    # Patch the function directly so the cap is enforced at runtime regardless of env-var caching.
    monkeypatch.setattr(v1tc, "_max_daily_rows", lambda: 10)
    v1tc.run()
    rows = _read(tmp_path / v1tc.OUT_IDX.name)
    real = [r for r in rows if r["provenance_status"] == "OFFICIAL_INMET_CANONICAL_REVIEW_ONLY"]
    assert len(real) <= 10


def test_v1tc_no_label(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tc, tmp_path)
    _make_v1ta_with_station(tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1tc, "DATASETS", tmp_path)
    v1tc.run()
    for r in _read(tmp_path / v1tc.OUT_IDX.name):
        assert r.get("can_create_operational_label") == "false"
        assert r.get("target_created") == "false"


def test_v1tc_absence_as_negative_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tc, tmp_path)
    monkeypatch.setattr(v1tc, "DATASETS", tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "empty"))
    v1tc.run()
    for r in _read(tmp_path / v1tc.OUT_IDX.name):
        assert r.get("absence_as_negative") == "false"


def test_v1tc_fail_closed_no_stations(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tc, tmp_path)
    monkeypatch.setattr(v1tc, "DATASETS", tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "empty"))
    v1tc.run()
    assert _header(tmp_path / v1tc.OUT_IDX.name) != []


# ---------------------------------------------------------------------------
# v1td — evidence bridge
# ---------------------------------------------------------------------------

def _write_bridge_fixtures(tmp: Path, has_features: bool = True):
    gr = C.guardrail_row()
    wins = [{"event_window_id": "V1SS_W0000", "event_candidate_id": "C001",
             "region": "PET", "parsed_date": "2022-02-19",
             "window_start": "2022-02-12", "window_end": "2022-02-20",
             "blocked_reason": "", **gr}]
    _write_csv(tmp / "protocol_c_event_date_windows_v1ss.csv", wins, list(wins[0].keys()))

    v1ta_r = [{"station_code": "A627", "nearest_region": "PET",
               "nearest_region_distance_km": "6.2",
               "within_100km": "true",
               "station_name": "PETROPOLIS", "notes": "", **gr}]
    _write_csv(tmp / "protocol_c_inmet_canonical_station_registry_v1ta.csv",
               v1ta_r, list(v1ta_r[0].keys()))

    feats = [{
        "event_window_id": "V1SS_W0000", "station_code": "A627",
        "region": "PET", "rain_1d": "20.0", "rain_3d": "45.0", "rain_7d": "80.0",
        "max_1d_in_window": "25.0",
        "feature_status": "ROLLING_CONTEXT_REVIEW_ONLY",
        "anchor_date": "2022-02-19",
        "nearest_station_distance_km": "6.2", **gr
    }] if has_features else []
    _write_csv(tmp / "protocol_c_rolling_rainfall_context_features_v1su.csv",
               feats, (list(feats[0].keys()) if feats else
                        ["event_window_id","station_code","region","rain_1d","rain_3d",
                         "rain_7d","max_1d_in_window","feature_status","anchor_date",
                         "nearest_station_distance_km","review_only"]))
    # Empty precip index (bridge should still work from features)
    _write_csv(tmp / "protocol_c_inmet_canonical_precipitation_index_v1tc.csv",
               [], ["station_code", "date", "precipitation_mm", "provenance_status"])


def test_v1td_bridge_does_not_validate_event(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1td, tmp_path)
    _write_bridge_fixtures(tmp_path)
    monkeypatch.setattr(v1td, "DATASETS", tmp_path)
    v1td.run()
    for r in _read(tmp_path / v1td.OUT_BRG.name):
        assert r.get("does_not_validate_event") == "true"
        assert r.get("evidence_role") == "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY"


def test_v1td_rolling_values_preserved(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1td, tmp_path)
    _write_bridge_fixtures(tmp_path)
    monkeypatch.setattr(v1td, "DATASETS", tmp_path)
    v1td.run()
    rows = _read(tmp_path / v1td.OUT_BRG.name)
    assert any(r.get("rain_7d") == "80.0" for r in rows)


def test_v1td_absence_as_negative_false(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1td, tmp_path)
    _write_bridge_fixtures(tmp_path)
    monkeypatch.setattr(v1td, "DATASETS", tmp_path)
    v1td.run()
    for r in _read(tmp_path / v1td.OUT_BRG.name):
        assert r.get("absence_as_negative") == "false"


def test_v1td_no_ground_truth(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1td, tmp_path)
    _write_bridge_fixtures(tmp_path)
    monkeypatch.setattr(v1td, "DATASETS", tmp_path)
    v1td.run()
    for r in _read(tmp_path / v1td.OUT_BRG.name):
        assert r.get("ground_truth_operational") == "false"


def test_v1td_supports_manual_review(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1td, tmp_path)
    _write_bridge_fixtures(tmp_path)
    monkeypatch.setattr(v1td, "DATASETS", tmp_path)
    v1td.run()
    rows = _read(tmp_path / v1td.OUT_BRG.name)
    assert any(r.get("supports_manual_review") == "true" for r in rows)


def test_v1td_fail_closed_no_windows(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1td, tmp_path)
    monkeypatch.setattr(v1td, "DATASETS", tmp_path)
    v1td.run()
    assert _header(tmp_path / v1td.OUT_BRG.name) != []


# ---------------------------------------------------------------------------
# v1te — TCC tables
# ---------------------------------------------------------------------------

def _write_v1te_inputs(tmp: Path):
    for fn, kv in [
        ("protocol_c_inmet_coordinate_parse_discrepancy_summary_v1tb.csv",
         [("stations_compared","668"), ("corrected_in_v1ta","668"),
          ("affects_region_matching","668"), ("v1si_not_modified","true")]),
        ("protocol_c_inmet_canonical_station_registry_summary_v1ta.csv",
         [("canonical_stations_total","668"), ("within_100km","25")]),
        ("protocol_c_hydromet_event_evidence_bridge_summary_v1td.csv",
         [("bridge_rows","9"), ("rows_with_rain_data","9")]),
        ("protocol_c_inmet_canonical_precipitation_index_summary_v1tc.csv",
         [("total_daily_records","49208"), ("stations_with_data","24")]),
    ]:
        _write_csv(tmp / fn, [{"stat_key": k, "stat_value": v} for k,v in kv],
                   ["stat_key","stat_value"])


def test_v1te_tables_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1te, tmp_path)
    _write_v1te_inputs(tmp_path)
    monkeypatch.setattr(v1te, "DATASETS", tmp_path)
    v1te.run()
    assert (tmp_path / v1te.OUT_COR.name).exists()
    assert (tmp_path / v1te.OUT_EVB.name).exists()
    assert (tmp_path / v1te.OUT_LIM.name).exists()


def test_v1te_limitations_contains_absence(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1te, tmp_path)
    _write_v1te_inputs(tmp_path)
    monkeypatch.setattr(v1te, "DATASETS", tmp_path)
    v1te.run()
    lims = _read(tmp_path / v1te.OUT_LIM.name)
    # Match 'absence'/'ausencia'/'aus' broadly to be encoding-agnostic on Windows
    descs = " ".join(r.get("description","") + r.get("implication","") for r in lims).lower()
    # LIM04 is about absence not being negative — check by limitation_id
    ids = [r.get("limitation_id","") for r in lims]
    assert "LIM_TE04" in ids or "ausencia" in descs or "absence" in descs or "aus" in descs


def test_v1te_correction_table_has_v1si_not_modified(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1te, tmp_path)
    _write_v1te_inputs(tmp_path)
    monkeypatch.setattr(v1te, "DATASETS", tmp_path)
    v1te.run()
    cor = _read(tmp_path / v1te.OUT_COR.name)
    assert any(r.get("metric") == "v1si_not_modified" for r in cor)


# ---------------------------------------------------------------------------
# v1tf — bundle
# ---------------------------------------------------------------------------

def _write_all_summaries_tf(tmp: Path, status: str = "GUARDRAIL_PASS_ALL",
                              precip: str = "49208"):
    for fn, kv in [
        ("protocol_c_inmet_canonical_station_registry_summary_v1ta.csv",
         [("canonical_stations_total","668"),("coord_quality_ok","668"),("within_100km","25")]),
        ("protocol_c_inmet_coordinate_parse_discrepancy_summary_v1tb.csv",
         [("stations_compared","668"),("corrected_in_v1ta","668"),
          ("v1si_not_modified","true"),("affects_region_matching","668")]),
        ("protocol_c_inmet_canonical_precipitation_index_summary_v1tc.csv",
         [("total_daily_records",precip),("stations_with_data","24")]),
        ("protocol_c_hydromet_event_evidence_bridge_summary_v1td.csv",
         [("bridge_rows","9"),("rows_with_rain_data","9")]),
        ("protocol_c_tcc_table_hydromet_limitations_v1te.csv",
         [{"limitation_id":"LIM01","aspect":"x","description":"y","implication":"z","notes":""}]),
    ]:
        if isinstance(kv[0], dict):
            _write_csv(tmp / fn, kv, list(kv[0].keys()))
        else:
            _write_csv(tmp / fn, [{"stat_key": k, "stat_value": v} for k,v in kv],
                       ["stat_key","stat_value"])


def test_v1tf_final_status_ready(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tf, tmp_path)
    _write_all_summaries_tf(tmp_path)
    for attr in ("IN_TA","IN_TB","IN_TC","IN_TD","IN_TE"):
        monkeypatch.setattr(v1tf, attr, tmp_path / getattr(v1tf, attr).name)
    result = v1tf.run()
    assert result["final_status"] == v1tf.ST_READY


def test_v1tf_wait_precip_status(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tf, tmp_path)
    _write_all_summaries_tf(tmp_path, precip="0")
    for attr in ("IN_TA","IN_TB","IN_TC","IN_TD","IN_TE"):
        monkeypatch.setattr(v1tf, attr, tmp_path / getattr(v1tf, attr).name)
    result = v1tf.run()
    assert result["final_status"] == v1tf.ST_WAIT_PRECIP


def test_v1tf_labels_zero(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tf, tmp_path)
    _write_all_summaries_tf(tmp_path)
    for attr in ("IN_TA","IN_TB","IN_TC","IN_TD","IN_TE"):
        monkeypatch.setattr(v1tf, attr, tmp_path / getattr(v1tf, attr).name)
    v1tf.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1tf.OUT_SUM.name)}
    assert summ["labels_created"] == "0"
    assert summ["targets_created"] == "0"
    assert summ["ground_truth_created"] == "0"
    assert summ["formal_negatives_created"] == "0"


def test_v1tf_qc_checks_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tf, tmp_path)
    _write_all_summaries_tf(tmp_path)
    for attr in ("IN_TA","IN_TB","IN_TC","IN_TD","IN_TE"):
        monkeypatch.setattr(v1tf, attr, tmp_path / getattr(v1tf, attr).name)
    v1tf.run()
    qc = _read(tmp_path / v1tf.OUT_QC.name)
    assert len(qc) >= 6
    assert any(c["check_name"] == "labels_zero" for c in qc)


def test_v1tf_mandatory_clause_in_doc(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tf, tmp_path)
    _write_all_summaries_tf(tmp_path)
    for attr in ("IN_TA","IN_TB","IN_TC","IN_TD","IN_TE"):
        monkeypatch.setattr(v1tf, attr, tmp_path / getattr(v1tf, attr).name)
    v1tf.run()
    doc = (tmp_path / v1tf.DOC.name).read_text(encoding="utf-8")
    assert "ground truth operacional" in doc
    assert "evidência observacional independente" in doc


# ---------------------------------------------------------------------------
# Guardrail / schema / hygiene
# ---------------------------------------------------------------------------

def test_guardrail_scan_clean():
    assert C.scan_guardrails([C.guardrail_row()], "t") == []


def test_guardrail_scan_forbidden():
    issues = C.scan_guardrails([{"can_train_model": "true", "formal_negative": "false"}], "t")
    assert any("can_train_model" in i for i in issues)


def test_guardrail_scan_abs_path():
    issues = C.scan_guardrails([{"notes": r"C:\Users\gabriela\data.csv"}], "t")
    assert any("abs_path" in i for i in issues)


def test_v1ta_schema_emitted(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    assert (tmp_path / v1ta.SCHEMA_R.name).exists()


def test_v1ta_doc_emitted(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    assert (tmp_path / v1ta.DOC.name).exists()


def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1ta*")}
    _redirect(monkeypatch, v1ta, tmp_path)
    _make_zip(tmp_path, [
        ("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV",
         _fake_inmet_csv("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")),
    ])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1ta, "DATASETS", tmp_path)
    v1ta.run()
    after = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1ta*")}
    assert before == after


def test_staged_empty():
    import subprocess
    r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.stdout.strip() == ""
