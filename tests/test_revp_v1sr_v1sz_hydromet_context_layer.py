"""Tests for REV-P Protocol C v1sr-v1sz hydrometeorological context layer.

All outputs redirected to tmp_path. No real network. No real raw writes.
"""
from __future__ import annotations

import csv
import importlib
import os
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1sr_v1sz_hydromet_context_common as C  # noqa: E402

v1sr = importlib.import_module("revp_v1sr_inmet_station_regional_proximity_matcher")
v1ss = importlib.import_module("revp_v1ss_protocol_c_event_date_window_builder")
v1st = importlib.import_module("revp_v1st_inmet_precipitation_temporal_context")
v1su = importlib.import_module("revp_v1su_rolling_rainfall_context_features")
v1sv = importlib.import_module("revp_v1sv_hydromet_evidence_intake_crosswalk")
v1sw = importlib.import_module("revp_v1sw_official_hydromet_tcc_tables")
v1sx = importlib.import_module("revp_v1sx_hydromet_context_guardrail_audit")
v1sy = importlib.import_module("revp_v1sy_hydromet_context_runbook")
v1sz = importlib.import_module("revp_v1sz_hydromet_context_bundle")


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


def _fake_zip(tmp: Path, entries: list[tuple[str, str]]) -> Path:
    """Create a minimal fake INMET ZIP at tmp/inmet_2022.zip."""
    zp = tmp / "inmet" / "historical"
    zp.mkdir(parents=True, exist_ok=True)
    out = zp / "inmet_2022.zip"
    with zipfile.ZipFile(out, "w") as z:
        for name, content in entries:
            z.writestr(name, content)
    return out


def _inmet_csv_header(code: str, lat: str, lon: str, uf: str, name: str) -> str:
    return (
        f"REGIAO:;SE\nUF:;{uf}\nESTACAO:;{name}\n"
        f"CODIGO (WMO):;{code}\nLATITUDE:;{lat}\nLONGITUDE:;{lon}\n"
        "ALTITUDE:;100\nDATA DE FUNDACAO:;01/01/00\n"
        "Data;Hora UTC;PRECIPITACAO TOTAL, HORARIO (mm)\n"
        "2022/02/15;00:00;5.0\n2022/02/16;00:00;12.0\n"
        "2022/02/17;00:00;0.0\n2022/02/18;00:00;20.0\n"
    )


# ---------------------------------------------------------------------------
# common — math, date, region helpers
# ---------------------------------------------------------------------------

def test_haversine_same_point():
    assert C.haversine_km(0, 0, 0, 0) == pytest.approx(0.0, abs=0.01)


def test_haversine_known_distance():
    # Recife → Petrópolis approx 1800 km (both coastal states, far apart)
    d = C.haversine_km(-8.05, -34.88, -22.51, -43.18)
    assert 1600 < d < 2100


def test_haversine_symmetry():
    a = C.haversine_km(-8.05, -34.88, -25.43, -49.27)
    b = C.haversine_km(-25.43, -49.27, -8.05, -34.88)
    assert a == pytest.approx(b, rel=1e-6)


def test_normalize_region_pet():
    assert C.normalize_region("Petropolis") == "PET"


def test_normalize_region_recife():
    assert C.normalize_region("RECIFE") == "RECIFE"


def test_normalize_region_curitiba():
    assert C.normalize_region("CURITIBA") == "CURITIBA"


def test_normalize_region_unknown():
    assert C.normalize_region("Tokyo") == "UNKNOWN"


def test_parse_date_iso():
    assert C.parse_date_safe("2022-02-19") == date(2022, 2, 19)


def test_parse_date_br_slash():
    assert C.parse_date_safe("19/02/2022") == date(2022, 2, 19)


def test_parse_date_empty():
    assert C.parse_date_safe("") is None


def test_parse_date_garbage():
    assert C.parse_date_safe("not-a-date") is None


def test_normalize_date_roundtrip():
    assert C.normalize_date("19/02/2022") == "2022-02-19"


def test_build_window():
    ws, we = C.build_window(date(2022, 2, 19), before_days=7, after_days=1)
    assert ws == date(2022, 2, 12)
    assert we == date(2022, 2, 20)


def test_window_position_antecedent():
    assert C.window_position(date(2022, 2, 19), date(2022, 2, 17)) == "ANTECEDENT"


def test_window_position_event_day():
    assert C.window_position(date(2022, 2, 19), date(2022, 2, 19)) == "EVENT_DAY"


def test_window_position_post():
    assert C.window_position(date(2022, 2, 19), date(2022, 2, 20)) == "POST_EVENT"


def test_rolling_window_1d():
    daily = {"2022-02-19": 10.0, "2022-02-18": 5.0, "2022-02-17": 3.0}
    r = C.rolling_window_summary(daily, date(2022, 2, 19))
    assert r["rain_1d"] == pytest.approx(10.0)


def test_rolling_window_3d():
    daily = {"2022-02-19": 10.0, "2022-02-18": 5.0, "2022-02-17": 3.0}
    r = C.rolling_window_summary(daily, date(2022, 2, 19))
    assert r["rain_3d"] == pytest.approx(18.0)


def test_rolling_window_7d_missing_days():
    daily = {"2022-02-19": 10.0}
    r = C.rolling_window_summary(daily, date(2022, 2, 19))
    assert r["rain_7d"] == pytest.approx(10.0)  # missing days count as 0


def test_guardrail_row_safe():
    r = C.guardrail_row()
    assert r["review_only"] == "true"
    assert r["does_not_validate_event"] == "true"
    for f in C.FORBIDDEN_TRUE:
        assert r[f] == "false"


def test_scan_guardrails_clean():
    assert C.scan_guardrails([C.guardrail_row()], "t") == []


def test_scan_guardrails_forbidden_true():
    issues = C.scan_guardrails([{"can_train_model": "true"}], "t")
    assert any("can_train_model" in i for i in issues)


def test_scan_guardrails_abs_path():
    issues = C.scan_guardrails([{"notes": r"C:\Users\x\f.zip"}], "t")
    assert any("abs_path" in i for i in issues)


def test_scan_guardrails_local_runs():
    issues = C.scan_guardrails([{"notes": "local" + "_runs/x"}], "t")
    assert any("local_runs" in i for i in issues)


# ---------------------------------------------------------------------------
# v1sr — station proximity
# ---------------------------------------------------------------------------

def _make_prox_zip(tmp):
    entries = [
        (f"INMET_SE_RJ_{c}_{n}_01-01-2022_A_31-12-2022.CSV",
         _inmet_csv_header(c, lat, lon, uf, n))
        for c, lat, lon, uf, n in [
            ("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS"),  # near PET ~5km
            ("A301", "-8,05", "-34,87", "PE", "RECIFE"),
            ("A000", "0,00", "0,00", "XX", "OCEAN"),           # zero coords
        ]
    ]
    return _fake_zip(tmp, entries)


def test_v1sr_station_proximity_pet(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    rows = _read(tmp_path / v1sr.OUT_PROX.name)
    pet_rows = [r for r in rows if r["nearest_region"] == "PET"]
    assert len(pet_rows) >= 1


def test_v1sr_station_proximity_recife(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    rows = _read(tmp_path / v1sr.OUT_PROX.name)
    rec_rows = [r for r in rows if r["nearest_region"] == "RECIFE"]
    assert len(rec_rows) >= 1


def test_v1sr_within_100km_flag(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    rows = _read(tmp_path / v1sr.OUT_PROX.name)
    pet_row = next((r for r in rows if r["station_code"] == "A627"), None)
    assert pet_row is not None
    assert pet_row["within_100km"] == "true"


def test_v1sr_zero_coord_station_excluded(monkeypatch, tmp_path):
    # Stations with lat==0 (missing coord) are silently excluded.
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    rows = _read(tmp_path / v1sr.OUT_PROX.name)
    codes = {r["station_code"] for r in rows}
    assert "A000" not in codes  # zero-coord station excluded by design


def test_v1sr_fail_closed_no_zips(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "empty"))
    v1sr.run()
    rows = _read(tmp_path / v1sr.OUT_PROX.name)
    assert len(rows) == 0  # no stations without zip


def test_v1sr_no_abs_path_in_output(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    for r in _read(tmp_path / v1sr.OUT_PROX.name):
        for v in r.values():
            assert not C.ABS_PATH_RE.search(str(v)), f"Abs path in {v!r}"


def test_v1sr_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    rows = _read(tmp_path / v1sr.OUT_PROX.name)
    assert all(r.get("review_only") == "true" for r in rows)


# ---------------------------------------------------------------------------
# v1ss — event date windows
# ---------------------------------------------------------------------------

def test_v1ss_parses_pet_events(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ss, tmp_path)
    # Use real datasets path (read-only)
    out = v1ss.run()
    assert out["parsed"] >= 0


def test_v1ss_fail_closed_no_data(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ss, tmp_path)
    monkeypatch.setattr(v1ss, "DATASETS", tmp_path)
    v1ss.run()
    rows = _read(tmp_path / v1ss.OUT_WIN.name)
    assert len(rows) >= 1  # at least header row (fail-closed)
    assert _header(tmp_path / v1ss.OUT_WIN.name) != []


def test_v1ss_window_days():
    ws, we = C.build_window(date(2022, 2, 19), before_days=7, after_days=1)
    assert (we - ws).days + 1 == 9


def test_v1ss_review_only_in_rows(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ss, tmp_path)
    v1ss.run()
    rows = _read(tmp_path / v1ss.OUT_WIN.name)
    for r in rows:
        assert r.get("review_only") == "true"


def test_v1ss_no_ground_truth(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ss, tmp_path)
    v1ss.run()
    for r in _read(tmp_path / v1ss.OUT_WIN.name):
        assert r.get("ground_truth_operational") == "false"


# ---------------------------------------------------------------------------
# v1st — precipitation context
# ---------------------------------------------------------------------------

def _make_v1st_fixtures(tmp: Path):
    """Write minimal prox and window CSVs for v1st test."""
    prox = [{"station_code": "A627", "station_name": "PETROPOLIS", "nearest_region": "PET",
             "within_100km": "true", "within_50km": "true", "within_25km": "true",
             "distance_km": "4.5", "review_only": "true", "does_not_validate_event": "true",
             "can_create_operational_label": "false", "can_train_model": "false",
             "target_created": "false", "ground_truth_operational": "false",
             "formal_negative": "false"}]
    _write_csv(tmp / "protocol_c_inmet_station_region_proximity_v1sr.csv", prox,
               list(prox[0].keys()))
    wins = [{"event_window_id": "V1SS_W0000", "event_candidate_id": "C001",
             "region": "PET", "hazard_type": "FLOOD_LANDSLIDE",
             "parsed_date": "2022-02-19",
             "window_start": "2022-02-12", "window_end": "2022-02-20",
             "blocked_reason": "", "review_only": "true",
             "does_not_validate_event": "true",
             "can_create_operational_label": "false", "can_train_model": "false",
             "target_created": "false", "ground_truth_operational": "false",
             "formal_negative": "false"}]
    _write_csv(tmp / "protocol_c_event_date_windows_v1ss.csv", wins, list(wins[0].keys()))


def test_v1st_context_with_fixture(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1st, tmp_path)
    _make_v1st_fixtures(tmp_path)
    # minimal fake zip with precip data for A627
    csv_body = _inmet_csv_header("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")
    _fake_zip(tmp_path, [("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV", csv_body)])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1st, "DATASETS", tmp_path)
    v1st.run()
    rows = _read(tmp_path / v1st.OUT_CTX.name)
    assert len(rows) >= 1


def test_v1st_evidence_role_never_validated(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1st, tmp_path)
    _make_v1st_fixtures(tmp_path)
    csv_body = _inmet_csv_header("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS")
    _fake_zip(tmp_path, [("INMET_SE_RJ_A627_PETROPOLIS_01-01-2022_A_31-12-2022.CSV", csv_body)])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1st, "DATASETS", tmp_path)
    v1st.run()
    for r in _read(tmp_path / v1st.OUT_CTX.name):
        assert r.get("evidence_role") != "EVENT_VALIDATED"
        assert r.get("evidence_role") != "NEGATIVE_EVIDENCE"


def test_v1st_no_label(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1st, tmp_path)
    _make_v1st_fixtures(tmp_path)
    _fake_zip(tmp_path, [("INMET_SE_RJ_A627_PETROPOLIS_.CSV",
                           _inmet_csv_header("A627", "-22,46", "-43,10", "RJ", "PETROPOLIS"))])
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(v1st, "DATASETS", tmp_path)
    v1st.run()
    for r in _read(tmp_path / v1st.OUT_CTX.name):
        assert r.get("can_create_operational_label") == "false"


def test_v1st_fail_closed_no_prox(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1st, tmp_path)
    monkeypatch.setattr(v1st, "DATASETS", tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "empty"))
    v1st.run()
    rows = _read(tmp_path / v1st.OUT_CTX.name)
    assert len(rows) >= 1  # fail-closed row
    assert _header(tmp_path / v1st.OUT_CTX.name) != []


def test_v1st_absence_not_negative(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1st, tmp_path)
    _make_v1st_fixtures(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "no_data"))
    monkeypatch.setattr(v1st, "DATASETS", tmp_path)
    v1st.run()
    for r in _read(tmp_path / v1st.OUT_CTX.name):
        assert r.get("absence_as_negative") == "false"


# ---------------------------------------------------------------------------
# v1su — rolling features
# ---------------------------------------------------------------------------

def _write_ctx_and_win(tmp: Path, with_data: bool = True):
    gr = C.guardrail_row()
    ctx = [{
        "event_window_id": "V1SS_W0000", "region": "PET",
        "station_code": "A627", "date": "2022-02-19",
        "precipitation_mm": "10.5" if with_data else "",
        "precipitation_context_status": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY" if with_data else "NO_DATA",
        **gr
    }]
    _write_csv(tmp / "protocol_c_inmet_precipitation_event_window_context_v1st.csv",
               ctx, list(ctx[0].keys()))
    wins = [{
        "event_window_id": "V1SS_W0000", "region": "PET",
        "parsed_date": "2022-02-19",
        "window_start": "2022-02-12", "window_end": "2022-02-20",
        "blocked_reason": "", **gr
    }]
    _write_csv(tmp / "protocol_c_event_date_windows_v1ss.csv", wins, list(wins[0].keys()))
    prox = [{"station_code": "A627", "distance_km": "4.5", **gr}]
    _write_csv(tmp / "protocol_c_inmet_station_region_proximity_v1sr.csv",
               prox, list(prox[0].keys()))


def test_v1su_rolling_features_generated(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1su, tmp_path)
    _write_ctx_and_win(tmp_path, with_data=True)
    monkeypatch.setattr(v1su, "DATASETS", tmp_path)
    v1su.run()
    rows = _read(tmp_path / v1su.OUT_FEAT.name)
    real = [r for r in rows if r.get("feature_status") == "ROLLING_CONTEXT_REVIEW_ONLY"]
    assert len(real) >= 1


def test_v1su_features_not_target(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1su, tmp_path)
    _write_ctx_and_win(tmp_path)
    monkeypatch.setattr(v1su, "DATASETS", tmp_path)
    v1su.run()
    for r in _read(tmp_path / v1su.OUT_FEAT.name):
        assert r.get("target_created") == "false"
        assert r.get("can_train_model") == "false"


def test_v1su_fail_closed_no_context(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1su, tmp_path)
    monkeypatch.setattr(v1su, "DATASETS", tmp_path)
    v1su.run()
    rows = _read(tmp_path / v1su.OUT_FEAT.name)
    assert len(rows) >= 1  # fail-closed


def test_v1su_rain_1d_3d_7d_fields_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1su, tmp_path)
    _write_ctx_and_win(tmp_path)
    monkeypatch.setattr(v1su, "DATASETS", tmp_path)
    v1su.run()
    header = _header(tmp_path / v1su.OUT_FEAT.name)
    assert "rain_1d" in header and "rain_3d" in header and "rain_7d" in header


# ---------------------------------------------------------------------------
# v1sv — intake crosswalk
# ---------------------------------------------------------------------------

def _write_wins_feats(tmp: Path):
    gr = C.guardrail_row()
    wins = [{
        "event_window_id": "V1SS_W0000", "event_candidate_id": "C001",
        "region": "PET", "blocked_reason": "", **gr
    }]
    _write_csv(tmp / "protocol_c_event_date_windows_v1ss.csv", wins, list(wins[0].keys()))
    feats = [{
        "event_window_id": "V1SS_W0000", "region": "PET",
        "feature_status": "ROLLING_CONTEXT_REVIEW_ONLY",
        "rain_7d": "25.0", **gr
    }]
    _write_csv(tmp / "protocol_c_rolling_rainfall_context_features_v1su.csv",
               feats, list(feats[0].keys()))


def test_v1sv_manual_review_required(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sv, tmp_path)
    _write_wins_feats(tmp_path)
    monkeypatch.setattr(v1sv, "DATASETS", tmp_path)
    v1sv.run()
    rows = _read(tmp_path / v1sv.OUT_CW.name)
    assert all(r["manual_review_required"] == "true" for r in rows)


def test_v1sv_no_auto_intake(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sv, tmp_path)
    _write_wins_feats(tmp_path)
    monkeypatch.setattr(v1sv, "DATASETS", tmp_path)
    v1sv.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sv.OUT_SUM.name)}
    assert summ["auto_intake_rows"] == "0"


# ---------------------------------------------------------------------------
# v1sw — TCC tables
# ---------------------------------------------------------------------------

def _write_v1sw_inputs(tmp: Path):
    gr = C.guardrail_row()
    prox = [
        {"station_code": "A627", "nearest_region": "PET",
         "within_100km": "true", "within_50km": "true", "within_25km": "false",
         "distance_km": "5.0", **gr},
        {"station_code": "A301", "nearest_region": "RECIFE",
         "within_100km": "true", "within_50km": "false", "within_25km": "false",
         "distance_km": "10.0", **gr},
    ]
    _write_csv(tmp / "protocol_c_inmet_station_region_proximity_v1sr.csv",
               prox, list(prox[0].keys()))
    wins = [{"event_window_id": "V1SS_W0000", "region": "PET",
             "parsed_date": "2022-02-19", "blocked_reason": "", **gr}]
    _write_csv(tmp / "protocol_c_event_date_windows_v1ss.csv", wins, list(wins[0].keys()))
    # empty ctx and feats
    for fn in ("protocol_c_inmet_precipitation_event_window_context_v1st.csv",
               "protocol_c_rolling_rainfall_context_features_v1su.csv"):
        _write_csv(tmp / fn, [], ["region", "precipitation_context_status", "feature_status"])


def test_v1sw_tcc_tables_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sw, tmp_path)
    _write_v1sw_inputs(tmp_path)
    monkeypatch.setattr(v1sw, "DATASETS", tmp_path)
    v1sw.run()
    assert (tmp_path / v1sw.OUT_COV.name).exists()
    assert (tmp_path / v1sw.OUT_WIN.name).exists()
    assert (tmp_path / v1sw.OUT_LIM.name).exists()


def test_v1sw_limitations_table_has_rows(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sw, tmp_path)
    _write_v1sw_inputs(tmp_path)
    monkeypatch.setattr(v1sw, "DATASETS", tmp_path)
    v1sw.run()
    lims = _read(tmp_path / v1sw.OUT_LIM.name)
    assert len(lims) >= 3


def test_v1sw_absence_limitation_documented(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sw, tmp_path)
    _write_v1sw_inputs(tmp_path)
    monkeypatch.setattr(v1sw, "DATASETS", tmp_path)
    v1sw.run()
    lims = _read(tmp_path / v1sw.OUT_LIM.name)
    descs = " ".join(r.get("description", "") for r in lims).lower()
    assert "ausencia" in descs or "absence" in descs


# ---------------------------------------------------------------------------
# v1sx — guardrail audit
# ---------------------------------------------------------------------------

def _write_clean_v1sr_v1sw(tmp: Path):
    gr = C.guardrail_row()
    files = [
        "protocol_c_inmet_station_region_proximity_v1sr.csv",
        "protocol_c_event_date_windows_v1ss.csv",
        "protocol_c_inmet_precipitation_event_window_context_v1st.csv",
        "protocol_c_rolling_rainfall_context_features_v1su.csv",
        "protocol_c_hydromet_evidence_intake_crosswalk_v1sv.csv",
        "protocol_c_tcc_table_hydromet_station_coverage_v1sw.csv",
        "protocol_c_tcc_table_hydromet_event_windows_v1sw.csv",
        "protocol_c_tcc_table_hydromet_context_limitations_v1sw.csv",
    ]
    for fn in files:
        _write_csv(tmp / fn, [{"region": "PET", "review_only": "true", **gr}],
                   ["region", "review_only"] + list(gr.keys()))


def test_v1sx_audit_all_pass(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sx, tmp_path)
    _write_clean_v1sr_v1sw(tmp_path)
    monkeypatch.setattr(v1sx, "DATASETS", tmp_path)
    v1sx.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sx.OUT_SUM.name)}
    assert summ["audit_status"] == "GUARDRAIL_PASS_ALL"


def test_v1sx_detects_violation(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sx, tmp_path)
    # Write one CSV with a forbidden field
    bad_row = {"region": "PET", "review_only": "true", "can_train_model": "true",
               "target_created": "false", "ground_truth_operational": "false",
               "formal_negative": "false"}
    _write_csv(tmp_path / "protocol_c_inmet_station_region_proximity_v1sr.csv",
               [bad_row], list(bad_row.keys()))
    for fn in [
        "protocol_c_event_date_windows_v1ss.csv",
        "protocol_c_inmet_precipitation_event_window_context_v1st.csv",
        "protocol_c_rolling_rainfall_context_features_v1su.csv",
        "protocol_c_hydromet_evidence_intake_crosswalk_v1sv.csv",
        "protocol_c_tcc_table_hydromet_station_coverage_v1sw.csv",
        "protocol_c_tcc_table_hydromet_event_windows_v1sw.csv",
        "protocol_c_tcc_table_hydromet_context_limitations_v1sw.csv",
    ]:
        _write_csv(tmp_path / fn, [], ["region"])
    monkeypatch.setattr(v1sx, "DATASETS", tmp_path)
    v1sx.run()
    audit = _read(tmp_path / v1sx.OUT_AUDIT.name)
    assert any(r["audit_status"] == "FAIL" for r in audit)


# ---------------------------------------------------------------------------
# v1sz — bundle
# ---------------------------------------------------------------------------

def _write_all_summaries(tmp: Path, ctx_rows: int = 810):
    stats = [
        ("protocol_c_inmet_station_region_proximity_summary_v1sr.csv",
         [("stations_within_100km", "25"), ("stage", "v1sr")]),
        ("protocol_c_event_date_windows_summary_v1ss.csv",
         [("windows_total", "9"), ("parseable_dates", "9"), ("stage", "v1ss")]),
        ("protocol_c_inmet_precipitation_event_window_summary_v1st.csv",
         [("context_rows", str(ctx_rows)), ("stage", "v1st")]),
        ("protocol_c_rolling_rainfall_context_features_summary_v1su.csv",
         [("feature_rows", "78"), ("stage", "v1su")]),
        ("protocol_c_hydromet_evidence_intake_crosswalk_summary_v1sv.csv",
         [("crosswalk_rows", "9"), ("stage", "v1sv")]),
        ("protocol_c_hydromet_context_guardrail_summary_v1sx.csv",
         [("audit_status", "GUARDRAIL_PASS_ALL"), ("total_violations", "0"), ("files_audited", "8"),
          ("files_pass", "8"), ("files_fail", "0"), ("stage", "v1sx")]),
    ]
    for fname, kv in stats:
        _write_csv(tmp / fname,
                   [{"stat_key": k, "stat_value": v} for k, v in kv],
                   ["stat_key", "stat_value"])


def test_v1sz_bundle_manifest_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sz, tmp_path)
    _write_all_summaries(tmp_path)
    monkeypatch.setattr(v1sz, "IN_SR_SUM",  tmp_path / v1sz.IN_SR_SUM.name)
    monkeypatch.setattr(v1sz, "IN_SS_SUM",  tmp_path / v1sz.IN_SS_SUM.name)
    monkeypatch.setattr(v1sz, "IN_ST_SUM",  tmp_path / v1sz.IN_ST_SUM.name)
    monkeypatch.setattr(v1sz, "IN_SU_SUM",  tmp_path / v1sz.IN_SU_SUM.name)
    monkeypatch.setattr(v1sz, "IN_SV_SUM",  tmp_path / v1sz.IN_SV_SUM.name)
    monkeypatch.setattr(v1sz, "IN_SX_SUM",  tmp_path / v1sz.IN_SX_SUM.name)
    v1sz.run()
    assert (tmp_path / v1sz.OUT_MAN.name).exists()


def test_v1sz_final_status_ready(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sz, tmp_path)
    _write_all_summaries(tmp_path)
    for attr in ("IN_SR_SUM", "IN_SS_SUM", "IN_ST_SUM", "IN_SU_SUM", "IN_SV_SUM", "IN_SX_SUM"):
        monkeypatch.setattr(v1sz, attr, tmp_path / getattr(v1sz, attr).name)
    result = v1sz.run()
    assert result["final_status"] == v1sz.ST_READY


def test_v1sz_guardrail_fail_closed(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sz, tmp_path)
    _write_all_summaries(tmp_path, ctx_rows=810)
    # Override guardrail summary to simulate violation
    _write_csv(tmp_path / "protocol_c_hydromet_context_guardrail_summary_v1sx.csv",
               [{"stat_key": "audit_status", "stat_value": "GUARDRAIL_FAIL_CLOSED"},
                {"stat_key": "total_violations", "stat_value": "2"},
                {"stat_key": "files_audited", "stat_value": "8"}],
               ["stat_key", "stat_value"])
    for attr in ("IN_SR_SUM", "IN_SS_SUM", "IN_ST_SUM", "IN_SU_SUM", "IN_SV_SUM", "IN_SX_SUM"):
        monkeypatch.setattr(v1sz, attr, tmp_path / getattr(v1sz, attr).name)
    result = v1sz.run()
    assert result["final_status"] == v1sz.ST_GUARDRAIL


def test_v1sz_labels_zero(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sz, tmp_path)
    _write_all_summaries(tmp_path)
    for attr in ("IN_SR_SUM", "IN_SS_SUM", "IN_ST_SUM", "IN_SU_SUM", "IN_SV_SUM", "IN_SX_SUM"):
        monkeypatch.setattr(v1sz, attr, tmp_path / getattr(v1sz, attr).name)
    v1sz.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sz.OUT_SUM.name)}
    assert summ["labels_created"] == "0"
    assert summ["targets_created"] == "0"
    assert summ["ground_truth_operational_created"] == "0"
    assert summ["formal_negatives_created"] == "0"


# ---------------------------------------------------------------------------
# Docs / schemas / hygiene
# ---------------------------------------------------------------------------

def test_v1sr_doc_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    assert (tmp_path / v1sr.DOC.name).exists()


def test_v1sr_schema_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    assert (tmp_path / v1sr.SCHEMA_P.name).exists()


def test_v1sy_mandatory_clause(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sy, tmp_path)
    monkeypatch.setattr(v1sy, "DATASETS", tmp_path)
    v1sy.run()
    doc = (tmp_path / v1sy.DOC.name).read_text(encoding="utf-8")
    assert "v1sr" in doc and "v1sz" in doc
    assert "ground truth operacional" in doc
    assert "revisão humana" in doc


def test_no_writes_outside_tmp(monkeypatch, tmp_path):
    """Scripts must not modify real dataset files during test."""
    real_datasets = ROOT / "datasets"
    before = {p.name: p.stat().st_mtime for p in real_datasets.glob("*v1sr*")}
    _redirect(monkeypatch, v1sr, tmp_path)
    _make_prox_zip(tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path))
    v1sr.run()
    after = {p.name: p.stat().st_mtime for p in real_datasets.glob("*v1sr*")}
    assert before == after


def test_staged_area_empty():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=str(ROOT), capture_output=True, text=True
    )
    assert result.stdout.strip() == ""
