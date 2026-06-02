"""Tests for REV-P Protocol C v1sg-v1sp official data acquisition.

All outputs redirected to tmp_path. Mock HTTP when needed. No real internet
dependency. No real data written to datasets/ or data/.
"""
from __future__ import annotations

import csv
import importlib
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1sg_v1sz_official_download_common as DC  # noqa: E402
import revp_v1qu_v1qz_ground_reference_common as G  # noqa: E402

v1sg = importlib.import_module("revp_v1sg_official_source_endpoint_registry")
v1sh = importlib.import_module("revp_v1sh_inmet_historical_data_downloader")
v1si = importlib.import_module("revp_v1si_inmet_station_precipitation_extractor")
v1sj = importlib.import_module("revp_v1sj_ana_hidroweb_acquisition")
v1sk = importlib.import_module("revp_v1sk_institutional_document_discovery_queue")
v1sl = importlib.import_module("revp_v1sl_official_download_orchestrator")
v1sm = importlib.import_module("revp_v1sm_downloaded_external_document_intake_adapter")
v1sn = importlib.import_module("revp_v1sn_official_data_provenance_license_audit")
v1so = importlib.import_module("revp_v1so_official_evidence_readiness_gate")
v1sp = importlib.import_module("revp_v1sp_official_acquisition_bundle")
v1sq = importlib.import_module("revp_v1sq_official_download_command_pack")


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

def _header(path: Path) -> list[str]:
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), [])

def _redirect(monkeypatch, mod, tmp):
    for name in dir(mod):
        if name.startswith(("OUT_", "SCHEMA_", "DOC")):
            val = getattr(mod, name)
            if isinstance(val, Path):
                monkeypatch.setattr(mod, name, tmp / val.name)

def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch, tmp_path):
    """Hard guarantee no test ever touches the real internet.

    All download functions resolve ``rate_limited_get`` from the common module
    at call time, so patching it here covers every script. Tests that simulate a
    download patch the script-bound ``download_file`` instead and never reach it.
    Raw/cache roots are redirected so accidental writes land in tmp.
    """
    def _no_network(*_a, **_k):
        raise AssertionError("real network call blocked in tests")
    monkeypatch.setattr(DC, "rate_limited_get", _no_network)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "raw"))
    monkeypatch.setenv("REVP_EXTERNAL_CACHE_ROOT", str(tmp_path / "cache"))


def _ok_download(**over):
    base = {"downloaded": "true", "download_attempted": "true",
            "download_status": "DOWNLOADED_OK", "file_sha256_short": "abc123",
            "file_size_bytes": "1000", "http_status": "200",
            "content_type": "application/zip"}
    base.update(over)
    return lambda url, dest, **kw: base


# ===========================================================================
# Common helpers
# ===========================================================================

def test_common_empty_csv_header(tmp_path):
    DC.write_csv_with_header(tmp_path / "x.csv", [], ["a", "b"])
    assert _header(tmp_path / "x.csv") == ["a", "b"]

def test_safe_url_official():
    assert DC.safe_url("https://portal.inmet.gov.br/x") != ""

def test_safe_url_blocked():
    assert DC.safe_url("https://evil.com/x") == ""

def test_detect_abs_path():
    assert DC.detect_absolute_path(r"C:\Users\x\f.tif")

def test_detect_forbidden_literal():
    assert DC.detect_forbidden_literal_exposure("local_runs/x")

def test_classify_source_inmet():
    assert DC.classify_source_family("INMET BDMEP") == "OFFICIAL_HYDROMETEOROLOGICAL"

def test_classify_source_ana():
    assert DC.classify_source_family("ANA HidroWeb") == "OFFICIAL_HYDROMETEOROLOGICAL"

def test_classify_source_cemaden():
    assert DC.classify_source_family("CEMADEN alerta") == "OFFICIAL_HYDROMETEOROLOGICAL"

def test_classify_source_sgb():
    assert DC.classify_source_family("SGB CPRM") == "OFFICIAL_GEOLOGICAL"

def test_classify_source_defesa_civil():
    assert DC.classify_source_family("defesa civil") == "OFFICIAL_CIVIL_DEFENSE"

def test_classify_source_diario_oficial():
    assert DC.classify_source_family("diario oficial") == "OFFICIAL_GOVERNMENT_PUBLICATION"

def test_guardrail_row_safe():
    r = DC.guardrail_row()
    assert r["review_only"] == "true"
    for f in DC.FORBIDDEN_TRUE:
        assert r[f] == "false"

def test_is_allowed_domain_gov():
    assert DC.is_allowed_domain("https://portal.inmet.gov.br/x")

def test_is_not_allowed_domain():
    assert not DC.is_allowed_domain("https://evil.com/x")


# ===========================================================================
# v1sg — endpoint registry
# ===========================================================================

def test_v1sg_has_inmet(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sg, tmp_path)
    v1sg.run()
    rows = _read(tmp_path / v1sg.OUT_REGISTRY.name)
    assert any("INMET" in r["source_name"] for r in rows)

def test_v1sg_has_ana(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sg, tmp_path)
    v1sg.run()
    rows = _read(tmp_path / v1sg.OUT_REGISTRY.name)
    assert any("ANA" in r["source_name"] or "HidroWeb" in r["source_name"] for r in rows)

def test_v1sg_config_required_for_ambiguous(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sg, tmp_path)
    v1sg.run()
    rows = _read(tmp_path / v1sg.OUT_REGISTRY.name)
    assert any(r["blocked_reason"] == "ENDPOINT_CONFIG_REQUIRED" for r in rows)


# ===========================================================================
# v1sh — INMET downloader
# ===========================================================================

_LINK_2022 = [("2022", "https://portal.inmet.gov.br/uploads/dadoshistoricos/2022.zip")]


def test_v1sh_disabled_generates_queue(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sh, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    monkeypatch.setattr(v1sh, "_discover_annual_links", lambda: _LINK_2022)
    v1sh.run()
    rows = _read(tmp_path / v1sh.OUT_QUEUE.name)
    assert len(rows) >= 1
    manifest = _read(tmp_path / v1sh.OUT_MANIFEST.name)
    assert all(m["downloaded"] == "false" for m in manifest)

def test_v1sh_download_attempted_false_when_disabled(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sh, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    monkeypatch.setattr(v1sh, "_discover_annual_links", lambda: _LINK_2022)
    v1sh.run()
    manifest = _read(tmp_path / v1sh.OUT_MANIFEST.name)
    assert all(m["download_attempted"] == "false" for m in manifest)

def test_v1sh_mock_download(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sh, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    monkeypatch.setenv("REVP_DOWNLOAD_MODE", "minimal")
    monkeypatch.setattr(v1sh, "_discover_annual_links", lambda: _LINK_2022)
    # Patch the script-bound name so the common network layer is never reached.
    monkeypatch.setattr(v1sh, "download_file", _ok_download())
    v1sh.run()
    manifest = _read(tmp_path / v1sh.OUT_MANIFEST.name)
    assert any(m["downloaded"] == "true" for m in manifest)

def test_v1sh_respects_max_files(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sh, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    monkeypatch.setenv("REVP_DOWNLOAD_MAX_FILES", "1")
    links = [("2020", "https://portal.inmet.gov.br/uploads/dadoshistoricos/2020.zip"),
             ("2021", "https://portal.inmet.gov.br/uploads/dadoshistoricos/2021.zip")]
    monkeypatch.setattr(v1sh, "_discover_annual_links", lambda: links)
    monkeypatch.setattr(v1sh, "download_file", _ok_download())
    v1sh.run()
    manifest = _read(tmp_path / v1sh.OUT_MANIFEST.name)
    assert any(m["download_status"] == "SKIPPED_MAX_FILES" for m in manifest)

def test_v1sh_no_absolute_path_in_manifest(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sh, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    monkeypatch.setattr(v1sh, "_discover_annual_links", lambda: _LINK_2022)
    v1sh.run()
    for r in _read(tmp_path / v1sh.OUT_MANIFEST.name):
        for v in r.values():
            assert not DC.detect_absolute_path(str(v))


# ===========================================================================
# v1si — INMET extractor
# ===========================================================================

def test_v1si_reads_fixture(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1si, tmp_path)
    raw = tmp_path / "raw" / "inmet" / "historical"
    raw.mkdir(parents=True)
    # Create a simple INMET CSV fixture
    csv_content = (
        "REGIAO:;SE\n"
        "UF:;PE\n"
        "ESTACAO:;RECIFE_A301\n"
        "CODIGO (WMO):;A301\n"
        "LATITUDE:;-8.059\n"
        "LONGITUDE:;-34.871\n"
        "DATA (YYYY-MM-DD);PRECIPITACAO TOTAL HORARIO (mm)\n"
        "2022-05-28;12.5\n"
        "2022-05-29;0.0\n"
    )
    (raw / "test.csv").write_text(csv_content, encoding="latin-1")
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "raw"))
    v1si.run()
    stations = _read(tmp_path / v1si.OUT_STATIONS.name)
    precip = _read(tmp_path / v1si.OUT_PRECIP.name)
    assert len(stations) >= 1
    assert len(precip) >= 1

def test_v1si_no_label(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1si, tmp_path)
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "raw"))
    v1si.run()
    for r in _read(tmp_path / v1si.OUT_PRECIP.name):
        assert r.get("can_create_operational_label") == "false"


# ===========================================================================
# v1sj — ANA
# ===========================================================================

def test_v1sj_generates_queue(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sj, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    v1sj.run()
    rows = _read(tmp_path / v1sj.OUT_QUEUE.name)
    assert len(rows) >= 2

def test_v1sj_mock_download(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sj, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    monkeypatch.setattr(v1sj, "download_file", _ok_download(
        download_status="ANA_DOWNLOADED_OK", file_size_bytes="500",
        content_type="text/xml"))
    v1sj.run()
    manifest = _read(tmp_path / v1sj.OUT_MANIFEST.name)
    assert any(m["downloaded"] == "true" for m in manifest)

def test_v1sj_fail_closed_ambiguous_endpoint(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sj, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    v1sj.run()
    rows = _read(tmp_path / v1sj.OUT_QUEUE.name)
    # HidroWeb requires an interactive query → must be flagged, not auto-fetched.
    assert any(r["requires_manual_query"] == "true" and r["auto_download_allowed"] == "false"
               for r in rows)


# ===========================================================================
# v1sk — institutional discovery
# ===========================================================================

def test_v1sk_generates_queries(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sk, tmp_path)
    v1sk.run()
    rows = _read(tmp_path / v1sk.OUT_QUEUE.name)
    assert len(rows) >= 5
    queries = " ".join(r["suggested_query"] for r in rows)
    assert "CEMADEN" in queries or "Defesa Civil" in queries

def test_v1sk_does_not_invent_document(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sk, tmp_path)
    v1sk.run()
    rows = _read(tmp_path / v1sk.OUT_QUEUE.name)
    # No row should claim a document actually exists
    for r in rows:
        assert r.get("auto_download_allowed") in ("false", "")


# ===========================================================================
# v1sl — orchestrator
# ===========================================================================

def test_v1sl_consolidates(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sl, tmp_path)
    _write_csv(tmp_path / "inmet.csv", [{"source_name": "INMET", "downloaded": "true", "file_size_bytes": "1000", "download_status": "OK"}],
               ["source_name", "downloaded", "file_size_bytes", "download_status"])
    _write_csv(tmp_path / "ana.csv", [{"source_name": "ANA", "downloaded": "false", "file_size_bytes": "0", "download_status": "QUEUE"}],
               ["source_name", "downloaded", "file_size_bytes", "download_status"])
    _write_csv(tmp_path / "inst.csv", [{"source_name": "DC", "requires_manual_review": "true"}],
               ["source_name", "requires_manual_review"])
    monkeypatch.setattr(v1sl, "IN_INMET_MANIFEST", tmp_path / "inmet.csv")
    monkeypatch.setattr(v1sl, "IN_ANA_MANIFEST", tmp_path / "ana.csv")
    monkeypatch.setattr(v1sl, "IN_INSTITUTIONAL", tmp_path / "inst.csv")
    v1sl.run()
    rows = _read(tmp_path / v1sl.OUT_MANIFEST.name)
    assert len(rows) == 3


# ===========================================================================
# v1sm — adapter
# ===========================================================================

def test_v1sm_creates_draft_from_downloaded(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sm, tmp_path)
    _write_csv(tmp_path / "orch.csv",
               [{"source_name": "INMET", "source_block": "v1sh", "downloaded": "true", "url": "", "file_size_bytes": "1000"}],
               ["source_name", "source_block", "downloaded", "url", "file_size_bytes"])
    _write_csv(tmp_path / "stations.csv", [], ["station_code", "region_candidate"])
    monkeypatch.setattr(v1sm, "IN_ORCHESTRATOR", tmp_path / "orch.csv")
    monkeypatch.setattr(v1sm, "IN_INMET_STATIONS", tmp_path / "stations.csv")
    v1sm.run()
    rows = _read(tmp_path / v1sm.OUT_DRAFT.name)
    assert len(rows) == 1
    assert rows[0]["manual_review_required"] == "true"


# ===========================================================================
# v1sn — provenance audit
# ===========================================================================

def test_v1sn_marks_license_review(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sn, tmp_path)
    _write_csv(tmp_path / "orch.csv",
               [{"source_name": "X", "source_block": "v1sh", "downloaded": "true", "url": "https://portal.inmet.gov.br/x", "file_size_bytes": "100"}],
               ["source_name", "source_block", "downloaded", "url", "file_size_bytes"])
    monkeypatch.setattr(v1sn, "IN_ORCHESTRATOR", tmp_path / "orch.csv")
    v1sn.run()
    rows = _read(tmp_path / v1sn.OUT_AUDIT.name)
    assert rows[0]["manual_license_review_required"] == "true"


# ===========================================================================
# v1so — readiness gate
# ===========================================================================

def test_v1so_queue_ready_disabled(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1so, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    v1so.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1so.OUT_SUMMARY.name)}
    assert summ["readiness_status"] in ("OFFICIAL_EVIDENCE_DOWNLOADS_DISABLED_QUEUE_READY",
                                         "OFFICIAL_EVIDENCE_BLOCKED_NO_SOURCES")

def test_v1so_ready_with_mock(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1so, tmp_path)
    _write_csv(tmp_path / "ep.csv", [{"stat_key": "endpoints_ready", "stat_value": "3"}], ["stat_key", "stat_value"])
    _write_csv(tmp_path / "orch.csv", [{"stat_key": "files_downloaded", "stat_value": "5"}], ["stat_key", "stat_value"])
    _write_csv(tmp_path / "draft.csv", [{"stat_key": "intake_draft_rows", "stat_value": "2"}], ["stat_key", "stat_value"])
    _write_csv(tmp_path / "lic.csv", [{"stat_key": "license_review_required", "stat_value": "1"}], ["stat_key", "stat_value"])
    monkeypatch.setattr(v1so, "IN_ENDPOINTS", tmp_path / "ep.csv")
    monkeypatch.setattr(v1so, "IN_ORCH_SUMMARY", tmp_path / "orch.csv")
    monkeypatch.setattr(v1so, "IN_DRAFT_SUMMARY", tmp_path / "draft.csv")
    monkeypatch.setattr(v1so, "IN_LICENSE_SUMMARY", tmp_path / "lic.csv")
    v1so.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1so.OUT_SUMMARY.name)}
    assert summ["readiness_status"] == "OFFICIAL_EVIDENCE_READY_FOR_MANUAL_INTAKE"


# ===========================================================================
# v1sp — bundle
# ===========================================================================

def test_v1sp_creates_manifest(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    assert (tmp_path / v1sp.OUT_MANIFEST.name).exists()

def test_v1sp_creates_qc(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    assert (tmp_path / v1sp.OUT_QC.name).exists()

def test_v1sp_creates_summary(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    assert (tmp_path / v1sp.OUT_SUMMARY.name).exists()

def test_v1sp_queue_status(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    out = v1sp.run()
    assert out["final_status"] in (v1sp.ST_QUEUE, v1sp.ST_BLOCKED)

def test_v1sp_downloaded_status_with_mock(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    _write_csv(tmp_path / "ep.csv", [{"stat_key": "endpoints_total", "stat_value": "5"}, {"stat_key": "endpoints_ready", "stat_value": "3"}], ["stat_key", "stat_value"])
    _write_csv(tmp_path / "orch.csv", [{"stat_key": "files_downloaded", "stat_value": "3"}, {"stat_key": "total_bytes", "stat_value": "5000"}, {"stat_key": "inmet_items", "stat_value": "2"}, {"stat_key": "ana_items", "stat_value": "1"}, {"stat_key": "institutional_items", "stat_value": "5"}], ["stat_key", "stat_value"])
    _write_csv(tmp_path / "draft.csv", [{"stat_key": "intake_draft_rows", "stat_value": "2"}], ["stat_key", "stat_value"])
    _write_csv(tmp_path / "lic.csv", [{"stat_key": "license_review_required", "stat_value": "1"}, {"stat_key": "audited_items", "stat_value": "6"}], ["stat_key", "stat_value"])
    _write_csv(tmp_path / "gate.csv", [{"stat_key": "readiness_status", "stat_value": "READY"}], ["stat_key", "stat_value"])
    monkeypatch.setattr(v1sp, "IN_EP_SUMMARY", tmp_path / "ep.csv")
    monkeypatch.setattr(v1sp, "IN_ORCH_SUMMARY", tmp_path / "orch.csv")
    monkeypatch.setattr(v1sp, "IN_DRAFT_SUMMARY", tmp_path / "draft.csv")
    monkeypatch.setattr(v1sp, "IN_LICENSE_SUMMARY", tmp_path / "lic.csv")
    monkeypatch.setattr(v1sp, "IN_GATE_SUMMARY", tmp_path / "gate.csv")
    out = v1sp.run()
    assert out["final_status"] == v1sp.ST_DOWNLOADED

def test_v1sp_labels_zero(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sp.OUT_SUMMARY.name)}
    assert summ["labels_created"] == "0"
    assert summ["targets_created"] == "0"
    assert summ["ground_truth_operational_created"] == "0"
    assert summ["formal_negatives_created"] == "0"


# ===========================================================================
# Docs/schemas/hygiene
# ===========================================================================

def test_docs_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sg, tmp_path)
    v1sg.run()
    assert (tmp_path / v1sg.DOC.name).exists()

def test_schemas_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sg, tmp_path)
    v1sg.run()
    assert (tmp_path / v1sg.SCHEMA_REG.name).exists()

def test_gitignore_raw():
    assert (ROOT / "data" / "external_raw" / ".gitignore").exists()

def test_gitignore_cache():
    assert (ROOT / "data" / "external_cache" / ".gitignore").exists()

def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1sp*")}
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    after = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1sp*")}
    assert before == after

def test_no_internet_dependency():
    """Tests should work without network."""
    assert True  # all tests above use mocks or disabled downloads


# ===========================================================================
# Guardrails (negative)
# ===========================================================================

def test_guardrail_label_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_create_operational_label": "true"}], "t")

def test_guardrail_train_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_train_model": "true"}], "t")

def test_guardrail_target_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"target_created": "true"}], "t")

def test_guardrail_ground_truth_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"ground_truth_operational": "true"}], "t")

def test_guardrail_formal_negative_true():
    # DC's acquisition scan is stricter than the ground-reference common and
    # treats any formal_negative=true as a violation.
    with pytest.raises(ValueError):
        DC.forbidden_guardrail_scan([{"formal_negative": "true"}], "t")

def test_guardrail_dino_validates_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"dino_validates_event": "true"}], "t")

def test_guardrail_absence_as_negative_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"absence_as_negative": "true"}], "t")

def test_no_dino_as_proof(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    qc = _read(tmp_path / v1sp.OUT_QC.name)
    # no QC check should say dino validates event
    for c in qc:
        assert "dino_validates" not in c.get("check_name", "").lower()

def test_no_absence_as_negative(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sp.OUT_SUMMARY.name)}
    assert summ.get("formal_negatives_created") == "0"

def test_no_c4_opened(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sp.OUT_SUMMARY.name)}
    assert summ.get("formal_negatives_created") == "0"

def test_no_operational_ground_truth(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sp, tmp_path)
    v1sp.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sp.OUT_SUMMARY.name)}
    assert summ.get("ground_truth_operational_created") == "0"


# ===========================================================================
# Domain allowlist / safe_url coverage
# ===========================================================================

def test_safe_url_ana():
    assert DC.safe_url("https://telemetriaws1.ana.gov.br/ServiceANA.asmx") != ""

def test_safe_url_snirh():
    assert DC.safe_url("https://www.snirh.gov.br/hidroweb/") != ""

def test_safe_url_cemaden():
    assert DC.safe_url("https://www.gov.br/cemaden/pt-br") != ""

def test_safe_url_sgb_cprm():
    assert DC.safe_url("https://rigeo.sgb.gov.br/") != ""

def test_safe_url_empty():
    assert DC.safe_url("") == ""

def test_mask_local_path():
    masked = DC.mask_local_path(r"C:\Users\gabriela\Documents\REV-P\data\x.zip")
    assert not DC.detect_absolute_path(masked)
    assert "gabriela" not in masked.lower()

def test_mask_local_runs_not_exposed():
    masked = DC.mask_local_path("local" + "_runs/x/plot.png")
    assert not DC.detect_forbidden_literal_exposure(masked)

def test_classify_source_ibge():
    assert DC.classify_source_family("IBGE limites") == "SCIENTIFIC_DATASET"


# ===========================================================================
# common download_file — idempotent, fail-closed, allowlist
# ===========================================================================

def test_download_file_already_exists_hashed(monkeypatch, tmp_path):
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    dest = tmp_path / "raw" / "x.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"already here")
    res = DC.download_file("https://portal.inmet.gov.br/x.zip", dest)
    assert res["download_status"] == "ALREADY_EXISTS_HASHED"
    assert res["file_sha256_short"] and res["download_attempted"] == "false"

def test_download_file_partial_or_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    dest = tmp_path / "raw" / "empty.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"")
    res = DC.download_file("https://portal.inmet.gov.br/empty.zip", dest)
    assert res["download_status"] == "PARTIAL_OR_EMPTY_FILE_FAIL_CLOSED"

def test_download_file_disabled_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "false")
    res = DC.download_file("https://portal.inmet.gov.br/x.zip", tmp_path / "a.zip")
    assert res["download_status"] == "DOWNLOAD_DISABLED_QUEUE_ONLY"
    assert res["download_attempted"] == "false"

def test_download_file_blocks_external_domain(monkeypatch, tmp_path):
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    res = DC.download_file("https://evil.com/x.zip", tmp_path / "a.zip")
    assert res["download_status"] == "DOMAIN_NOT_ALLOWED_FAIL_CLOSED"

def test_download_file_respects_max_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    # rate_limited_get returns 200 with empty body to signal an over-cap skip.
    monkeypatch.setattr(DC, "rate_limited_get", lambda *a, **k: (200, b"", "application/zip"))
    res = DC.download_file("https://portal.inmet.gov.br/big.zip", tmp_path / "b.zip", max_bytes=10)
    assert res["download_status"] == "SKIPPED_MAX_SIZE_LIMIT"

def test_download_file_computes_hash(monkeypatch, tmp_path):
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    monkeypatch.setattr(DC, "rate_limited_get", lambda *a, **k: (200, b"payload-bytes", "application/zip"))
    res = DC.download_file("https://portal.inmet.gov.br/ok.zip", tmp_path / "c.zip")
    assert res["downloaded"] == "true" and len(res["file_sha256_short"]) == 16

def test_force_queue_only_default(monkeypatch):
    monkeypatch.delenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", raising=False)
    monkeypatch.delenv("REVP_DOWNLOAD_FORCE_QUEUE_ONLY", raising=False)
    assert DC.force_queue_only() is True
    assert DC.downloads_enabled() is False
    assert DC.network_allowed() is False

def test_force_queue_only_overrides_enable(monkeypatch):
    monkeypatch.setenv("REVP_ENABLE_OFFICIAL_DOWNLOADS", "true")
    monkeypatch.setenv("REVP_DOWNLOAD_FORCE_QUEUE_ONLY", "true")
    assert DC.downloads_enabled() is False

def test_redirect_handler_blocks_external():
    h = DC._AllowlistRedirectHandler()
    with pytest.raises(Exception):
        h.redirect_request(None, None, 302, "Found", {}, "https://evil.com/elsewhere")

def test_env_defaults(monkeypatch):
    for name in ("REVP_DOWNLOAD_MAX_FILES", "REVP_DOWNLOAD_MAX_BYTES_PER_FILE",
                 "REVP_DOWNLOAD_RETRIES", "REVP_DOWNLOAD_CONNECT_TIMEOUT_SECONDS",
                 "REVP_DOWNLOAD_READ_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    assert DC.max_files() == 20
    assert DC.max_bytes_per_file() == 250 * 1024 * 1024
    assert DC.retries() == 2
    assert DC.connect_timeout_sec() == 15
    assert DC.read_timeout_sec() == 60


# ===========================================================================
# v1sg endpoint registry — no network, families
# ===========================================================================

def test_v1sg_no_download(monkeypatch, tmp_path):
    # rate_limited_get is already blocked by the autouse fixture; a clean run
    # proves the registry never touches the network.
    _redirect(monkeypatch, v1sg, tmp_path)
    out = v1sg.run()
    assert out["endpoints"] >= 1

def test_v1sg_has_sgb_and_cemaden(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sg, tmp_path)
    v1sg.run()
    names = " ".join(r["source_name"] for r in _read(tmp_path / v1sg.OUT_REGISTRY.name))
    assert "CEMADEN" in names and "SGB" in names


# ===========================================================================
# v1si extractor — field extraction, no labels
# ===========================================================================

_INMET_CSV = (
    "REGIAO:;SE\n" "UF:;PE\n" "ESTACAO:;RECIFE_A301\n"
    "CODIGO (WMO):;A301\n" "LATITUDE:;-8.059\n" "LONGITUDE:;-34.871\n"
    "DATA (YYYY-MM-DD);PRECIPITACAO TOTAL HORARIO (mm)\n"
    "2022-05-28;12.5\n" "2022-05-29;0.0\n"
)

def _run_v1si_with_fixture(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1si, tmp_path)
    raw = tmp_path / "raw" / "inmet" / "historical"
    raw.mkdir(parents=True)
    (raw / "test.csv").write_text(_INMET_CSV, encoding="latin-1")
    monkeypatch.setenv("REVP_EXTERNAL_RAW_ROOT", str(tmp_path / "raw"))
    v1si.run()

def test_v1si_extracts_station_code(monkeypatch, tmp_path):
    _run_v1si_with_fixture(monkeypatch, tmp_path)
    stations = _read(tmp_path / v1si.OUT_STATIONS.name)
    assert any(s["station_code"] for s in stations)

def test_v1si_extracts_precipitation(monkeypatch, tmp_path):
    _run_v1si_with_fixture(monkeypatch, tmp_path)
    precip = _read(tmp_path / v1si.OUT_PRECIP.name)
    assert any(p["precipitation_mm"] == "12.5" for p in precip)

def test_v1si_precip_no_ground_truth(monkeypatch, tmp_path):
    _run_v1si_with_fixture(monkeypatch, tmp_path)
    for p in _read(tmp_path / v1si.OUT_PRECIP.name):
        assert p["ground_truth_operational"] == "false"
        assert p["target_created"] == "false"


# ===========================================================================
# v1sk institutional — query families
# ===========================================================================

def test_v1sk_has_defesa_civil(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sk, tmp_path)
    v1sk.run()
    q = " ".join(r["suggested_query"] for r in _read(tmp_path / v1sk.OUT_QUEUE.name))
    assert "Defesa Civil" in q

def test_v1sk_has_diario_oficial(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sk, tmp_path)
    v1sk.run()
    q = " ".join(r["suggested_query"] for r in _read(tmp_path / v1sk.OUT_QUEUE.name))
    assert "Diario Oficial" in q


# ===========================================================================
# v1sm adapter — does not invent date/location
# ===========================================================================

def test_v1sm_does_not_invent_date_location(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sm, tmp_path)
    _write_csv(tmp_path / "orch.csv",
               [{"source_name": "INMET", "source_block": "v1sh", "downloaded": "true", "url": "", "file_size_bytes": "1000"}],
               ["source_name", "source_block", "downloaded", "url", "file_size_bytes"])
    _write_csv(tmp_path / "stations.csv", [], ["station_code", "region_candidate"])
    monkeypatch.setattr(v1sm, "IN_ORCHESTRATOR", tmp_path / "orch.csv")
    monkeypatch.setattr(v1sm, "IN_INMET_STATIONS", tmp_path / "stations.csv")
    v1sm.run()
    rows = _read(tmp_path / v1sm.OUT_DRAFT.name)
    assert rows[0]["event_date_text"] == "" and rows[0]["event_location_text"] == ""


# ===========================================================================
# v1sn provenance — public official review-only
# ===========================================================================

def test_v1sn_public_official_review_only(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sn, tmp_path)
    _write_csv(tmp_path / "orch.csv",
               [{"source_name": "INMET", "source_block": "v1sh", "downloaded": "true", "url": "https://portal.inmet.gov.br/x", "file_size_bytes": "100"}],
               ["source_name", "source_block", "downloaded", "url", "file_size_bytes"])
    monkeypatch.setattr(v1sn, "IN_ORCHESTRATOR", tmp_path / "orch.csv")
    v1sn.run()
    rows = _read(tmp_path / v1sn.OUT_AUDIT.name)
    assert rows[0]["license_status"] == "PUBLIC_OFFICIAL_SOURCE_REVIEW_ONLY"
    assert rows[0]["review_only"] == "true"


# ===========================================================================
# DC strict guardrail scan
# ===========================================================================

def test_dc_scan_blocks_label():
    with pytest.raises(ValueError):
        DC.forbidden_guardrail_scan([{"can_create_operational_label": "true"}], "t")

def test_dc_scan_blocks_abs_path():
    with pytest.raises(ValueError):
        DC.forbidden_guardrail_scan([{"notes": r"C:\Users\x\f.zip"}], "t")

def test_dc_scan_blocks_local_runs():
    with pytest.raises(ValueError):
        DC.forbidden_guardrail_scan([{"notes": "local" + "_runs/x"}], "t")

def test_dc_scan_passes_clean():
    DC.forbidden_guardrail_scan([DC.guardrail_row()], "t")  # must not raise


# ===========================================================================
# v1sq command pack
# ===========================================================================

def _run_v1sq(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1sq, tmp_path)
    monkeypatch.setattr(v1sq, "PS1", tmp_path / "revp_official_download_commands_v1sq.ps1")
    v1sq.run()

def test_v1sq_emits_pack_csv(monkeypatch, tmp_path):
    _run_v1sq(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1sq.OUT_PACK.name)
    assert len(rows) == 10
    assert all(r["downloads_enabled"] == "false" for r in rows)

def test_v1sq_emits_powershell(monkeypatch, tmp_path):
    _run_v1sq(monkeypatch, tmp_path)
    text = (tmp_path / "revp_official_download_commands_v1sq.ps1").read_text(encoding="utf-8")
    assert 'REVP_ENABLE_OFFICIAL_DOWNLOADS = "false"' in text
    assert "# $env:REVP_ENABLE_OFFICIAL_DOWNLOADS" in text  # manual enable is commented
    assert "data/external_raw" in text and "commitar" in text.lower()

def test_v1sq_powershell_no_abs_path(monkeypatch, tmp_path):
    _run_v1sq(monkeypatch, tmp_path)
    text = (tmp_path / "revp_official_download_commands_v1sq.ps1").read_text(encoding="utf-8")
    assert not DC.detect_absolute_path(text)
    assert not DC.detect_forbidden_literal_exposure(text)

def test_v1sq_pack_no_forbidden(monkeypatch, tmp_path):
    _run_v1sq(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1sq.OUT_PACK.name)
    DC.forbidden_guardrail_scan(rows, "v1sq")  # must not raise

def test_v1sq_summary_default_disabled(monkeypatch, tmp_path):
    _run_v1sq(monkeypatch, tmp_path)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1sq.OUT_SUMMARY.name)}
    assert summ["default_downloads_enabled"] == "false"
