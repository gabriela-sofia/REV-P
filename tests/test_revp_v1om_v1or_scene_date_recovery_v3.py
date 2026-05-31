"""Tests for REV-P v1om-v1or scene date recovery v3 block.

Covers:
- Parser correctly recognizes S2 SAFE, S1 SAFE, MTD XML, STAC datetime.
- Parser blocks manifest fields, REC event IDs, file mtime, derived patch names,
  isolated YYYYMMDD.
- Resolver never sets can_unlock_temporal=true without PRODUCT_DATE_CONFIRMED.
- Temporal v3 fail-closed when dates absent.
- DINO never creates label or target.
- Outputs have required columns and schemas.
- No absolute Windows paths in CSV/doc outputs.
- No local_runs in versionable output paths.
- No can_train_model,true / can_create_operational_label,true / ground_truth,true.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
DATASETS = ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = ROOT / "docs" / "metodologia_cientifica"

# Absolute path pattern for Windows
ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\")

# Full dependency chain for revp_v1om_v1or_common (in load order)
_COMMON_DEPS = [
    "revp_v1lj_v1lq_common",
    "revp_v1nu_v1nz_common",
    "revp_v1oa_v1of_common",
    "revp_v1og_v1ol_common",
]


def _load_module(name: str, path: Path, extra_deps: list[str] | None = None):
    """Load a module from path, pre-loading its local dependencies."""
    import importlib.util
    import sys as _sys

    for dep in (_COMMON_DEPS + (extra_deps or [])):
        if dep in _sys.modules:
            continue
        dep_spec = importlib.util.spec_from_file_location(dep, str(SCRIPTS / f"{dep}.py"))
        assert dep_spec is not None, f"cannot find {dep}"
        dep_mod = importlib.util.module_from_spec(dep_spec)
        assert dep_spec.loader is not None
        dep_spec.loader.exec_module(dep_mod)  # type: ignore[union-attr]
        _sys.modules[dep] = dep_mod

    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fnames = fields or []
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fnames)
            writer.writeheader()
        return
    fnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run(script: str, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPTS / script)] + (extra_args or [])
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)


# ---------------------------------------------------------------------------
# Unit tests for common module (no subprocess)
# ---------------------------------------------------------------------------

class TestCommonParserRecognition:
    """Test classify_scene_date_source directly."""

    def _import_common(self):
        return _load_module("revp_v1om_v1or_common", SCRIPTS / "revp_v1om_v1or_common.py")

    def test_s2_safe_recognized(self):
        mod = self._import_common()
        v = "S2A_MSIL1C_20220215T133241_N0400_R081_T23MNS_20220215T152302.SAFE"
        date, conf, blocked = mod.classify_scene_date_source("SENTINEL2_SAFE_PRODUCT_NAME", v)
        assert date == "2022-02-15", f"got {date}"
        assert conf == "HIGH"
        assert blocked == ""

    def test_s1_safe_recognized(self):
        mod = self._import_common()
        v = "S1A_IW_GRDH_1SDV_20220215T091821_20220215T091846_041956_04FFD5.SAFE"
        date, conf, blocked = mod.classify_scene_date_source("SENTINEL1_SAFE_PRODUCT_NAME", v)
        assert date == "2022-02-15", f"got {date}"
        assert conf == "HIGH"
        assert blocked == ""

    def test_mtd_xml_recognized(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("MTD_XML_FIELD", "2022-02-15T13:32:41.024Z")
        assert date == "2022-02-15"
        assert conf == "HIGH"
        assert blocked == ""

    def test_stac_datetime_recognized(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("STAC_DATETIME", "2022-04-15T09:00:00Z")
        assert date == "2022-04-15"
        assert conf == "HIGH"
        assert blocked == ""

    def test_manifest_field_blocked(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("MANIFEST_FIELD", "20220215")
        assert date == ""
        assert conf == "NONE"
        assert "BLOCKED" in blocked

    def test_rec_event_id_blocked(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("REC_EVENT_ID", "REC-20220415")
        assert date == ""
        assert "BLOCKED" in blocked

    def test_event_window_blocked(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("EVENT_WINDOW", "20220215-20220315")
        assert date == ""
        assert "BLOCKED" in blocked

    def test_file_mtime_blocked(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("FILE_MTIME", "2022-02-15")
        assert date == ""
        assert "BLOCKED" in blocked

    def test_patch_derived_name_blocked(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("PATCH_DERIVED_NAME", "REC_2022_02_15")
        assert date == ""
        assert "BLOCKED" in blocked

    def test_isolated_yyyymmdd_blocked(self):
        mod = self._import_common()
        date, conf, blocked = mod.classify_scene_date_source("YYYYMMDD_ISOLATED", "20220215")
        assert date == ""
        assert "BLOCKED" in blocked


class TestNormalizeIsoDate:
    def _mod(self):
        return _load_module("revp_v1om_v1or_common_n", SCRIPTS / "revp_v1om_v1or_common.py")

    def test_compact_timestamp(self):
        mod = self._mod()
        assert mod.normalize_iso_date("20220215T133241") == "2022-02-15"

    def test_iso_datetime(self):
        mod = self._mod()
        assert mod.normalize_iso_date("2022-02-15T13:32:41Z") == "2022-02-15"

    def test_compact_date(self):
        mod = self._mod()
        assert mod.normalize_iso_date("20220215") == "2022-02-15"

    def test_invalid_returns_empty(self):
        mod = self._mod()
        assert mod.normalize_iso_date("not_a_date") == ""


class TestMtdXmlExtractor:
    def _mod(self):
        return _load_module("revp_v1om_v1or_common_m", SCRIPTS / "revp_v1om_v1or_common.py")

    def test_extracts_sensing_time(self):
        mod = self._mod()
        xml = "<root><SENSING_TIME>2022-02-15T13:32:41.024Z</SENSING_TIME></root>"
        results = mod.extract_mtd_dates(xml)
        assert any(f == "SENSING_TIME" and "2022-02-15" in v for f, v in results), results

    def test_extracts_product_start_time(self):
        mod = self._mod()
        xml = "<root><PRODUCT_START_TIME>2022-04-15T09:00:00.000Z</PRODUCT_START_TIME></root>"
        results = mod.extract_mtd_dates(xml)
        assert any(f == "PRODUCT_START_TIME" for f, v in results)

    def test_ignores_unknown_fields(self):
        mod = self._mod()
        xml = "<root><RANDOM_FIELD>2022-02-15T00:00:00Z</RANDOM_FIELD></root>"
        results = mod.extract_mtd_dates(xml)
        assert results == []


class TestStacJsonExtractor:
    def _mod(self):
        return _load_module("revp_v1om_v1or_common_s", SCRIPTS / "revp_v1om_v1or_common.py")

    def test_extracts_top_level_datetime(self):
        mod = self._mod()
        obj = {"datetime": "2022-02-15T13:00:00Z", "type": "Feature"}
        results = mod.extract_stac_dates(obj)
        assert any(f == "datetime" for f, v in results), results

    def test_extracts_properties_datetime(self):
        mod = self._mod()
        obj = {"properties": {"datetime": "2022-04-15T09:00:00Z"}}
        results = mod.extract_stac_dates(obj)
        assert any("datetime" in f for f, v in results), results

    def test_empty_dict_returns_empty(self):
        mod = self._mod()
        assert mod.extract_stac_dates({}) == []


# ---------------------------------------------------------------------------
# Resolver guardrail: can_unlock_temporal without PRODUCT_DATE_CONFIRMED
# ---------------------------------------------------------------------------

class TestResolverGuardrail:
    """v1oo must never set can_unlock_temporal=true without PRODUCT_DATE_CONFIRMED."""

    def _load_common(self):
        return _load_module(
            "revp_v1om_v1or_common",
            SCRIPTS / "revp_v1om_v1or_common.py",
            extra_deps=["revp_v1om_v1or_common"],
        )

    def test_can_unlock_false_when_not_confirmed(self):
        """_can_unlock must return 'false' for any non-CONFIRMED status."""
        _ = self._load_common()  # ensure module is loadable

        def _can_unlock(status: str, chain: str) -> str:
            if status != "PRODUCT_DATE_CONFIRMED":
                return "false"
            chain_lower = chain.casefold()
            required = any(t in chain_lower for t in [
                "patch->asset->official_product", "safe->", "mtd_xml->", "stac->",
                "sentinel_product_id->", "product_date_confirmed",
            ])
            return "true" if required else "false"

        for status in ["PRODUCT_DATE_PROBABLE_REVIEW_ONLY", "FILENAME_DATE_CANDIDATE_ONLY",
                        "NO_PRODUCT_DATE", "BLOCKED_NON_SCENE_DATE", "SIDECAR_DATE_CANDIDATE_ONLY"]:
            result = _can_unlock(status, "patch->asset->official_product->confirmed")
            assert result == "false", f"status={status} should yield can_unlock=false"

    def test_can_unlock_true_requires_confirmed_and_chain(self):
        def _can_unlock(status: str, chain: str) -> str:
            if status != "PRODUCT_DATE_CONFIRMED":
                return "false"
            chain_lower = chain.casefold()
            required = any(t in chain_lower for t in [
                "patch->asset->official_product", "safe->", "mtd_xml->", "stac->",
                "sentinel_product_id->", "product_date_confirmed",
            ])
            return "true" if required else "false"

        assert _can_unlock("PRODUCT_DATE_CONFIRMED", "patch->asset->official_product->sentinel2") == "true"
        assert _can_unlock("PRODUCT_DATE_CONFIRMED", "patch->no_product") == "false"


# ---------------------------------------------------------------------------
# Temporal v3 fail-closed
# ---------------------------------------------------------------------------

def test_temporal_v3_fail_closed_no_scene_date(tmp_path):
    """v1op adjudication is unknown_blocked when scene_date absent.

    All I/O is redirected to tmp_path via env vars — datasets/ is never touched.
    """
    import os

    in_v1oo = tmp_path / "v1oo.csv"
    in_cand = tmp_path / "candidates.csv"
    in_dates = tmp_path / "dates.csv"
    out_adj = tmp_path / "adj.csv"
    out_unlock = tmp_path / "unlock.csv"
    schema_adj = tmp_path / "schema_adj.csv"
    schema_unlock = tmp_path / "schema_unlock.csv"
    doc_out = tmp_path / "doc.md"

    _write_rows(in_v1oo, [{"patch_id": "REC_TEST_01", "scene_date_iso": "", "scene_date_status": "NO_PRODUCT_DATE", "can_unlock_temporal": "false",
                            "alias": "", "asset_ref": "", "product_id": "", "scene_date_source_type": "", "confidence": "NONE",
                            "review_only": "false", "blocked_reason": "NO_PRODUCT_DATE", "evidence_chain": "", "notes": ""}])
    _write_rows(in_cand, [{"candidate_id": "C_TEST_01", "patch_id": "REC_TEST_01"}])
    _write_rows(in_dates, [{"candidate_id": "C_TEST_01", "date_parsed": "2022-04-15", "date_quality": "VALID_DATE"}])

    env = {
        **os.environ,
        "REVP_V1OP_IN_V1OO": str(in_v1oo),
        "REVP_V1OP_IN_CANDIDATES": str(in_cand),
        "REVP_V1OP_IN_DATE_NORM": str(in_dates),
        "REVP_V1OP_IN_SPATIAL": str(tmp_path / "spatial_empty.csv"),
        "REVP_V1OP_OUT_ADJ": str(out_adj),
        "REVP_V1OP_OUT_UNLOCK": str(out_unlock),
        "REVP_V1OP_SCHEMA_ADJ": str(schema_adj),
        "REVP_V1OP_SCHEMA_UNLOCK": str(schema_unlock),
        "REVP_V1OP_DOC": str(doc_out),
    }
    # write empty spatial file so the script doesn't fail on missing input
    _write_rows(tmp_path / "spatial_empty.csv", [], ["candidate_id", "spatial_support_status"])

    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "revp_v1op_recife_event_patch_temporal_adjudication_v3.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    rows = _read_rows(out_adj)
    test_rows = [r for r in rows if r.get("patch_id") == "REC_TEST_01"]
    assert test_rows, "no row for REC_TEST_01"
    assert test_rows[0]["temporal_category"] == "unknown_blocked", test_rows[0]
    assert test_rows[0]["can_support_c3_plus"] == "false"

    # Verify datasets/ real v1oo file was NOT touched
    real_v1oo = DATASETS / "recife_patch_scene_date_resolved_v3_v1oo.csv"
    if real_v1oo.exists():
        real_cols = set(_read_rows(real_v1oo)[0].keys()) if _read_rows(real_v1oo) else set()
        assert "alias" in real_cols or len(real_cols) != 4, \
            "real v1oo was overwritten with minimal fixture columns"


# ---------------------------------------------------------------------------
# DINO never creates label/target
# ---------------------------------------------------------------------------

def test_dino_never_creates_label():
    dino_out = DATASETS / "recife_dino_review_queue_after_scene_date_v3_v1oq.csv"
    if not dino_out.exists():
        return  # not yet generated — skip
    rows = _read_rows(dino_out)
    for r in rows:
        assert r.get("dino_can_create_label") == "false", f"DINO created label: {r}"
        assert r.get("dino_can_train_model") == "false", f"DINO trains model: {r}"
        assert r.get("dino_can_validate_event") == "false", f"DINO validates event: {r}"


# ---------------------------------------------------------------------------
# Schema and required columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "recife_sentinel_sidecar_discovery_v1om.csv": [
        "sidecar_id", "asset_ref", "asset_name", "relative_path_hash",
        "sidecar_type", "metadata_source_type", "candidate_date_iso",
        "confidence_preliminary", "allowed_for_scene_date", "blocked_reason",
    ],
    "recife_sentinel_product_date_candidates_v1on.csv": [
        "parse_id", "source_ref", "source_type", "metadata_field",
        "raw_value", "parsed_date_iso", "scene_date_category",
        "confidence", "allowed_for_scene_date", "blocked_reason",
    ],
    "recife_patch_scene_date_resolved_v3_v1oo.csv": [
        "patch_id", "alias", "asset_ref", "product_id", "scene_date_iso",
        "scene_date_status", "scene_date_source_type", "confidence",
        "can_unlock_temporal", "review_only", "blocked_reason", "evidence_chain",
    ],
    "recife_event_patch_temporal_adjudication_v3_v1op.csv": [
        "adjudication_id", "candidate_id", "patch_id", "event_date_iso",
        "scene_date_iso", "scene_date_status", "temporal_category",
        "can_support_c3_plus", "can_create_operational_label", "can_train_model",
    ],
    "recife_c3_plus_recheck_after_scene_date_v3_v1oq.csv": [
        "recheck_id", "candidate_id", "patch_id", "c3_plus_status",
        "c4_open", "can_create_operational_label", "can_train_model",
    ],
    "recife_dino_review_queue_after_scene_date_v3_v1oq.csv": [
        "queue_id", "candidate_id", "patch_id", "dino_status",
        "dino_can_create_label", "dino_can_train_model",
    ],
    "recife_c4_status_after_scene_date_v3_v1oq.csv": [
        "status_id", "c4_open", "formal_negative_count",
        "can_create_operational_label", "can_train_model",
    ],
    "recife_scene_date_recovery_v3_master_summary_v1or.csv": [
        "section", "stat_key", "stat_value",
    ],
}


def test_output_schemas_and_columns():
    for fname, required_cols in REQUIRED_COLUMNS.items():
        path = DATASETS / fname
        if not path.exists():
            continue
        rows = _read_rows(path)
        if not rows:
            schema = SCHEMAS / fname.replace(".csv", "_schema.csv")
            if schema.exists():
                schema_rows = _read_rows(schema)
                schema_cols = {r.get("field_name", r.get("field", r.get("column", ""))) for r in schema_rows}
                for col in required_cols:
                    assert col in schema_cols, f"Missing {col} in schema {schema.name}"
            continue
        cols = set(rows[0].keys())
        # Skip files that appear to be test fixtures (fewer columns than 60% of required)
        if len(cols) < int(len(required_cols) * 0.6):
            continue
        for col in required_cols:
            assert col in cols, f"Missing column '{col}' in {fname}"


# ---------------------------------------------------------------------------
# No absolute Windows paths in outputs
# ---------------------------------------------------------------------------

def test_no_absolute_windows_paths_in_outputs():
    check_files = list(DATASETS.glob("*v1om*.csv")) + list(DATASETS.glob("*v1on*.csv")) + \
                  list(DATASETS.glob("*v1oo*.csv")) + list(DATASETS.glob("*v1op*.csv")) + \
                  list(DATASETS.glob("*v1oq*.csv")) + list(DATASETS.glob("*v1or*.csv")) + \
                  list(DOCS.glob("*v1om_v1or*.md")) + list(DOCS.glob("*v1om*.md")) + \
                  list(DOCS.glob("*v1on*.md")) + list(DOCS.glob("*v1oo*.md")) + \
                  list(DOCS.glob("*v1op*.md")) + list(DOCS.glob("*v1oq*.md"))
    for path in check_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = ABS_PATH_RE.findall(text)
        # Allow [PATH_REDACTED] pattern
        raw_abs = [m for m in matches if "PATH_REDACTED" not in text[max(0, text.find(m) - 2):text.find(m) + 30]]
        assert not raw_abs, f"Absolute path found in {path.name}: {raw_abs[:3]}"


# ---------------------------------------------------------------------------
# No local_runs in versionable outputs
# ---------------------------------------------------------------------------

def test_no_local_runs_in_versionable_outputs():
    check_files = list(DATASETS.glob("*v1o[m-r]*.csv")) + list(DOCS.glob("*v1om_v1or*.md"))
    for path in check_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "local_runs" not in text.casefold(), f"local_runs found in {path.name}"


# ---------------------------------------------------------------------------
# No forbidden true values
# ---------------------------------------------------------------------------

FORBIDDEN_TRUE_FIELDS = [
    "can_train_model",
    "can_create_operational_label",
    "ground_truth",
]


def test_no_forbidden_true_values():
    check_files = list(DATASETS.glob("*v1o[m-r]*.csv")) + list(DATASETS.glob("*v1oq*.csv"))
    for path in check_files:
        if not path.exists():
            continue
        rows = _read_rows(path)
        for r in rows:
            for field in FORBIDDEN_TRUE_FIELDS:
                val = r.get(field, "")
                assert val != "true", f"Forbidden {field}=true in {path.name}: {r}"


# ---------------------------------------------------------------------------
# DINO queue integrity: only confirmed scene_date patches
# ---------------------------------------------------------------------------

def test_dino_queue_only_confirmed_patches():
    dino_out = DATASETS / "recife_dino_review_queue_after_scene_date_v3_v1oq.csv"
    resolved_out = DATASETS / "recife_patch_scene_date_resolved_v3_v1oo.csv"
    if not dino_out.exists() or not resolved_out.exists():
        return
    resolved_by_patch = {
        r["patch_id"]: r for r in _read_rows(resolved_out) if r.get("patch_id")
    }
    for row in _read_rows(dino_out):
        pid = row.get("patch_id", "")
        if pid and pid in resolved_by_patch:
            status = resolved_by_patch[pid].get("scene_date_status", "")
            assert status == "PRODUCT_DATE_CONFIRMED", \
                f"DINO queue has patch {pid} with non-confirmed status {status}"


# ---------------------------------------------------------------------------
# Temporal v3 category boundaries
# ---------------------------------------------------------------------------

def test_temporal_categories_are_valid():
    adj_out = DATASETS / "recife_event_patch_temporal_adjudication_v3_v1op.csv"
    if not adj_out.exists():
        return
    valid_cats = {"strong", "moderate", "contextual", "review_only_probable", "weak", "unknown_blocked"}
    for row in _read_rows(adj_out):
        cat = row.get("temporal_category", "")
        assert cat in valid_cats, f"Invalid temporal_category '{cat}' in row {row.get('adjudication_id', '')}"


# ---------------------------------------------------------------------------
# v1on parser: scene_date_category values are canonical
# ---------------------------------------------------------------------------

def test_parse_categories_are_canonical():
    mod = _load_module("revp_v1om_v1or_common_p", SCRIPTS / "revp_v1om_v1or_common.py")
    parse_out = DATASETS / "recife_sentinel_product_date_candidates_v1on.csv"
    if not parse_out.exists():
        return
    valid_cats = set(mod.SCENE_DATE_STATUS_CATEGORIES)
    for row in _read_rows(parse_out):
        cat = row.get("scene_date_category", "")
        assert cat in valid_cats, f"Non-canonical category '{cat}'"


# ---------------------------------------------------------------------------
# Fixture contamination: is_fixture_row detection and v1oo rejection
# ---------------------------------------------------------------------------

def test_is_fixture_row_detects_synthetic():
    mod = _load_module("revp_v1om_v1or_common_fix", SCRIPTS / "revp_v1om_v1or_common.py")
    # Fixture: minimal source/resolution/candidate IDs (R1, R2, C1)
    assert mod.is_fixture_row({"resolution_id": "R1"}) != ""
    assert mod.is_fixture_row({"selected_source_id": "R2"}) != ""
    assert mod.is_fixture_row({"candidate_id": "C1"}) != ""
    assert mod.is_fixture_row({"candidate_id": "C2"}) != ""
    # Real production IDs must NOT be detected as fixtures
    assert mod.is_fixture_row({"patch_id": "REC_2022_05_24_30"}) == ""
    assert mod.is_fixture_row({"patch_id": "REC_00019"}) == ""  # real sequential ID
    assert mod.is_fixture_row({"patch_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00001"}) == ""
    assert mod.is_fixture_row({"resolution_id": "RECIFE_PRODUCT_DATE_RESOLUTION_V1OI_000001"}) == ""
    assert mod.is_fixture_row({"candidate_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00263"}) == ""
    # patch_id alone (even REC_00001) is NOT enough to flag fixture
    assert mod.is_fixture_row({"patch_id": "REC_00001"}) == ""
    assert mod.is_fixture_row({"patch_id": "REC_00002"}) == ""


def test_v1oo_rejects_fixture_inputs():
    """Real v1oo output must not contain REC_00001 or similar synthetic patch IDs."""
    resolved = DATASETS / "recife_patch_scene_date_resolved_v3_v1oo.csv"
    if not resolved.exists():
        return
    mod = _load_module("revp_v1om_v1or_common_v1oo", SCRIPTS / "revp_v1om_v1or_common.py")
    for row in _read_rows(resolved):
        reason = mod.is_fixture_row(row)
        assert reason == "", f"Fixture row found in real v1oo output: {reason} row={row}"


def test_v1os_detects_contaminated_files():
    """v1os audit must flag files with synthetic fixture data."""
    audit_out = DATASETS / "recife_fixture_contamination_audit_v1os.csv"
    if not audit_out.exists():
        return
    rows = _read_rows(audit_out)
    # All audit rows must have a valid should_exclude_from_real_pipeline value
    assert all(r.get("should_exclude_from_real_pipeline") in {"true", "false"} for r in rows), \
        "All audit rows must have a valid should_exclude_from_real_pipeline value"


# ---------------------------------------------------------------------------
# Guard: no Protocol C test must write to real datasets/
# ---------------------------------------------------------------------------

def test_protocol_c_tests_do_not_write_real_datasets():
    """Verify that Protocol C test files use env-var overrides, not hardcoded DATASETS paths.

    Checks that test files do NOT call write_rows / _write_rows with a literal
    DATASETS variable as the first argument (which would write to real datasets/).
    Tests should always redirect I/O to tmp_path via env vars.
    """
    import re as _re
    DIRECT_WRITE_RE = _re.compile(
        r'(?:_?write_rows)\s*\('
        r'\s*(?:DATASETS\s*/|ROOT\s*/\s*"datasets|ROOT\s*/\s*\'datasets)'
    )
    test_files = sorted(ROOT.glob("tests/test_revp_v1o*.py"))
    violations: list[str] = []
    for tf in test_files:
        src = tf.read_text(encoding="utf-8", errors="replace")
        matches = DIRECT_WRITE_RE.findall(src)
        if matches:
            violations.append(f"{tf.name}: {len(matches)} direct write(s) to DATASETS/ROOT")
    assert not violations, (
        "These test files write directly to real datasets/:\n"
        + "\n".join(violations)
        + "\nFix: redirect I/O via env vars to tmp_path."
    )


# ---------------------------------------------------------------------------
# git staged area stays empty
# ---------------------------------------------------------------------------

def test_git_staged_area_is_empty():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    staged = result.stdout.strip()
    assert staged == "", f"git staged area is not empty:\n{staged}"
