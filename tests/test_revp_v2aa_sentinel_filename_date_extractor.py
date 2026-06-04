from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def test_filename_extractor_recognizes_patterns_and_blocks_ambiguous(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    common.run_patch_source_registry_scanner(common.parse_args([]))
    rows = common.run_sentinel_filename_date_extractor(common.parse_args([]))
    by_patch = {r["patch_id"]: r for r in rows}
    # S2A canonical SAFE name -> date + platform recognized.
    assert by_patch["P_S2A"]["extracted_date"] == "2022-05-25"
    assert by_patch["P_S2A"]["sentinel_platform"] == "S2A"
    assert by_patch["P_S2A"]["extraction_status"] == "DATE_EXTRACTED"
    # S2B short form.
    assert by_patch["P_S2B"]["extracted_date"] == "2022-05-30"
    assert by_patch["P_S2B"]["sentinel_platform"] == "S2B"
    # Ambiguous multi-date value is blocked, not selected.
    assert by_patch["P_AMB"]["extraction_status"] == "BLOCKED_AMBIGUOUS_MULTIPLE_DATES"
    assert by_patch["P_AMB"]["ambiguity_status"] == "AMBIGUOUS"
    assert by_patch["P_AMB"]["extracted_date"] == ""
    # No-date filename produces no extraction row.
    assert "P_NODATE" not in by_patch


def test_filename_extractor_rejects_bare_year(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    # A value with only a bare year must not yield a date.
    dates, pattern = common.extract_dates_from_text("annual_report_2022_recife")
    assert dates == []
