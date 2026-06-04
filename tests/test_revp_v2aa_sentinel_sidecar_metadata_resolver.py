from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def test_sidecar_resolver_uses_explicit_dates_not_created_modified(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    common.run_patch_source_registry_scanner(common.parse_args([]))
    rows = common.run_sentinel_sidecar_metadata_resolver(common.parse_args([]))
    by_patch = {}
    for r in rows:
        by_patch.setdefault(r["patch_id"], []).append(r)
    # datetime sidecar resolves.
    assert by_patch["P_DT"][0]["resolved_date"] == "2022-05-24"
    assert by_patch["P_DT"][0]["date_field_used"] == "datetime"
    assert by_patch["P_DT"][0]["source_type"] == "JSON_SIDECAR"
    # sensing_date sidecar resolves with high hint.
    assert by_patch["P_SENS"][0]["resolved_date"] == "2022-05-25"
    assert by_patch["P_SENS"][0]["confidence_hint"] == "HIGH"
    # created_at / modified_at trap must NOT resolve a scene date.
    assert "P_TRAP" not in by_patch
    # No resolution ever uses created_at/modified_at as the field.
    assert all(r["date_field_used"] not in {"created_at", "modified_at"} for r in rows)
