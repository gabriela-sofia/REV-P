from tests.test_revp_v2ac_event_patch_v2_package_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ac_common as common


def test_migration_diff_confirms_additive_migration(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Snapshot v1us candidate registry before the diff.
    src = data / "v1us_event_patch_candidate_registry.csv"
    before = src.read_bytes()
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    rows = common.run_migration_diff_auditor(common.parse_args([]))
    assert rows
    assert all(r["old_outputs_modified"] == "false" for r in rows)
    assert all(r["migration_additive"] == "true" for r in rows)
    assert all(r["source_version"] == "v1us" and r["target_version"] == "v2ac" for r in rows)
    # New v2 schema fields are reported as added.
    added = rows[0]["fields_added"]
    for field in ("patch_namespace", "patch_source_registry", "explicit_crosswalk_id",
                  "sentinel_date_status", "crosswalk_status"):
        assert field in added
    # v1us source file untouched.
    assert src.read_bytes() == before
