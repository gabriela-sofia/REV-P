from tests.test_revp_v2ae_canonical_region_registry_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ae_common as common


def test_blocker_consolidation_global_and_region(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_multiregion_blocker_consolidator(common.parse_args([]))
    blockers = {r["blocker"] for r in rows}
    # Global blockers present.
    assert "no_observed_geometry" in blockers
    assert "no_ground_reference" in blockers
    # Classified correctly.
    by_blocker = {r["blocker"]: r for r in rows if r["scope"] == "GLOBAL"}
    assert by_blocker["no_observed_geometry"]["blocker_class"] == "CRITICAL_BLOCKER"
    assert by_blocker["no_overlay"]["blocker_class"] == "GLOBAL_BLOCKER"
    # Region descriptor blockers present.
    region_blockers = {(r["region"], r["blocker"]) for r in rows if r["scope"] == "REGION"}
    assert ("REC", "locality_only") in region_blockers
    assert ("PET", "document_only") in region_blockers
    assert ("CUR", "context_only") in region_blockers
    # No blocker is removed (all status BLOCKED).
    assert all(r["status"] == "BLOCKED" for r in rows)
