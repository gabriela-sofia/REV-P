import scripts.protocolo_c.revp_v1uo_multiregion_common as common


def test_source_registry_contains_sources_by_region(tmp_path):
    rows = common.run_public_source_registry_builder(str(tmp_path / "sources.csv"))
    regions = {r["region"] for r in rows}
    assert {"REC", "PET", "CUR", "ALL"}.issubset(regions)
    assert any(r["source_type"] == "CKAN" for r in rows)
    assert any(r["can_contain_observed_geometry"] == "true" for r in rows)
