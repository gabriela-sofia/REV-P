import scripts.protocolo_c.revp_v1uo_multiregion_common as common


def test_region_adapters_exist_for_recife_petropolis_curitiba(tmp_path):
    rows = common.run_region_adapter_factory(str(tmp_path / "adapters.csv"))
    names = {r["adapter_name"] for r in rows}
    assert "recife_ckan_adapter" in names
    assert "petropolis_geohazard_adapter" in names
    assert "curitiba_public_geo_adapter" in names
