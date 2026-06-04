from tests.test_revp_v2ae_canonical_region_registry_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ae_common as common


def test_reopen_conditions_do_not_execute_search(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_region_reopen_condition_builder(common.parse_args([]))
    by_region = {r["region"]: r for r in rows}
    assert set(by_region) == {"REC", "PET", "CUR"}
    for r in rows:
        # Reopen requires a new public source, never region/name/order/inference.
        assert "required_public_evidence" in r and r["required_public_evidence"]
        for forbidden in ("region", "name_similarity", "file_order", "inferred_date", "inferred_crosswalk"):
            assert forbidden in r["forbidden_reopen_basis"]
        assert "no search" in r["notes"].lower()
