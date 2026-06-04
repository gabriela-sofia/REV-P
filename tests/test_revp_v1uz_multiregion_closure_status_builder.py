from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def test_closure_status_synthesizes_three_regions(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_curitiba_context_only_hold_builder(common.parse_args([]))
    rows = common.run_multiregion_closure_status_builder(common.parse_args([]))
    by_region = {r["region"]: r for r in rows}
    assert by_region["REC"]["closure_status"] == "RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL"
    assert by_region["PET"]["closure_status"] == "PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA"
    assert by_region["CUR"]["closure_status"] == "CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL"
    # Curitiba closure uses the candidate event id from the hold registry.
    assert by_region["CUR"]["event_id"] == "CUR_2022_01_15"
    assert all(r["geometry_status"] == "NO_OBSERVED_GEOMETRY" for r in rows)
    assert all(r["overlay_status"] == "BLOCKED" for r in rows)
    assert all(r["ground_reference_status"] == "BLOCKED" for r in rows)
