from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def _prepare(data):
    install_base_inputs(data)
    common.run_curitiba_context_only_hold_builder(common.parse_args([]))
    common.run_multiregion_closure_status_builder(common.parse_args([]))


def test_readiness_synthesizer_blocks_overlay_ground_reference_training(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    _prepare(data)
    rows = common.run_multiregion_readiness_synthesizer(common.parse_args([]))
    blocked_dims = {"overlay_readiness", "ground_reference_readiness", "training_readiness"}
    for row in rows:
        if row["dimension"] in blocked_dims:
            assert row["classification"] == "BLOCKED"
    # Occurrence geometry / occurrence coordinate are never STRONG.
    for row in rows:
        if row["dimension"] in {"occurrence_coordinate_support", "observed_geometry_support"}:
            assert row["classification"] == "ABSENT"
    assert all(r["no_overlay_executed"] == "true" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
