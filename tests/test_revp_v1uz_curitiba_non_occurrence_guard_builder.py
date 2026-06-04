from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def test_guard_prevents_context_alert_hydromet_as_occurrence(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_curitiba_non_occurrence_guard_builder(common.parse_args([]))
    evidence = {r["evidence_type"] for r in rows}
    assert "ADMINISTRATIVE_CONTEXT_LAYER" in evidence
    assert "DRAINAGE_CONTEXT_LAYER" in evidence
    assert "HYDROMET_SUPPORT" in evidence
    assert "OFFICIAL_ALERT_OR_NOTICE" in evidence
    assert "REGION_ONLY_EVENT_PATCH_LINKAGE" in evidence
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
    assert all(r["patch_bound_truth"] == "false" for r in rows)
    # Each guard must name a prohibited occurrence/truth use.
    assert all("use_as" in r["prohibited_use"] for r in rows)
