from tests.test_revp_v1ux_curitiba_download_target_builder import run_fixture_pipeline, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_candidate_evidence_classifier_allows_review_only_advancement(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    run_fixture_pipeline(data, raw)
    rows = common.load_csv(common.dataset_path("v1ux_curitiba_candidate_evidence_classification.csv"))
    assert any(r["evidence_class"] == "CURITIBA_EVENT_TABLE_CANDIDATE_FOR_REVIEW" for r in rows)
    assert any(r["evidence_class"] == "CURITIBA_GEODATA_CONTEXT_LAYER" for r in rows)
    assert all(r["can_advance_to_overlay_preflight"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
