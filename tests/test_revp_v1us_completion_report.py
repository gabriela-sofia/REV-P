import csv, glob, os
import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def _run_full_pipeline():
    common.run_event_temporal_window_linker()
    common.run_external_evidence_attacher()
    common.run_phenomenon_status_attacher()
    common.run_geometry_blocker_attacher()
    common.run_event_patch_readiness_matrix_builder()
    common.run_dino_review_support_attacher()
    common.run_next_action_ranker()
    return common.run_completion_report()


def test_completion_writes_docs_manifest_and_blocker_matrix(tmp_path, monkeypatch):
    data, root, docs, _ = set_env(tmp_path, monkeypatch)
    monkeypatch.setattr(common, "V1US_ARTIFACTS", [])
    build_chain(data, root)
    result = _run_full_pipeline()
    assert result["next_action"] == "v1ut - Recife Coordinate Recovery from Public CKAN"
    assert os.path.exists(os.path.join(str(docs), "protocolo_c_status_atual_v1us.md"))
    assert os.path.exists(os.path.join(str(data), "v1us_ground_reference_blocker_matrix.csv"))
    assert os.path.exists(os.path.join(str(data), "v1us_next_actions_registry.csv"))


def test_no_forbidden_truth_flags_in_any_csv(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    monkeypatch.setattr(common, "V1US_ARTIFACTS", [])
    build_chain(data, root)
    _run_full_pipeline()
    forbidden = ("can_create_ground_reference", "can_create_training_label", "ground_truth_operational")
    for path in glob.glob(os.path.join(str(data), "v1us_*.csv")):
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for col in forbidden:
                    if col in row:
                        assert row[col].strip().lower() != "true", f"{col}=true in {path}"
