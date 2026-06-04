import os
from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, install_v1us_rec_candidate, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_completion_report_writes_manifest_and_guardrail_docs(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_contextual_equipment.csv", "context_equipment",
                  classification="INFRASTRUCTURE_CONTEXT")
    install_v1us_rec_candidate(data)
    common.run_all()
    manifest = common.load_csv(os.path.join(data, "v1ut_versionable_artifacts_manifest.csv"))
    blockers = common.load_csv(os.path.join(data, "v1ut_recife_ground_reference_blocker_matrix.csv"))
    assert manifest
    assert all(r["ground_truth_operational"] == "false" for r in blockers)
    assert os.path.exists(os.path.join(common.DOCS_DIR, "protocolo_c_status_atual_v1ut.md"))
