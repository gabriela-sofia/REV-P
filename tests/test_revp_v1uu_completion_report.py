import os
from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_completion_report_writes_manifest_and_blockers(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    result = common.run_all()
    manifest = common.load_csv(os.path.join(data, "v1uu_versionable_artifacts_manifest.csv"))
    blockers = common.load_csv(os.path.join(data, "v1uu_recife_ground_reference_blocker_matrix.csv"))
    assert result["attachments"] == 1
    assert manifest
    assert all(r["can_create_ground_reference"] == "false" for r in blockers)
    assert os.path.exists(os.path.join(common.DOCS_DIR, "protocolo_c_status_atual_v1uu.md"))
