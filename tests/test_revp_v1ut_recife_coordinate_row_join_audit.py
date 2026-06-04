import os
import shutil
from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_row_join_audit_detects_coordinate_context_not_candidate_table(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_contextual_equipment.csv", "context_equipment",
                  classification="INFRASTRUCTURE_CONTEXT")
    shutil.copy(os.path.join(os.path.dirname(__file__), "fixtures", "v1ut", "recife_candidate_row_registry_no_coord.csv"),
                os.path.join(data, "v1uk_recife_candidate_row_registry.csv"))
    common.run_coordinate_asset_locator()
    common.run_coordinate_schema_reparser()
    rows = common.run_coordinate_row_join_audit()
    assert rows[0]["join_status"] == "COORDINATE_ROWS_IN_DIFFERENT_CONTEXT_ASSET"
    assert rows[0]["can_create_ground_reference"] == "false"
