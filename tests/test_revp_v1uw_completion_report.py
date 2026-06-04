import os
from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_completion_report_writes_manifest_and_next_action(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    result = common.run_all(common.parse_args(["--dry-run"]))
    manifest = common.load_csv(os.path.join(data, "v1uw_versionable_artifacts_manifest.csv"))
    next_actions = common.load_csv(os.path.join(data, "v1uw_next_actions_registry.csv"))
    assert manifest
    assert next_actions
    assert result["prelinks"] == 1
