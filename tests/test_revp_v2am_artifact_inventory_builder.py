import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all, read_csv


def test_inventory_lists_stage_artifacts(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_artifact_inventory_builder(common.parse_args([]))
    assert rows
    stages = {r["stage"] for r in rows}
    assert {"v2ah", "v2ai", "v2aj", "v2ak", "v2al"} & stages
    for r in rows:
        assert not r["path"].startswith("C:")
        assert ":\\" not in r["path"]
    assert (atlas / "v2am_appendix_artifact_index.md").exists()


def test_inventory_rejects_absolute_path_in_content():
    with __import__("pytest").raises(ValueError):
        common.assert_no_absolute_paths_in_content(
            [{"path": "C:\\Users\\gabriela\\x.csv"}])
