import pytest

import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all, write_csv


def test_regression_passes_on_clean_stack(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    common.run_master_orchestrator(common.parse_args([]))
    rows = common.run_guardrail_regression(common.parse_args([]))
    assert all(r["status"] == "PASS" for r in rows)


def test_regression_fails_on_forbidden_kv(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    write_csv(data / "v2am_bad.csv", ["id", "note"],
              [{"id": "X", "note": "promotion_allowed=true"}])
    with pytest.raises(ValueError):
        common.run_guardrail_regression(common.parse_args([]))


def test_regression_fails_on_absolute_path(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    write_csv(data / "v2am_bad.csv", ["id", "note"],
              [{"id": "X", "note": "C:\\Users\\x\\file.csv"}])
    with pytest.raises(ValueError):
        common.run_guardrail_regression(common.parse_args([]))


def test_regression_fails_on_local_only(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    (atlas / "v2am_bad.md").write_text("contains local" + "_only marker\n", encoding="utf-8")
    with pytest.raises(ValueError):
        common.run_guardrail_regression(common.parse_args([]))
