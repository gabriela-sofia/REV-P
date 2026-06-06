import pytest

import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, write_csv


def test_regression_passes_on_clean_outputs(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_markdown_section_bundle_builder(common.parse_args([]))
    common.run_latex_section_bundle_builder(common.parse_args([]))
    common.run_table_caption_export_builder(common.parse_args([]))
    rows = common.run_safe_language_regression(common.parse_args([]))
    assert all(r["status"] == "PASS" for r in rows)


def test_regression_fails_on_absolute_path(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    write_csv(data / "v2al_bad_artifact.csv", ["regression_id", "note"],
              [{"regression_id": "X", "note": "C:\\Users\\x\\manuscrito.tex"}])
    with pytest.raises(ValueError):
        common.run_safe_language_regression(common.parse_args([]))


def test_regression_fails_on_forbidden_kv(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    write_csv(data / "v2al_bad_artifact.csv", ["regression_id", "note"],
              [{"regression_id": "X", "note": "ground_truth=true"}])
    with pytest.raises(ValueError):
        common.run_safe_language_regression(common.parse_args([]))


def test_regression_fails_on_local_only(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    write_csv(data / "v2al_bad_artifact.csv", ["regression_id", "note"],
              [{"regression_id": "X", "note": "local" + "_only artifact"}])
    with pytest.raises(ValueError):
        common.run_safe_language_regression(common.parse_args([]))


def test_regression_fails_on_overclaim_doc(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    (integration / "v2al_bad_section.md").write_text(
        "# X\n\nO sistema entrega ground truth validado e deteccao de enchente.\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        common.run_safe_language_regression(common.parse_args([]))
