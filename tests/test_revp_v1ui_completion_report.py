"""Tests for v1ui — Completion Report."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_completion_report.py")


def _run(tmp_path):
    docs_dir = os.path.join(tmp_path, "docs")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--out-dir", "datasets/protocolo_c",
         "--docs-dir", docs_dir],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return docs_dir


class TestCompletionReport:
    def test_runs_without_crash(self, tmp_path):
        _run(str(tmp_path))

    def test_next_actions_created(self, tmp_path):
        _run(str(tmp_path))
        path = "datasets/protocolo_c/v1ui_next_actions_registry.csv"
        assert os.path.exists(path)

    def test_manifest_created(self, tmp_path):
        _run(str(tmp_path))
        path = "datasets/protocolo_c/v1ui_versionable_artifacts_manifest.csv"
        assert os.path.exists(path)

    def test_report_contains_guardrails(self, tmp_path):
        docs_dir = _run(str(tmp_path))
        path = os.path.join(docs_dir, "protocolo_c_relatorio_v1ui_public_official_discovery.md")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ground_truth_operational" in content
        assert "can_create_ground_reference" in content
        assert "no_overlay_executed" in content
        assert "LEGACY_SECONDARY_ONLY" in content

    def test_report_contains_invariants(self, tmp_path):
        docs_dir = _run(str(tmp_path))
        path = os.path.join(docs_dir, "protocolo_c_relatorio_v1ui_public_official_discovery.md")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Nenhum ground reference criado" in content
        assert "Nenhum label de treinamento criado" in content
        assert "Nenhum overlay executado" in content

    def test_status_file_created(self, tmp_path):
        docs_dir = _run(str(tmp_path))
        path = os.path.join(docs_dir, "protocolo_c_status_atual_v1ui.md")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ground_truth_operational=false" in content
        assert "LEGACY_SECONDARY_ONLY" in content

    def test_no_ground_reference_in_manifest(self, tmp_path):
        _run(str(tmp_path))
        path = "datasets/protocolo_c/v1ui_versionable_artifacts_manifest.csv"
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert not r["artifact_path"].startswith("C:")
            assert "local_only" not in r["artifact_path"]

    def test_no_absolute_paths_in_next_actions(self, tmp_path):
        _run(str(tmp_path))
        path = "datasets/protocolo_c/v1ui_next_actions_registry.csv"
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for v in r.values():
                assert not v.startswith("C:\\"), f"Absolute path found: {v}"
