"""Tests for v1uh — Completion Report."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uh_completion_report.py")


def _run(tmp_path, out_dir=None, docs_dir=None):
    if out_dir is None:
        out_dir = "datasets/protocolo_c"
    if docs_dir is None:
        docs_dir = os.path.join(tmp_path, "docs")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--out-dir", out_dir, "--docs-dir", docs_dir],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out_dir, docs_dir


class TestCompletionReport:
    def test_runs_without_crash(self, tmp_path):
        _run(str(tmp_path))

    def test_blocker_matrix_created(self, tmp_path):
        out_dir, _ = _run(str(tmp_path))
        path = os.path.join(out_dir, "v1uh_ground_reference_candidate_blocker_matrix.csv")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    def test_blocker_matrix_has_label_forbidden(self, tmp_path):
        out_dir, _ = _run(str(tmp_path))
        path = os.path.join(out_dir, "v1uh_ground_reference_candidate_blocker_matrix.csv")
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        label_rows = [r for r in rows if r["blocker"] == "label_forbidden"]
        assert len(label_rows) >= 1
        for r in label_rows:
            assert r["blocker_status"] == "ACTIVE"

    def test_next_actions_created(self, tmp_path):
        out_dir, _ = _run(str(tmp_path))
        path = os.path.join(out_dir, "v1uh_next_actions_registry.csv")
        assert os.path.exists(path)

    def test_manifest_created(self, tmp_path):
        out_dir, _ = _run(str(tmp_path))
        path = os.path.join(out_dir, "v1uh_versionable_artifacts_manifest.csv")
        assert os.path.exists(path)

    def test_report_contains_guardrails(self, tmp_path):
        _, docs_dir = _run(str(tmp_path))
        path = os.path.join(docs_dir,
                            "protocolo_c_relatorio_v1uh_formal_response_intake.md")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ground_truth_operational" in content
        assert "can_create_ground_reference" in content
        assert "no_overlay_executed" in content

    def test_report_contains_invariants(self, tmp_path):
        _, docs_dir = _run(str(tmp_path))
        path = os.path.join(docs_dir,
                            "protocolo_c_relatorio_v1uh_formal_response_intake.md")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Nenhum ground reference criado" in content
        assert "Nenhum label de treinamento criado" in content
        assert "Nenhum overlay executado" in content

    def test_status_file_created(self, tmp_path):
        _, docs_dir = _run(str(tmp_path))
        path = os.path.join(docs_dir, "protocolo_c_status_atual_v1uh.md")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ground_truth_operational=false" in content

    def test_no_ground_reference_in_any_csv(self, tmp_path):
        out_dir, _ = _run(str(tmp_path))
        for fname in os.listdir(out_dir):
            if not fname.startswith("v1uh_") or not fname.endswith(".csv"):
                continue
            path = os.path.join(out_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "can_create_ground_reference" in row:
                        assert row["can_create_ground_reference"] != "true", (
                            f"can_create_ground_reference=true found in {fname}")
                    if "can_create_training_label" in row:
                        assert row["can_create_training_label"] != "true", (
                            f"can_create_training_label=true found in {fname}")

    def test_no_absolute_paths_in_manifest(self, tmp_path):
        out_dir, _ = _run(str(tmp_path))
        path = os.path.join(out_dir, "v1uh_versionable_artifacts_manifest.csv")
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            ap = r.get("artifact_path", "")
            assert not ap.startswith("C:"), f"Absolute path in manifest: {ap}"
            assert not ap.startswith("/home"), f"Absolute path in manifest: {ap}"
