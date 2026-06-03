"""Tests for v1ug — Completion Report."""

import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_completion_report.py")


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_completion_report.md")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestCompletionReport:
    def test_runs_and_produces_output(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)

    def test_contains_guardrails(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ground_truth_operational" in content
        assert "can_create_ground_reference" in content
        assert "can_create_training_label" in content
        assert "no_overlay_executed" in content
        assert "no_coordinates_invented" in content
        assert "review_package_only" in content
        assert "formal_request_only" in content

    def test_contains_event_summary(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "PET_2022_02_15" in content
        assert "PET_2024_03_21_28" in content
        assert "REC_2022_05_24_30" in content

    def test_contains_gap_summary(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "FAIL" in content
        assert "PASS" in content
        assert "training_label_allowed" in content

    def test_contains_next_steps(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Next Steps" in content
        assert "ground reference" in content.lower()

    def test_contains_invariants(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Invariants" in content
        assert "Nenhum evento atingiu READY_FOR_GROUND_REFERENCE" in content
        assert "Nenhuma geometria observada adquirida" in content
        assert "Nenhum overlay executado" in content
        assert "Nenhum label de treinamento criado" in content

    def test_contains_artifact_manifest(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Artifact Manifest" in content
        assert "v1ug_event_gap_matrix.csv" in content

    def test_no_ready_for_ground_reference(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "READY_FOR_GROUND_REFERENCE" not in content or \
               "Nenhum evento atingiu READY_FOR_GROUND_REFERENCE" in content
