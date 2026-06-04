"""Tests for v1uj — Completion Report."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_completion_report.py")


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _run(out_dir, docs_dir):
    cmd = [sys.executable, SCRIPT, "--out-dir", out_dir, "--docs-dir", docs_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return result


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestCompletionReport:
    def test_runs_empty(self, tmp_path):
        out_dir = str(tmp_path)
        docs = os.path.join(out_dir, "docs")
        _run(out_dir, docs)
        events = _load(os.path.join(out_dir, "v1uj_event_status_registry.csv"))
        assert len(events) == 3  # tres eventos minimos
        assert os.path.exists(os.path.join(out_dir, "v1uj_next_actions_registry.csv"))
        assert os.path.exists(os.path.join(out_dir, "v1uj_versionable_artifacts_manifest.csv"))
        assert os.path.exists(os.path.join(
            docs, "protocolo_c_relatorio_v1uj_focused_public_source_deepening.md"))
        assert os.path.exists(os.path.join(docs, "protocolo_c_status_atual_v1uj.md"))

    def test_supervisor_review_path(self, tmp_path):
        out_dir = str(tmp_path)
        docs = os.path.join(out_dir, "docs")
        _write(os.path.join(out_dir, "v1uj_observed_candidate_promotion_audit.csv"),
               ["promotion_audit_id", "event_id", "source_tag", "max_status"],
               [{"promotion_audit_id": "PROM_0001", "event_id": "REC_2022_05_24_30",
                 "source_tag": "ckan",
                 "max_status": "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"}])
        _run(out_dir, docs)
        events = _load(os.path.join(out_dir, "v1uj_event_status_registry.csv"))
        rec = next(e for e in events if e["event_id"] == "REC_2022_05_24_30")
        assert rec["observed_candidates_for_review"] == "1"
        assert rec["path_to_supervisor_review"] == "READY"
        actions = _load(os.path.join(out_dir, "v1uj_next_actions_registry.csv"))
        assert any(a["action_type"] == "SUPERVISOR_REVIEW" for a in actions)

    def test_no_candidate_recommends_regional(self, tmp_path):
        out_dir = str(tmp_path)
        docs = os.path.join(out_dir, "docs")
        _run(out_dir, docs)
        actions = _load(os.path.join(out_dir, "v1uj_next_actions_registry.csv"))
        assert all(a["action_type"] == "DEEPEN_FOCUSED_OR_REGIONAL" for a in actions)
        status = os.path.join(docs, "protocolo_c_status_atual_v1uj.md")
        with open(status, "r", encoding="utf-8") as f:
            content = f.read()
        assert "v1uk - Recife CKAN Schema Deep Audit" in content

    def test_manifest_lists_configs(self, tmp_path):
        out_dir = str(tmp_path)
        docs = os.path.join(out_dir, "docs")
        _run(out_dir, docs)
        manifest = _load(os.path.join(out_dir, "v1uj_versionable_artifacts_manifest.csv"))
        paths = {m["artifact_path"] for m in manifest}
        assert any("v1uj_copernicus_ems_targets.yaml" in p for p in paths)
        assert any("v1uj_download_policy.yaml" in p for p in paths)
