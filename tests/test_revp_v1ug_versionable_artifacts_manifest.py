"""Tests for v1ug — Versionable Artifacts Manifest."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_versionable_artifacts_manifest.py")

COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_versionable_artifacts_manifest.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestVersionableArtifactsManifest:
    def test_runs_and_produces_output(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)

    def test_columns(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in COLUMNS:
            assert col in cols, f"Column missing: {col}"

    def test_artifact_ids_unique(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [r["artifact_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_all_protocol_version_v1ug(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["protocol_version"] == "v1ug"

    def test_no_local_runs_or_raw_data(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert "local_runs" not in r["artifact_path"]
            assert "local_only" not in r["artifact_path"]
            assert not r["artifact_path"].endswith(".tif")
            assert not r["artifact_path"].endswith(".npz")

    def test_has_configs_scripts_datasets(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        types = {r["artifact_type"] for r in rows}
        assert "config" in types
        assert "script" in types
        assert "dataset" in types

    def test_existing_files_have_sha256(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["is_versionable"] == "true":
                assert r["sha256_prefix"] != "MISSING", (
                    f"Existing file should have SHA256: {r['artifact_path']}"
                )
                assert len(r["sha256_prefix"]) == 16

    def test_at_least_15_artifacts(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 15, f"Expected >= 15 artifacts, got {len(rows)}"
