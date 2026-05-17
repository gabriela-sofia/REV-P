from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.dino.revp_v1gp_dino_github_release_readiness_audit import (
    FORBIDDEN_EXTENSIONS,
    METHODOLOGICAL_PROTECTIONS,
    REQUIRED_DOCS,
    REQUIRED_SCRIPTS,
    REQUIRED_TESTS,
    build_methodology_matrix,
    check_docs_coverage,
    check_forbidden_artifacts,
    check_operational_coverage,
    check_private_paths,
    collect_versionable_files,
    determine_readiness,
    is_forbidden_file,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tree(base: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# is_forbidden_file
# ---------------------------------------------------------------------------

class TestIsForbiddenFile:
    @pytest.mark.parametrize("name", [
        "embedding.npz",
        "patch.npy",
        "scene.tif",
        "scene.tiff",
        "layer.vrt",
        "layer.aux.xml",
    ])
    def test_forbidden_extensions(self, name: str, tmp_path: Path) -> None:
        p = tmp_path / name
        p.write_bytes(b"")
        assert is_forbidden_file(p)

    @pytest.mark.parametrize("name", [
        "script.py",
        "protocol.md",
        "config.yaml",
        "summary.json",
        "manifest.csv",
    ])
    def test_allowed_extensions(self, name: str, tmp_path: Path) -> None:
        p = tmp_path / name
        p.write_bytes(b"")
        assert not is_forbidden_file(p)


# ---------------------------------------------------------------------------
# check_forbidden_artifacts
# ---------------------------------------------------------------------------

class TestCheckForbiddenArtifacts:
    def test_no_issues_when_clean(self, tmp_path: Path) -> None:
        make_tree(tmp_path, {
            "scripts/dino/script.py": "# ok",
            "README.md": "# readme",
        })
        issues = check_forbidden_artifacts(tmp_path)
        assert issues == []

    def test_detects_npz_outside_local_runs(self, tmp_path: Path) -> None:
        make_tree(tmp_path, {
            "scripts/dino/embedding.npz": "",
        })
        issues = check_forbidden_artifacts(tmp_path)
        assert any("embedding.npz" in i["file"] for i in issues)
        assert all(i["status"] == "FAIL" for i in issues)

    def test_ignores_npz_inside_local_runs(self, tmp_path: Path) -> None:
        make_tree(tmp_path, {
            "local_runs/dino_embeddings/v1fx/embeddings/patch.npz": "",
        })
        issues = check_forbidden_artifacts(tmp_path)
        assert issues == []

    def test_detects_tif_outside_local_runs(self, tmp_path: Path) -> None:
        make_tree(tmp_path, {
            "docs/scene.tif": "",
        })
        issues = check_forbidden_artifacts(tmp_path)
        assert any("scene.tif" in i["file"] for i in issues)

    def test_allows_py_outside_local_runs(self, tmp_path: Path) -> None:
        make_tree(tmp_path, {
            "scripts/dino/audit.py": "print('ok')",
        })
        issues = check_forbidden_artifacts(tmp_path)
        assert issues == []


# ---------------------------------------------------------------------------
# check_private_paths
# ---------------------------------------------------------------------------

class TestCheckPrivatePaths:
    def test_no_issues_when_clean(self, tmp_path: Path) -> None:
        p = tmp_path / "script.py"
        p.write_text("import pathlib\n", encoding="utf-8")
        issues = check_private_paths(tmp_path, [p])
        assert issues == []

    def test_detects_windows_private_path(self, tmp_path: Path) -> None:
        p = tmp_path / "config.yaml"
        _priv = "C:\\Users\\" + "gabriela" + "\\Documents\\PROJETO"
        p.write_text(f"input_dir: {_priv}\n", encoding="utf-8")
        issues = check_private_paths(tmp_path, [p])
        assert len(issues) >= 1
        assert issues[0]["status"] == "FAIL"

    def test_detects_posix_home_path(self, tmp_path: Path) -> None:
        p = tmp_path / "script.py"
        _priv = "/home/" + "gabriela" + "/data"
        p.write_text(f"ROOT = '{_priv}'\n", encoding="utf-8")
        issues = check_private_paths(tmp_path, [p])
        assert len(issues) >= 1

    def test_neutral_example_paths_are_ok(self, tmp_path: Path) -> None:
        p = tmp_path / "config.yaml"
        p.write_text("# set input_dir to your data directory\ninput_dir: /path/to/data\n", encoding="utf-8")
        issues = check_private_paths(tmp_path, [p])
        assert issues == []


# ---------------------------------------------------------------------------
# build_methodology_matrix
# ---------------------------------------------------------------------------

class TestBuildMethodologyMatrix:
    def test_returns_row_per_file(self, tmp_path: Path) -> None:
        matrix = build_methodology_matrix(tmp_path)
        assert isinstance(matrix, list)
        assert len(matrix) > 0
        for row in matrix:
            assert "file" in row
            assert "exists" in row

    def test_absent_file_marked_no(self, tmp_path: Path) -> None:
        matrix = build_methodology_matrix(tmp_path)
        for row in matrix:
            assert row["exists"] == "no"

    def test_detects_term_present(self, tmp_path: Path) -> None:
        doc = tmp_path / "docs" / "dino_sentinel_embedding_protocol.md"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(
            "review_only=true\nsupervised_training=false\nlabels_created=false\npredictive_claims=false\nmultimodal hold\n",
            encoding="utf-8",
        )
        matrix = build_methodology_matrix(tmp_path)
        protocol_row = next((r for r in matrix if "dino_sentinel_embedding_protocol" in r["file"]), None)
        assert protocol_row is not None
        assert protocol_row["exists"] == "yes"
        assert protocol_row["review_only"] == "present"
        assert protocol_row["supervised_training"] == "present"


# ---------------------------------------------------------------------------
# check_docs_coverage
# ---------------------------------------------------------------------------

class TestCheckDocsCoverage:
    def test_all_missing(self, tmp_path: Path) -> None:
        rows = check_docs_coverage(tmp_path)
        missing = [r for r in rows if r.get("exists") == "no"]
        assert len(missing) > 0

    def test_present_docs_detected(self, tmp_path: Path) -> None:
        for rel in REQUIRED_DOCS:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# doc\n", encoding="utf-8")
        readme = tmp_path / "README.md"
        readme.write_text(
            "See docs/dino_sentinel_embedding_protocol.md\n"
            "See docs/dino_sentinel_scientific_evidence_summary.md\n"
            "See docs/dino_command_registry.md\n",
            encoding="utf-8",
        )
        rows = check_docs_coverage(tmp_path)
        missing = [r for r in rows if r.get("exists") == "no"]
        assert missing == []


# ---------------------------------------------------------------------------
# check_operational_coverage
# ---------------------------------------------------------------------------

class TestCheckOperationalCoverage:
    def test_missing_scripts_detected(self, tmp_path: Path) -> None:
        rows = check_operational_coverage(tmp_path)
        file_rows = [r for r in rows if r.get("kind") == "file" and r.get("exists") == "no"]
        assert len(file_rows) > 0

    def test_present_scripts_detected(self, tmp_path: Path) -> None:
        for rel in REQUIRED_SCRIPTS + REQUIRED_TESTS:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# script\n", encoding="utf-8")
        registry = tmp_path / "docs" / "dino_command_registry.md"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text("## v1gn\n## v1go\n## v1gp\n", encoding="utf-8")
        rows = check_operational_coverage(tmp_path)
        file_rows = [r for r in rows if r.get("kind") == "file"]
        assert all(r.get("exists") == "yes" for r in file_rows)


# ---------------------------------------------------------------------------
# determine_readiness
# ---------------------------------------------------------------------------

class TestDetermineReadiness:
    def _good_matrix(self) -> list[dict[str, str]]:
        row: dict[str, str] = {"file": "docs/dino_sentinel_embedding_protocol.md", "exists": "yes"}
        for term in METHODOLOGICAL_PROTECTIONS:
            row[term] = "present"
        return [row]

    def _good_docs(self) -> list[dict[str, str]]:
        rows = [{"doc": d, "exists": "yes"} for d in REQUIRED_DOCS]
        rows += [{"doc": f"README.md -> marker", "exists": "yes"}]
        return rows

    def _good_ops(self) -> list[dict[str, str]]:
        rows = [{"item": s, "kind": "file", "exists": "yes"} for s in REQUIRED_SCRIPTS + REQUIRED_TESTS]
        rows += [{"item": "registry v1gn", "kind": "registry_entry", "exists": "yes"}]
        return rows

    def test_ready_when_all_pass(self) -> None:
        status, notes = determine_readiness([], [], self._good_docs(), self._good_ops(), self._good_matrix())
        assert status == "READY_FOR_LOCAL_COMMIT"
        assert notes == []

    def test_blocked_by_forbidden_artifact(self) -> None:
        forbidden = [{"file": "scripts/dino/bad.npz", "reason": "forbidden_extension", "status": "FAIL"}]
        status, notes = determine_readiness(forbidden, [], self._good_docs(), self._good_ops(), self._good_matrix())
        assert status == "BLOCKED"
        assert any("bad.npz" in n for n in notes)

    def test_blocked_by_private_path(self) -> None:
        _pat = "C:\\\\Users\\\\" + "gabriela"
        private = [{"file": "config.yaml", "pattern": _pat, "match_count": "1", "status": "FAIL"}]
        status, notes = determine_readiness([], private, self._good_docs(), self._good_ops(), self._good_matrix())
        assert status == "BLOCKED"
        assert any("config.yaml" in n for n in notes)

    def test_blocked_by_missing_doc(self) -> None:
        docs = [{"doc": REQUIRED_DOCS[0], "exists": "no"}] + [
            {"doc": d, "exists": "yes"} for d in REQUIRED_DOCS[1:]
        ]
        status, notes = determine_readiness([], [], docs, self._good_ops(), self._good_matrix())
        assert status == "BLOCKED"

    def test_blocked_by_missing_protection(self) -> None:
        bad_matrix: list[dict[str, str]] = [
            {"file": "docs/dino_sentinel_embedding_protocol.md", "exists": "yes",
             "review_only": "absent", "supervised_training": "absent",
             "labels_created": "absent", "predictive_claims": "absent", "multimodal": "absent"}
        ]
        status, notes = determine_readiness([], [], self._good_docs(), self._good_ops(), bad_matrix)
        assert status == "BLOCKED"

    def test_blocked_by_missing_script(self) -> None:
        ops = [{"item": REQUIRED_SCRIPTS[0], "kind": "file", "exists": "no"}]
        status, notes = determine_readiness([], [], self._good_docs(), ops, self._good_matrix())
        assert status == "BLOCKED"


# ---------------------------------------------------------------------------
# collect_versionable_files
# ---------------------------------------------------------------------------

class TestCollectVersionableFiles:
    def test_returns_files_from_expected_dirs(self, tmp_path: Path) -> None:
        make_tree(tmp_path, {
            "scripts/dino/audit.py": "# ok",
            "tests/test_audit.py": "# ok",
            "docs/protocol.md": "# ok",
            "configs/config.yaml": "key: value",
            "README.md": "# readme",
        })
        files = collect_versionable_files(tmp_path)
        rels = {f.relative_to(tmp_path).as_posix() for f in files}
        assert "scripts/dino/audit.py" in rels
        assert "tests/test_audit.py" in rels
        assert "docs/protocol.md" in rels
        assert "configs/config.yaml" in rels
        assert "README.md" in rels

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        make_tree(tmp_path, {
            "scripts/dino/__pycache__/audit.cpython-311.pyc": "",
        })
        files = collect_versionable_files(tmp_path)
        rels = {f.relative_to(tmp_path).as_posix() for f in files}
        assert not any("__pycache__" in r for r in rels)


# ---------------------------------------------------------------------------
# write_csv / write_json
# ---------------------------------------------------------------------------

class TestWriteHelpers:
    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json(path, {"key": "value"})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["key"] == "value"

    def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.csv"
        write_csv(path, [{"a": "1", "b": "2"}], ["a", "b"])
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "a,b"
        assert lines[1] == "1,2"
