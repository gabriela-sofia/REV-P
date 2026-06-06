import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_scanner_lists_candidates_without_autowrite(tmp_path, monkeypatch):
    data, docs, _ = install_all(tmp_path, monkeypatch)
    (tmp_path / "tcc").mkdir()
    (tmp_path / "tcc" / "main.tex").write_text(
        "\\section{Introducao}\n\\section{Metodologia}\n\\section{Resultados}\n",
        encoding="utf-8")
    common.run_manuscript_candidate_scanner(common.parse_args([]))
    rows = read_csv(data / "v2al_manuscript_candidate_registry.csv")
    assert rows
    assert all(r["safe_to_autowrite"] == "false" for r in rows)
    paths = [r["path"] for r in rows]
    assert any(p.endswith("tcc/main.tex") for p in paths)
    assert all(not p.startswith("C:") and ":\\" not in p for p in paths)


def test_scanner_handles_empty_repo(tmp_path, monkeypatch):
    data, docs, _ = install_all(tmp_path, monkeypatch)
    # docs only has v2ak drafts; scanner still produces a registry, never autowrite
    common.run_manuscript_candidate_scanner(common.parse_args([]))
    rows = read_csv(data / "v2al_manuscript_candidate_registry.csv")
    assert all(r["safe_to_autowrite"] == "false" for r in rows)
