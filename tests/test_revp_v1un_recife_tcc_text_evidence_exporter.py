import scripts.protocolo_c.revp_v1un_recife_common as common


def test_tcc_paragraphs_exist_and_avoid_operational_language(tmp_path):
    out = tmp_path / "tcc.csv"
    doc = tmp_path / "v1un_recife_tcc_paragraphs.md"
    rows = common.run_tcc_text_evidence_exporter(str(out), str(doc))
    text = doc.read_text(encoding="utf-8").lower()
    assert len(rows) == 6
    assert "patch positivo" not in text
    assert "patch negativo" not in text
    assert "inundacao validada no patch" not in text
    assert "treino supervisionado liberado" not in text
    assert "human review" in text.lower()
