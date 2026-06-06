import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_orientador_packet_has_mandatory_questions(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    rows = common.run_orientador_review_packet_builder(common.parse_args([]))
    questions = [r["question"] for r in rows if r["category"] == "pergunta"]
    assert len(questions) == 5
    joined = " ".join(questions).lower()
    assert "metodologia, resultado ou discussao" in joined
    assert "corpo ou apendice" in joined
    assert "ground truth operacional" in joined
    md = (integration / "v2al_orientador_review_packet.md").read_text(encoding="utf-8")
    assert "Perguntas objetivas" in md
    common.assert_safe_manuscript_language(md)
