import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_defense_bank_has_mandatory_questions_and_safe_answers(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_defense_question_bank_builder(common.parse_args([]))
    questions = " ".join(r["question"].lower() for r in rows)
    for must in ("ground truth", "validam o projeto", "172 candidatos",
                 "dinov2 detecta enchente", "gis virou label", "treinaram o modelo",
                 "protocolo b", "invalida o trabalho", "provaram"):
        assert must in questions
    # every answer must be safe (forbidden phrases only negated/avoided)
    for r in rows:
        common.assert_safe_text(r["short_answer"])
        common.assert_safe_text(r["technical_answer"])
    md = (atlas / "v2am_defense_question_bank.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
