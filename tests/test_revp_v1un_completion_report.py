import os

import scripts.protocolo_c.revp_v1un_recife_common as common
from tests.test_revp_v1un_recife_human_review_evidence_consolidator import make_base


def test_completion_report_writes_docs_manifest_and_next_action(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    docs = tmp_path / "docs" / "metodologia_cientifica"
    writing = tmp_path / "docs" / "writing_support" / "protocolo_c"
    configs = tmp_path / "configs" / "protocolo_c"
    docs.mkdir(parents=True)
    writing.mkdir(parents=True)
    configs.mkdir(parents=True)
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "WRITING_DIR", str(writing))
    monkeypatch.setattr(common, "CONFIG_DIR", str(configs))
    monkeypatch.setattr(common, "V1UN_ARTIFACTS", [])
    common.run_human_review_evidence_consolidator(str(data / "v1un_recife_human_review_evidence_consolidation.csv"))
    common.run_evidence_strength_classifier(str(data / "v1un_recife_evidence_strength_registry.csv"))
    common.run_safe_claims_generator(str(data / "v1un_recife_safe_claims_registry.csv"), str(data / "v1un_recife_prohibited_claims_registry.csv"))
    common.run_limitations_matrix_builder(str(data / "v1un_recife_limitations_matrix.csv"))
    common.run_tcc_text_evidence_exporter(str(data / "v1un_recife_tcc_evidence_export_registry.csv"), str(writing / "v1un_recife_tcc_paragraphs.md"))
    common.run_protocol_c_status_updater(str(data / "v1un_recife_protocol_c_status_registry.csv"))
    result = common.run_completion_report()
    assert result["next_action"] == "v1uo - Protocolo C Recife Scientific Writing Integration"
    assert os.path.exists(docs / "protocolo_c_status_atual_v1un.md")
    assert os.path.exists(data / "v1un_versionable_artifacts_manifest.csv")
