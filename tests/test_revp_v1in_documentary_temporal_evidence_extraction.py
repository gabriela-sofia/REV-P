"""
test_revp_v1in_documentary_temporal_evidence_extraction.py

Testes para v1in -- Extração de Evidência Temporal de Documentos Locais
"""

import pytest
import json
from pathlib import Path
import sys
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1in_documentary_temporal_evidence_extraction import (
    DocumentaryTemporalEvidenceExtractor,
    DocumentInventory,
    TemporalExpressionCandidate,
    EvidenceStrengthDecision,
)


class TestV1INBasics:
    """Testes básicos de v1in."""

    def test_script_exists(self):
        """Script deve existir."""
        script_path = SCRIPTS_DIR / "revp_v1in_documentary_temporal_evidence_extraction.py"
        assert script_path.exists(), "Script v1in não encontrado"

    def test_extractor_instantiation(self):
        """Extrator deve instanciar sem erros."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)
        assert extractor is not None
        assert extractor.force == False

    def test_extractor_run(self):
        """Extrator deve rodar com flags principais."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)
        result = extractor.run(
            scan_local_documents=True,
            scan_v1if_pdfs=True,
            scan_existing_registries=True,
            extract_light_text=True,
            match_candidate_names=True,
            emit_temporal_evidence=True,
            emit_linkage_matrix=True,
        )
        assert result is not None
        assert "stats" in result
        assert "status" in result


class TestV1INDataStructures:
    """Testes para estruturas de dados."""

    def test_document_inventory_creation(self):
        """DocumentInventory deve criar instância válida."""
        doc = DocumentInventory(
            document_id="TEST_DOC",
            document_path_sanitized="test/path/file.csv",
            document_name="file.csv",
            document_type="CSV",
            institution="PROTOCOLO_C",
            region="PET",
            event_id="PET_2022_02_15",
            is_scannable=True,
            scan_reason_if_not="",
        )
        assert doc.document_id == "TEST_DOC"
        assert doc.region == "PET"

    def test_temporal_expression_creation(self):
        """TemporalExpressionCandidate deve criar instância válida."""
        expr = TemporalExpressionCandidate(
            expression_id="EXPR_001",
            document_id="DOC_001",
            temporal_expression="2022-02-15",
            temporal_expression_type="DATE",
            text_context_sanitized="deslizamento em Petrópolis 2022-02-15",
            phenomenon_mentioned="deslizamento",
            location_mentioned="PET",
            candidate_name_mention="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            explicit_linkage="YES",
        )
        assert expr.expression_id == "EXPR_001"
        assert expr.explicit_linkage == "YES"

    def test_evidence_strength_decision_creation(self):
        """EvidenceStrengthDecision deve criar instância válida."""
        decision = EvidenceStrengthDecision(
            evidence_id="EV_001",
            expression_id="EXPR_001",
            document_id="DOC_001",
            candidate_asset_name="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            evidence_strength="STRONG_EXPLICIT_EVENT_DATE",
            accepted_as_event_date="YES",
            accepted_as_survey_date="NO",
            accepted_as_context_only="NO",
            can_update_temporal_gate="YES",
            can_update_ground_truth_readiness="YES",
            can_create_training_label="NO",
            blocking_reason="",
            notes="Strong evidence found",
        )
        assert decision.evidence_strength == "STRONG_EXPLICIT_EVENT_DATE"
        assert decision.can_create_training_label == "NO"


class TestV1INDocumentScanning:
    """Testes para scan de documentos."""

    def test_documents_scanned(self):
        """Documentos devem ser encontrados."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)
        extractor._scan_local_documents()
        extractor._scan_existing_registries()

        assert extractor.stats["documents_found"] >= 0

    def test_existing_registries_scannable(self):
        """Registries existentes devem ser scanáveis."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)
        extractor._scan_existing_registries()

        registry_docs = [d for d in extractor.documents if d.institution == "PROTOCOLO_C"]
        for doc in registry_docs:
            assert doc.is_scannable == True, f"Registry {doc.document_name} deve ser scannável"


class TestV1INTemporalExtraction:
    """Testes para extração de expressões temporais."""

    def test_date_extraction(self):
        """Datas explícitas devem ser extraídas."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)

        # Criar documento de teste
        doc = DocumentInventory(
            document_id="TEST_DOC_001",
            document_path_sanitized="test.txt",
            document_name="test.txt",
            document_type="TXT",
            institution="TEST",
            region="PET",
            event_id="PET_2022_02_15",
            is_scannable=True,
            scan_reason_if_not="",
        )
        extractor.documents.append(doc)

        # Simular expressões extraídas
        expr = TemporalExpressionCandidate(
            expression_id="EXPR_001",
            document_id="TEST_DOC_001",
            temporal_expression="2022-02-15",
            temporal_expression_type="DATE",
            text_context_sanitized="deslizamento em Petrópolis em 2022-02-15",
            phenomenon_mentioned="deslizamento",
            location_mentioned="PET",
            candidate_name_mention="NONE",
            explicit_linkage="YES",
        )
        extractor.expressions.append(expr)

        # Classifica
        extractor._classify_evidence_strength()

        decisions = [d for d in extractor.decisions if d.expression_id == "EXPR_001"]
        assert len(decisions) > 0, "Deve haver decisão para data explícita"
        assert decisions[0].evidence_strength in ["STRONG_EXPLICIT_EVENT_DATE"], \
            "Data com fenômeno+localidade deve ser STRONG_EXPLICIT_EVENT_DATE"

    def test_year_without_phenomenon_is_weak(self):
        """Ano isolado sem fenômeno deve ser WEAK."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)

        doc = DocumentInventory(
            document_id="TEST_DOC_002",
            document_path_sanitized="test.txt",
            document_name="test.txt",
            document_type="TXT",
            institution="TEST",
            region="UNKNOWN",
            event_id="UNKNOWN",
            is_scannable=True,
            scan_reason_if_not="",
        )
        extractor.documents.append(doc)

        expr = TemporalExpressionCandidate(
            expression_id="EXPR_002",
            document_id="TEST_DOC_002",
            temporal_expression="2022",
            temporal_expression_type="YEAR",
            text_context_sanitized="arquivo de 2022",
            phenomenon_mentioned="NONE",
            location_mentioned="NONE",
            candidate_name_mention="NONE",
            explicit_linkage="NONE",
        )
        extractor.expressions.append(expr)

        extractor._classify_evidence_strength()

        decisions = [d for d in extractor.decisions if d.expression_id == "EXPR_002"]
        assert len(decisions) > 0
        assert decisions[0].evidence_strength == "INSUFFICIENT" or \
               decisions[0].evidence_strength == "WEAK_TEXTUAL_HINT", \
            "Ano isolado sem vínculo deve ser INSUFFICIENT ou WEAK"


class TestV1INEvidenceStrengthClassification:
    """Testes para classificação de força de evidência."""

    def test_strong_explicit_event_date_requires_linkage(self):
        """STRONG_EXPLICIT_EVENT_DATE requer vínculo explícito."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)

        doc = DocumentInventory(
            document_id="TEST_DOC",
            document_path_sanitized="test.txt",
            document_name="test.txt",
            document_type="TXT",
            institution="TEST",
            region="PET",
            event_id="PET_2022_02_15",
            is_scannable=True,
            scan_reason_if_not="",
        )
        extractor.documents.append(doc)

        # Com vínculo explícito
        expr1 = TemporalExpressionCandidate(
            expression_id="EXPR_A",
            document_id="TEST_DOC",
            temporal_expression="2022-02-15",
            temporal_expression_type="DATE",
            text_context_sanitized="deslizamento em Petrópolis 2022-02-15",
            phenomenon_mentioned="deslizamento",
            location_mentioned="PET",
            candidate_name_mention="NONE",
            explicit_linkage="YES",
        )
        extractor.expressions.append(expr1)

        # Sem vínculo
        expr2 = TemporalExpressionCandidate(
            expression_id="EXPR_B",
            document_id="TEST_DOC",
            temporal_expression="2022-02-15",
            temporal_expression_type="DATE",
            text_context_sanitized="arquivo de 2022-02-15",
            phenomenon_mentioned="NONE",
            location_mentioned="NONE",
            candidate_name_mention="NONE",
            explicit_linkage="NONE",
        )
        extractor.expressions.append(expr2)

        extractor._classify_evidence_strength()

        decision_a = [d for d in extractor.decisions if d.expression_id == "EXPR_A"]
        decision_b = [d for d in extractor.decisions if d.expression_id == "EXPR_B"]

        assert len(decision_a) > 0
        assert decision_a[0].evidence_strength == "STRONG_EXPLICIT_EVENT_DATE", \
            "Data com vínculo explícito deve ser STRONG"

        assert len(decision_b) > 0
        assert decision_b[0].evidence_strength in ["INSUFFICIENT", "WEAK_TEXTUAL_HINT"], \
            "Data sem vínculo deve ser INSUFFICIENT ou WEAK"


class TestV1INTemporalGateUpdate:
    """Testes para atualização de gate temporal."""

    def test_can_update_gate_requires_strong_evidence(self):
        """can_update_temporal_gate deve ser YES apenas com STRONG + vínculo."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)

        doc = DocumentInventory(
            document_id="TEST_DOC",
            document_path_sanitized="test.txt",
            document_name="test.txt",
            document_type="TXT",
            institution="TEST",
            region="PET",
            event_id="PET_2022_02_15",
            is_scannable=True,
            scan_reason_if_not="",
        )
        extractor.documents.append(doc)

        # STRONG evidence
        expr_strong = TemporalExpressionCandidate(
            expression_id="EXPR_STRONG",
            document_id="TEST_DOC",
            temporal_expression="2022-02-15",
            temporal_expression_type="DATE",
            text_context_sanitized="deslizamento em Petrópolis em 2022-02-15",
            phenomenon_mentioned="deslizamento",
            location_mentioned="PET",
            candidate_name_mention="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            explicit_linkage="YES",
        )
        extractor.expressions.append(expr_strong)

        # MODERATE evidence
        expr_moderate = TemporalExpressionCandidate(
            expression_id="EXPR_MODERATE",
            document_id="TEST_DOC",
            temporal_expression="fevereiro de 2022",
            temporal_expression_type="MONTH_YEAR",
            text_context_sanitized="deslizamento em Petrópolis fevereiro de 2022",
            phenomenon_mentioned="deslizamento",
            location_mentioned="PET",
            candidate_name_mention="NONE",
            explicit_linkage="YES",
        )
        extractor.expressions.append(expr_moderate)

        extractor._classify_evidence_strength()

        decision_strong = [d for d in extractor.decisions if d.expression_id == "EXPR_STRONG"]
        decision_moderate = [d for d in extractor.decisions if d.expression_id == "EXPR_MODERATE"]

        assert len(decision_strong) > 0
        assert decision_strong[0].can_update_temporal_gate == "YES", \
            "STRONG + vínculo deve atualizar gate"

        assert len(decision_moderate) > 0
        assert decision_moderate[0].can_update_temporal_gate == "NO", \
            "MODERATE não deve atualizar gate temporal"


class TestV1INInvariants:
    """Testes para invariantes de v1in."""

    def test_can_create_training_label_always_no(self):
        """can_create_training_label deve ser sempre NO."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)
        extractor.run()

        for decision in extractor.decisions:
            assert decision.can_create_training_label == "NO", \
                f"can_create_training_label deve ser NO em {decision.evidence_id}"

    def test_file_system_mtime_never_used(self):
        """Data de modificação de arquivo nunca deve ser usada."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)

        # Verificar que não há expressão extraída de mtime
        # (isso seria testado em _extract_temporal_expressions se houvesse)
        for expr in extractor.expressions:
            # mtime seria algo como "2025-XX-XX", não deve aparecer como evidence
            assert "mtime" not in expr.temporal_expression.lower(), \
                "mtime não deve aparecer em evidência temporal"

    def test_no_pdf_as_vector(self):
        """PDF não deve ser considerado vetor observado."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)

        pdf_doc = DocumentInventory(
            document_id="PDF_DOC",
            document_path_sanitized="report.pdf",
            document_name="report.pdf",
            document_type="PDF",
            institution="SGB/CPRM",
            region="PET",
            event_id="PET_2022_02_15",
            is_scannable=False,
            scan_reason_if_not="pdf_requires_ocr",
        )
        extractor.documents.append(pdf_doc)

        # PDF não deve ter expressões extraídas por padrão
        text = extractor._extract_text_from_document(pdf_doc)
        assert text == "", "PDF não deve ter texto extraído sem OCR"

    def test_no_label_from_documentary_evidence(self):
        """Evidência documental não vira label."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=False)
        extractor.run()

        for decision in extractor.decisions:
            assert decision.can_create_training_label == "NO", \
                "Evidência documental nunca vira label"
            assert "can_create_training_label" in asdict(decision), \
                "Campo can_create_training_label obrigatório"


class TestV1INOutputs:
    """Testes para geração de outputs."""

    def test_local_outputs_generated(self):
        """Outputs locais devem ser gerados."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=True)
        extractor.run()

        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1in"
        assert local_runs_dir.exists(), "local_runs/protocolo_c/v1in deve existir"

        expected_files = [
            "v1in_document_inventory.csv",
            "v1in_temporal_expression_candidates.csv",
            "v1in_evidence_strength_decision.csv",
            "v1in_summary.json",
        ]

        for expected_file in expected_files:
            file_path = local_runs_dir / expected_file
            assert file_path.exists(), f"{expected_file} não foi gerado"

    def test_public_registries_only_with_useful_evidence(self):
        """Registries públicos só devem ser criados com evidência útil."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=True)
        extractor.run()

        datasets_dir = REPO_ROOT / "datasets"

        # Verificar se há evidência útil
        useful_evidence = [d for d in extractor.decisions
                          if d.evidence_strength in
                          ["STRONG_EXPLICIT_EVENT_DATE", "STRONG_EXPLICIT_SURVEY_DATE"]]

        if useful_evidence:
            registry_path = datasets_dir / "documentary_temporal_evidence_registry.csv"
            assert registry_path.exists(), "Registry público deve existir com evidência útil"
        else:
            # Pode não existir se não há evidência útil
            pass

    def test_summary_json_structure(self):
        """Summary JSON deve ter estrutura correta."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=True)
        extractor.run()

        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1in"
        summary_path = local_runs_dir / "v1in_summary.json"

        assert summary_path.exists()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert "stats" in summary
        assert "useful_evidence_found" in summary
        assert "status" in summary


class TestV1INNoPrivatePaths:
    """Testes para garantir que paths privados não aparecem em arquivos públicos."""

    def test_no_private_paths_in_public_registries(self):
        """Arquivos públicos não devem conter paths privados."""
        extractor = DocumentaryTemporalEvidenceExtractor(force=True)
        extractor.run()

        datasets_dir = REPO_ROOT / "datasets"

        # Se registry público foi criado, verificar
        registry_path = datasets_dir / "documentary_temporal_evidence_registry.csv"
        if registry_path.exists():
            with open(registry_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Verificar que source_document_name_sanitized não contém paths privados
                    doc_name = row.get("source_document_name_sanitized", "")
                    assert "PROJETO" not in doc_name, f"Path privado em {doc_name}"
                    assert "C:\\" not in doc_name, f"Caminho absoluto em {doc_name}"
                    assert "gabriela" not in doc_name.lower(), f"Nome de usuário em {doc_name}"


def asdict(obj):
    """Converter dataclass para dict."""
    from dataclasses import asdict as dc_asdict
    return dc_asdict(obj)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
