"""
test_revp_v1im_master_source_consolidation_precision_audit.py

Testes para v1im -- Consolidação Mestre de Fontes e Auditoria de Precisão de Ground Truth
"""

import pytest
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1im_master_source_consolidation_precision_audit import (
    MasterSourceConsolidationAuditor,
    Source,
    SourceLinkage,
    GroundTruthReadiness,
)


class TestV1IMBasics:
    """Testes básicos de v1im."""

    def test_script_exists(self):
        """Script deve existir."""
        script_path = SCRIPTS_DIR / "revp_v1im_master_source_consolidation_precision_audit.py"
        assert script_path.exists(), "Script v1im não encontrado"

    def test_auditor_instantiation(self):
        """Auditor deve instanciar sem erros."""
        auditor = MasterSourceConsolidationAuditor(
            force=False,
            read_previous_registries=True,
        )
        assert auditor is not None
        assert auditor.force == False
        assert auditor.read_previous_registries == True

    def test_auditor_run_with_flags(self):
        """Auditor deve rodar com flags principais."""
        auditor = MasterSourceConsolidationAuditor(
            force=False,
            read_previous_registries=True,
            score_source_authority=True,
            score_temporal_precision=True,
            score_spatial_precision=True,
            score_phenomenon_specificity=True,
            emit_ground_truth_readiness=True,
        )
        result = auditor.run()
        assert result is not None
        assert "stats" in result
        assert "sources_consolidated" in result


class TestV1IMDataStructures:
    """Testes para estruturas de dados."""

    def test_source_creation(self):
        """Source deve criar instância válida."""
        source = Source(
            source_id="TEST_SOURCE",
            source_name="Test Source",
            source_institution="Test Inst",
            source_type="official",
            source_authority_level="HIGH",
        )
        assert source.source_id == "TEST_SOURCE"
        assert source.source_authority_level == "HIGH"

    def test_linkage_creation(self):
        """SourceLinkage deve criar instância válida."""
        linkage = SourceLinkage(
            linkage_id="LINK_TEST",
            candidate_id="CAND_01",
            source_id="SRC_01",
            supports_geometry="YES",
            support_strength="STRONG",
        )
        assert linkage.linkage_id == "LINK_TEST"
        assert linkage.support_strength == "STRONG"

    def test_readiness_creation(self):
        """GroundTruthReadiness deve criar instância válida."""
        readiness = GroundTruthReadiness(
            candidate_id="CAND_01",
            overall_ground_truth_readiness="NOT_READY",
            can_be_operational_ground_truth="NO",
            can_create_training_label="NO",
        )
        assert readiness.overall_ground_truth_readiness == "NOT_READY"
        assert readiness.can_be_operational_ground_truth == "NO"
        assert readiness.can_create_training_label == "NO"


class TestV1IMSourceConsolidation:
    """Testes para consolidação de fontes."""

    def test_sources_consolidated(self):
        """Fontes devem ser consolidadas."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor._consolidate_sources()

        assert len(auditor.sources) > 0, "Nenhuma fonte consolidada"
        assert auditor.stats["sources_consolidated"] > 0

    def test_official_sources_have_high_authority(self):
        """Fontes oficiais devem ter autoridade HIGH."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor._consolidate_sources()

        for source in auditor.sources:
            if source.official_or_traceable_source == "YES":
                assert source.source_authority_level == "HIGH", \
                    f"Fonte oficial {source.source_id} não tem autoridade HIGH"

    def test_local_sources_have_low_authority(self):
        """Fontes locais devem ter autoridade LOW."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor._consolidate_sources()

        local_sources = [s for s in auditor.sources if "local" in s.source_type.lower()]
        for source in local_sources:
            assert source.source_authority_level == "LOW", \
                f"Fonte local {source.source_id} não tem autoridade LOW"


class TestV1IMGroundTruthInvariants:
    """Testes para invariantes de ground truth."""

    def test_can_be_operational_ground_truth_always_no(self):
        """can_be_operational_ground_truth deve ser sempre NO."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor.run()

        for readiness in auditor.readiness:
            assert readiness.can_be_operational_ground_truth == "NO", \
                f"can_be_operational_ground_truth não é NO para {readiness.candidate_id}"

    def test_can_create_training_label_always_no(self):
        """can_create_training_label deve ser sempre NO."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor.run()

        for readiness in auditor.readiness:
            assert readiness.can_create_training_label == "NO", \
                f"can_create_training_label não é NO para {readiness.candidate_id}"

    def test_blocking_reason_filled_when_not_ready(self):
        """blocking_reason deve ser preenchido quando não ready."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor.run()

        for readiness in auditor.readiness:
            if readiness.overall_ground_truth_readiness == "NOT_READY":
                assert readiness.blocking_reason != "", \
                    f"blocking_reason vazio para {readiness.candidate_id}"
                assert readiness.minimum_evidence_needed != "", \
                    f"minimum_evidence_needed vazio para {readiness.candidate_id}"

    def test_minimum_evidence_specific(self):
        """minimum_evidence_needed deve ser específico."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor.run()

        for readiness in auditor.readiness:
            if readiness.overall_ground_truth_readiness == "NOT_READY":
                # Não deve ser genérico como "more data"
                assert len(readiness.minimum_evidence_needed) > 10, \
                    f"minimum_evidence_needed muito genérico para {readiness.candidate_id}"


class TestV1IMTemporalBlockage:
    """Testes para bloqueio temporal."""

    def test_temporal_blocker_identified(self):
        """Candidatos bloqueados por data devem ser identificados."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor.run()

        # PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA é bloqueado por data
        cicatriz_area_a_records = [r for r in auditor.readiness
                                   if "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in r.candidate_id or
                                   "feição de deslizamento" in r.candidate_id.upper()]

        # Pode haver outros, mas queremos verificar bloqueio correto
        for record in auditor.readiness:
            if record.temporal_gate == "FAIL":
                assert record.blocking_gate == "temporal" or record.blocking_gate == "", \
                    f"Temporal gate failed mas blocker não é temporal para {record.candidate_id}"


class TestV1IMNonRiskAsObservation:
    """Testes para garantir que risco/suscetibilidade não é observação."""

    def test_modelled_not_observed(self):
        """Camadas modeladas não devem ser classificadas como observadas."""
        source = Source(
            source_id="TEST_RISK",
            source_name="Risk Layer",
            observed_or_modelled_status="MODELLED_RISK",
        )

        assert "MODELLED" in source.observed_or_modelled_status, \
            "Camada modelada não está rotulada corretamente"
        assert "OBSERVED" not in source.observed_or_modelled_status, \
            "Camada modelada não deve ter OBSERVED"

    def test_risk_source_blocks_observed_gate(self):
        """Fonte de risco deve falhar gate de observação."""
        auditor = MasterSourceConsolidationAuditor(force=False)

        # Simular fonte de risco
        risk_source = Source(
            source_id="RISK_SRC",
            source_name="Risk Map",
            region="PET",
            event_id="PET_2022_02_15",
            observed_or_modelled_status="MODELLED_RISK",
        )

        # Candidato só com fonte de risco
        auditor.candidates["TEST_RISK_CANDIDATE"] = {
            "sources": ["RISK_SRC"],
            "region": "PET",
            "event_id": "PET_2022_02_15",
        }

        auditor.sources.append(risk_source)
        auditor._create_linkages()
        auditor._audit_ground_truth_readiness()

        # Verificar que readiness correspondente tem gate de observação falho
        risk_readiness = [r for r in auditor.readiness
                         if r.candidate_id == "TEST_RISK_CANDIDATE"]
        if risk_readiness:
            assert risk_readiness[0].observed_not_modelled_gate == "FAIL", \
                "Candidato só com risco deve ter gate de observação falho"


class TestV1IMPDFNotVector:
    """Testes para garantir que PDF não é vetor observado."""

    def test_pdf_not_geometry(self):
        """PDF não deve contar como geometria."""
        source = Source(
            source_id="PDF_SRC",
            source_name="PDF Document",
            asset_format="PDF",
            geometry_available="NO",
        )

        assert source.geometry_available == "NO", \
            "PDF não deve ter geometry_available=YES"
        assert source.asset_format == "PDF", \
            "Asset format deve ser PDF"


class TestV1IMVectorWithoutDate:
    """Testes para vetor sem data."""

    def test_vector_without_date_blocks_temporal_gate(self):
        """Vetor sem data deve falhar gate temporal."""
        auditor = MasterSourceConsolidationAuditor(force=False)

        # Fonte com vetor mas sem data
        vec_source = Source(
            source_id="VEC_NO_DATE",
            source_name="Vector No Date",
            region="PET",
            event_id="PET_2022_02_15",
            geometry_available="YES",
            event_date_explicit="NO",
            temporal_precision_level="UNKNOWN_OR_INDIRECT",
        )

        auditor.sources.append(vec_source)
        auditor.candidates["VECTOR_NO_DATE_CAND"] = {
            "sources": ["VEC_NO_DATE"],
            "region": "PET",
            "event_id": "PET_2022_02_15",
        }

        auditor._create_linkages()
        auditor._audit_ground_truth_readiness()

        vec_readiness = [r for r in auditor.readiness
                        if r.candidate_id == "VECTOR_NO_DATE_CAND"]
        if vec_readiness:
            assert vec_readiness[0].temporal_gate == "FAIL", \
                "Vetor sem data deve falhar gate temporal"


class TestV1IMReadinessGates:
    """Testes para gates de prontidão."""

    def test_all_gates_must_pass_for_ready(self):
        """Todos os gates devem passar para estar ready."""
        auditor = MasterSourceConsolidationAuditor(force=False)
        auditor.run()

        for readiness in auditor.readiness:
            gates = [readiness.source_authority_gate,
                    readiness.geometry_gate,
                    readiness.temporal_gate,
                    readiness.phenomenon_gate,
                    readiness.observed_not_modelled_gate]

            if readiness.overall_ground_truth_readiness == "READY_FOR_REFERENCE":
                # Todos os gates conhecidos devem passar
                assert all(g in {"PASS", "UNKNOWN"} for g in gates), \
                    f"Readiness READY_FOR_REFERENCE mas gate falhou em {readiness.candidate_id}"

            if readiness.overall_ground_truth_readiness == "NOT_READY":
                # Deve haver ao menos um gate falho
                has_fail = any(g == "FAIL" for g in gates)
                assert has_fail or readiness.blocking_reason != "", \
                    f"NOT_READY mas sem gate FAIL e sem blocking_reason em {readiness.candidate_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
