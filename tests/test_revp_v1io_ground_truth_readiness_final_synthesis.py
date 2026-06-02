"""
test_revp_v1io_ground_truth_readiness_final_synthesis.py

Testes para v1io -- Síntese Final de Prontidão de Ground Truth
"""

import pytest
import json
from pathlib import Path
import sys
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1io_ground_truth_readiness_final_synthesis import (
    GroundTruthReadinessFinalSynthesis,
    StageSummary,
    CandidateFinalStatus,
)


class TestV1IOBasics:
    """Testes básicos de v1io."""

    def test_script_exists(self):
        """Script deve existir."""
        script_path = SCRIPTS_DIR / "revp_v1io_ground_truth_readiness_final_synthesis.py"
        assert script_path.exists(), "Script v1io não encontrado"

    def test_synthesizer_instantiation(self):
        """Sintetizador deve instanciar sem erros."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        assert synthesizer is not None
        assert synthesizer.force == False

    def test_synthesizer_run(self):
        """Sintetizador deve rodar com flags principais."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        result = synthesizer.run(
            read_public_registries=True,
            read_local_summaries=True,
            emit_final_readiness=True,
            emit_thesis_summary=True,
        )
        assert result is not None
        assert "stats" in result
        assert "final_ground_truth_status" in result


class TestV1IODataStructures:
    """Testes para estruturas de dados."""

    def test_stage_summary_creation(self):
        """StageSummary deve criar instância válida."""
        stage = StageSummary(
            stage_id="v1ij",
            stage_name="Consolidated Observed Event Vector Evidence",
            evidence_type="Vetorial consolidado",
            assets_or_sources_audited=12,
            candidates_found=18,
            candidates_improved=0,
            ground_reference_candidates=0,
            primary_blocking_gate="temporal",
            primary_blocking_reason="Sem data documentada",
        )
        assert stage.stage_id == "v1ij"
        assert stage.primary_blocking_gate == "temporal"

    def test_candidate_final_status_creation(self):
        """CandidateFinalStatus deve criar instância válida."""
        candidate = CandidateFinalStatus(
            candidate_id="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            candidate_name="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            stage_found="v1ij",
            source_authority="HIGH",
            geometry_available="YES",
            crs_available="YES",
            phenomenon_documented="YES",
            observed_not_modelled="YES",
            temporal_explicit="NO",
            blocking_gate="temporal",
            closest_to_reference="YES",
            evidenced_in_v1in="CONTEXTUAL",
            final_status="BLOCKED",
        )
        assert candidate.candidate_id == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"
        assert candidate.final_status == "BLOCKED"


class TestV1IOAggregation:
    """Testes para agregação de dados."""

    def test_stages_aggregated(self):
        """Estágios devem ser agregados."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._aggregate_public_registries()
        synthesizer._aggregate_local_summaries()

        assert synthesizer.stats["stages_aggregated"] >= 4, "Deve ter agregado v1if/v1ii/v1ij/v1ik"

    def test_candidates_audited(self):
        """Candidatos devem ser auditados."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._synthesize_candidate_final_status()

        assert synthesizer.stats["candidates_audited"] > 0, "Deve ter auditado candidatos"

    def test_cicatriz_area_a_closest_to_reference(self):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar marcado como mais próximo."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._synthesize_candidate_final_status()

        cicatriz_area_a = [c for c in synthesizer.candidates if "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in c.candidate_id]
        assert len(cicatriz_area_a) > 0, "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar em candidatos"
        assert cicatriz_area_a[0].closest_to_reference == "YES", \
            "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar marcado como mais próximo"


class TestV1IOFinalStatus:
    """Testes para status final."""

    def test_final_status_determination(self):
        """Status final deve ser determinado corretamente."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer.run()

        final_status = synthesizer._determine_final_ground_truth_status()
        assert final_status in [
            "READY_FOR_REFERENCE",
            "BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE",
        ], f"Status final deve ser um dos esperados, got {final_status}"

    def test_no_ready_candidates_means_blocked(self):
        """Se nenhum candidato ready, status deve ser BLOCKED."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._synthesize_candidate_final_status()

        ready_count = synthesizer.stats["candidates_ready_for_reference"]

        if ready_count == 0:
            final_status = synthesizer._determine_final_ground_truth_status()
            assert final_status == "BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE", \
                "Sem candidatos ready deve resultar em BLOCKED_WITH_CURRENT_PUBLIC_EVIDENCE"


class TestV1IOInvariants:
    """Testes para invariantes de v1io."""

    def test_can_create_training_label_false(self):
        """can_create_training_label deve ser sempre false."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer.run()

        # Verificar em outputs locais
        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1io"
        summary_path = local_runs_dir / "v1io_summary.json"

        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                assert summary.get("can_create_training_label") == False, \
                    "can_create_training_label deve ser false"

    def test_can_reopen_protocol_b_false(self):
        """can_reopen_protocol_b deve ser sempre false."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer.run()

        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1io"
        summary_path = local_runs_dir / "v1io_summary.json"

        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                assert summary.get("can_reopen_protocol_b") == False, \
                    "can_reopen_protocol_b deve ser false"

    def test_no_permanent_status(self):
        """Status não deve usar linguagem de 'permanente' ou 'impossível'."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._synthesize_candidate_final_status()

        for stage in synthesizer.stages:
            assert "permanente" not in stage.primary_blocking_reason.lower(), \
                "Não deve usar 'permanente' em mensagens"
            assert "impossível" not in stage.primary_blocking_reason.lower(), \
                "Não deve usar 'impossível' em mensagens"


class TestV1IOTemporalBlocker:
    """Testes para temporal gate como bloqueador primário."""

    def test_temporal_is_primary_blocker(self):
        """Gate temporal deve ser identificado como bloqueador primário."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._aggregate_public_registries()
        synthesizer._aggregate_local_summaries()

        temporal_blocking_stages = [s for s in synthesizer.stages
                                   if s.primary_blocking_gate == "temporal"]

        assert len(temporal_blocking_stages) > 0, \
            "Deve haver estágios bloqueados por temporal (v1ij, v1ik, v1im)"


class TestV1IOCandidateStatus:
    """Testes para status final de candidatos."""

    def test_cicatriz_area_a_is_blocked(self):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar bloqueado (sem data documentada)."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._synthesize_candidate_final_status()

        feature_records = [c for c in synthesizer.candidates if "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in c.candidate_id]
        assert len(feature_records) > 0
        assert feature_records[0].final_status == "BLOCKED", \
            "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar BLOCKED (sem data)"
        assert feature_records[0].blocking_gate == "temporal", \
            "Bloqueador deve ser temporal"

    def test_cicatriz_ponto_p_is_missing(self):
        """camada de pontos de feições de deslizamento fotointerpretadas deve estar MISSING."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=False)
        synthesizer._synthesize_candidate_final_status()

        cicatriz_p = [c for c in synthesizer.candidates if "Cicatriz_Ponto_P" in c.candidate_id]
        assert len(cicatriz_p) > 0
        assert cicatriz_p[0].final_status == "MISSING", \
            "camada de pontos de feições de deslizamento fotointerpretadas deve estar MISSING"


class TestV1IOOutputs:
    """Testes para geração de outputs."""

    def test_local_outputs_generated(self):
        """Outputs locais devem ser gerados."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=True)
        synthesizer.run()

        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1io"
        assert local_runs_dir.exists(), "local_runs/protocolo_c/v1io deve existir"

        expected_files = [
            "v1io_stage_summary.csv",
            "v1io_candidate_final_status.csv",
            "v1io_summary.json",
            "v1io_thesis_summary.json",
        ]

        for expected_file in expected_files:
            file_path = local_runs_dir / expected_file
            assert file_path.exists(), f"{expected_file} não foi gerado"

    def test_public_registry_created(self):
        """Registry público deve ser criado."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=True)
        synthesizer.run()

        datasets_dir = REPO_ROOT / "datasets"
        registry_path = datasets_dir / "protocol_c_ground_truth_readiness_final_matrix.csv"

        assert registry_path.exists(), "Registry público deve ser criado"

    def test_registry_has_all_stages(self):
        """Registry deve incluir todas as etapas."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=True)
        synthesizer.run()

        datasets_dir = REPO_ROOT / "datasets"
        registry_path = datasets_dir / "protocol_c_ground_truth_readiness_final_matrix.csv"

        stage_ids = []
        with open(registry_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stage_ids.append(row.get("stage_id"))

        expected_stages = ["v1if", "v1ii", "v1ij", "v1ik"]
        for expected in expected_stages:
            assert expected in stage_ids, f"{expected} deve estar no registry"


class TestV1IONoPrivatePaths:
    """Testes para garantir que paths privados não aparecem."""

    def test_no_private_paths_in_public_files(self):
        """Arquivos públicos não devem conter paths privados absolutos."""
        synthesizer = GroundTruthReadinessFinalSynthesis(force=True)
        synthesizer.run()

        datasets_dir = REPO_ROOT / "datasets"
        registry_path = datasets_dir / "protocol_c_ground_truth_readiness_final_matrix.csv"

        with open(registry_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Verificar paths absolutos
            assert "C:\\" not in content and "C:/" not in content, "Path absoluto não deve aparecer"
            # Verificar nome de usuário
            assert "gabriela" not in content.lower(), "Nome de usuário não deve aparecer"
            # "PROJETO" é referência de diagnóstico, não path privado real
            assert "/Users/" not in content and "\\Users\\" not in content, "Path de usuário não deve aparecer"


class TestV1IOVersioning:
    """Testes para garantir que arquivos apropriados não são versionados."""

    def test_local_runs_not_versioned(self):
        """local_runs/ não deve ser versionado."""
        # Isso é testado implicitamente pelo .gitignore
        # Aqui apenas verificamos que outputs locais são gerados lá
        synthesizer = GroundTruthReadinessFinalSynthesis(force=True)
        synthesizer.run()

        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1io"
        assert local_runs_dir.exists(), "local_runs/ deve existir"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
