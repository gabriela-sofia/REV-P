"""
test_revp_v1ip_composite_ground_reference_evidence_builder.py

Testes para v1ip -- Composite Ground Reference Evidence Builder
"""

import pytest
import json
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1ip_composite_ground_reference_evidence_builder import (
    CompositeGroundReferenceBuilder,
    CompositeCandidate,
)


class TestV1IPBasics:
    """Testes básicos de v1ip."""

    def test_script_exists(self):
        """Script deve existir."""
        script_path = SCRIPTS_DIR / "revp_v1ip_composite_ground_reference_evidence_builder.py"
        assert script_path.exists(), "Script v1ip não encontrado"

    def test_builder_instantiation(self):
        """Builder deve instanciar sem erros."""
        builder = CompositeGroundReferenceBuilder(force=False)
        assert builder is not None
        assert builder.force == False

    def test_builder_run(self):
        """Builder deve rodar com flags principais."""
        builder = CompositeGroundReferenceBuilder(force=False)
        result = builder.run(
            read_vector_candidates=True,
            read_documentary_evidence=True,
            read_temporal_evidence=True,
            read_source_lineage=True,
            focus_best_candidates=True,
            emit_composite_dossier=True,
            emit_reference_decision=True,
        )
        assert result is not None
        assert "stats" in result


class TestV1IPCandidateStructure:
    """Testes para estrutura de candidato composto."""

    def test_composite_candidate_creation(self):
        """CompositeCandidate deve criar instância válida."""
        candidate = CompositeCandidate(
            composite_candidate_id="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            candidate_asset_name="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            source_layer_alias="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            source_layer_display_name="Feições poligonais de deslizamento fotointerpretadas",
            source_layer_original_name="Cicatriz_Area_A.shp",
            region="PET",
            event_id="PET_2022_02_15",
            source_institution="SGB/CPRM",
            source_document_name_sanitized="SGB_CPRM_PETRÓPOLIS_2022_PDF",
            source_asset_name_sanitized="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA",
            geometry_available="YES",
            crs_available="YES",
            phenomenon_available="YES",
            phenomenon_group="movement_of_mass",
            observed_not_modelled_status="OBSERVED",
            event_date_documented="YES",
            survey_date_documented="NO",
            document_vector_package_link="MODERATE",
            source_lineage_match="STRONG",
            region_match="STRONG",
            phenomenon_match="STRONG",
            temporal_link_strength="MODERATE",
            spatial_usability="FEATURE_LEVEL",
            composite_evidence_strength="MODERATE",
            ground_reference_status="STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK",
            can_be_ground_reference_candidate="NO",
            can_be_operational_ground_truth="NO",
            can_create_training_label="NO",
            can_train_model="NO",
            can_reopen_protocol_b="NO",
            primary_blocker="temporal_linkage_document_vector",
            minimum_evidence_needed="Explicit linkage in official document",
        )
        assert candidate.composite_candidate_id == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"
        assert candidate.geometry_available == "YES"


class TestV1IPIdentification:
    """Testes para identificação de candidatos fortes."""

    def test_strong_candidates_identified(self):
        """Candidatos fortes devem ser identificados."""
        builder = CompositeGroundReferenceBuilder(force=False)
        strong = builder._identify_strong_candidates()

        assert len(strong) > 0, "Deve ter identificado candidatos fortes"
        # PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar na lista
        cicatriz_names = [c["candidate_id"] for c in strong]
        assert "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in cicatriz_names, "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar em fortes"


class TestV1IPDossierBuilding:
    """Testes para construção de dossiês compostos."""

    def test_dossiers_built(self):
        """Dossiês devem ser construídos."""
        builder = CompositeGroundReferenceBuilder(force=False)
        strong = builder._identify_strong_candidates()

        for candidate in strong:
            builder._build_composite_dossier(candidate)

        assert len(builder.candidates) > 0, "Deve ter construído dossiês"
        assert builder.stats["candidates_evaluated"] > 0


class TestV1IPInvariants:
    """Testes para invariantes de v1ip."""

    def test_can_create_training_label_false(self):
        """can_create_training_label deve ser sempre NO."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        for candidate in builder.candidates:
            assert candidate.can_create_training_label == "NO", \
                f"can_create_training_label deve ser NO para {candidate.composite_candidate_id}"

    def test_can_train_model_false(self):
        """can_train_model deve ser sempre NO."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        for candidate in builder.candidates:
            assert candidate.can_train_model == "NO", \
                f"can_train_model deve ser NO para {candidate.composite_candidate_id}"

    def test_can_be_operational_ground_truth_false(self):
        """can_be_operational_ground_truth deve ser sempre NO."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        for candidate in builder.candidates:
            assert candidate.can_be_operational_ground_truth == "NO", \
                f"can_be_operational_ground_truth deve ser NO para {candidate.composite_candidate_id}"

    def test_can_reopen_protocol_b_false(self):
        """can_reopen_protocol_b deve ser sempre NO."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        for candidate in builder.candidates:
            assert candidate.can_reopen_protocol_b == "NO", \
                f"can_reopen_protocol_b deve ser NO para {candidate.composite_candidate_id}"


class TestV1IPCicatrizAreaA:
    """Testes específicos para PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""

    def test_cicatriz_area_a_in_registry(self):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar no registry."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        cicatriz_records = [c for c in builder.candidates if "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in c.composite_candidate_id]
        assert len(cicatriz_records) > 0, "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar nos candidatos"

    def test_cicatriz_area_a_has_composite_status(self):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve ter status composto explícito."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        feature_records = [c for c in builder.candidates if c.composite_candidate_id == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"]
        assert len(feature_records) > 0
        assert feature_records[0].ground_reference_status in [
            "GROUND_REFERENCE_CANDIDATE",
            "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK",
            "VECTOR_OBSERVED_BUT_EVENT_LINK_INSUFFICIENT",
            "DOCUMENTED_EVENT_BUT_VECTOR_LINK_INSUFFICIENT",
            "CONTEXT_ONLY",
            "NOT_USABLE",
        ], "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve ter status válido"

    def test_cicatriz_area_a_is_strong_composite_but_temporal_weak(self):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve ser STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        feature_records = [c for c in builder.candidates if c.composite_candidate_id == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"]
        assert len(feature_records) > 0
        # v1ip deve concluir que tem evidência forte mas linkage temporal é fraco
        assert feature_records[0].ground_reference_status == "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK", \
            "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve ter status de evidência composta forte mas linkage temporal fraco"


class TestV1IPDocumentVectorLinkage:
    """Testes para linkage documento-vetor."""

    def test_data_isolated_insufficient(self):
        """Data isolada não deve ser suficiente para GROUND_REFERENCE_CANDIDATE."""
        builder = CompositeGroundReferenceBuilder(force=False)

        # Se houvesse apenas data de documento sem vínculo claro com vetor
        # não deveria ser GROUND_REFERENCE_CANDIDATE
        for candidate in builder._identify_strong_candidates():
            builder._build_composite_dossier(candidate)

        # Todos os candidatos têm temporal linkage fraco ou nenhum
        for candidate in builder.candidates:
            if candidate.temporal_link_strength != "STRONG":
                assert candidate.can_be_ground_reference_candidate == "NO", \
                    "Sem temporal linkage forte, não pode ser ground reference candidate"

    def test_vector_isolated_insufficient(self):
        """Vetor isolado sem documento não deve ser suficiente."""
        # PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA tem geometria e CRS, mas sem documento explícito linkando data
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        feature_records = [c for c in builder.candidates if c.composite_candidate_id == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"]
        if len(feature_records) > 0:
            # Se document_vector_package_link não é STRONG, então não é GROUND_REFERENCE_CANDIDATE
            if feature_records[0].document_vector_package_link != "STRONG":
                assert feature_records[0].can_be_ground_reference_candidate == "NO"


class TestV1IPOutputs:
    """Testes para geração de outputs."""

    def test_local_outputs_generated(self):
        """Outputs locais devem ser gerados."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ip"
        assert local_runs_dir.exists(), "local_runs/protocolo_c/v1ip deve existir"

        expected_files = [
            "v1ip_composite_candidate_inventory.csv",
            "v1ip_summary.json",
            "v1ip_candidate_dossiers.json",
        ]

        for expected_file in expected_files:
            file_path = local_runs_dir / expected_file
            assert file_path.exists(), f"{expected_file} não foi gerado"

    def test_public_registries_created(self):
        """Registries públicos devem ser criados."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        datasets_dir = REPO_ROOT / "datasets"

        registry_path = datasets_dir / "composite_ground_reference_candidate_registry.csv"
        assert registry_path.exists(), "Registry de candidatos deve ser criado"

        matrix_path = datasets_dir / "composite_ground_reference_gate_matrix.csv"
        assert matrix_path.exists(), "Matriz de gates deve ser criada"

    def test_registry_has_cicatriz_area_a(self):
        """Registry deve conter PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        datasets_dir = REPO_ROOT / "datasets"
        registry_path = datasets_dir / "composite_ground_reference_candidate_registry.csv"

        candidate_ids = []
        with open(registry_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidate_ids.append(row.get("composite_candidate_id"))

        assert "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in candidate_ids, "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve estar no registry"


class TestV1IPNoPrivatePaths:
    """Testes para garantir que paths privados não aparecem."""

    def test_no_private_paths_in_public_registry(self):
        """Registry público não deve conter paths privados."""
        builder = CompositeGroundReferenceBuilder(force=True)
        builder.run()

        datasets_dir = REPO_ROOT / "datasets"
        registry_path = datasets_dir / "composite_ground_reference_candidate_registry.csv"

        with open(registry_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "C:\\" not in content and "C:/" not in content, "Path absoluto não deve aparecer"
            assert "gabriela" not in content.lower(), "Nome de usuário não deve aparecer"


class TestV1IPStats:
    """Testes para estatísticas."""

    def test_stats_calculated(self):
        """Estatísticas devem ser calculadas."""
        builder = CompositeGroundReferenceBuilder(force=False)
        builder.run()

        assert builder.stats["candidates_evaluated"] >= 1, "Deve ter avaliado candidatos"
        assert builder.stats["ground_reference_candidates"] >= 0, "Stats completadas"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
