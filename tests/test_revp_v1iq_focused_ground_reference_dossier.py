"""
test_revp_v1iq_focused_ground_reference_dossier.py

Testes para v1iq -- Focused Ground Reference Dossier for PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA

Expectativa correta: PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA NÃO é GROUND_REFERENCE_CANDIDATE.
Resultado esperado: STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK.
Bloqueio: feições de deslizamento cumulativas sem data específica (SIG histórico 2013-2015).
"""

import pytest
import json
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1iq_focused_ground_reference_dossier import (
    FocusedGroundReferenceDossierBuilder,
    CicatrizAreaDossier,
    GateResult,
)


class TestV1IQBasics:
    """Testes básicos de v1iq."""

    def test_script_exists(self):
        """Script deve existir."""
        script_path = SCRIPTS_DIR / "revp_v1iq_focused_ground_reference_dossier.py"
        assert script_path.exists(), "Script v1iq não encontrado"

    def test_builder_instantiation(self):
        """Builder deve instanciar sem erros."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        assert builder is not None

    def test_builder_run_returns_dict(self):
        """run() deve retornar dict com chaves esperadas."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        result = builder.run()
        assert isinstance(result, dict)
        assert "promotion_decision" in result
        assert "gates_pass" in result
        assert "gates_fail" in result

    def test_builder_run_with_all_flags(self):
        """Builder deve rodar com todos os flags explícitos."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        result = builder.run(
            focus_cicatriz_area=True,
            read_composite_evidence=True,
            read_documentary_evidence=True,
            read_local_metadata=True,
            extract_targeted_text=True,
            emit_dossier=True,
            emit_promotion_decision=True,
        )
        assert result is not None
        assert "promotion_decision" in result


class TestV1IQDossierStructure:
    """Testes para estrutura do dossiê."""

    def test_dossier_created_after_run(self):
        """Dossiê deve ser criado após run()."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.dossier is not None

    def test_dossier_candidate_asset_name(self):
        """Dossiê deve referenciar PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.candidate_asset_name == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"

    def test_dossier_region(self):
        """Dossiê deve ter região PET."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.region == "PET"

    def test_dossier_source_institution(self):
        """Fonte deve ser SGB/CPRM."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.dossier is not None
        assert "SGB" in builder.dossier.source_institution


class TestV1IQGateEvaluation:
    """Testes para avaliação de gates."""

    def test_gates_list_populated(self):
        """Lista de gates deve ter 8 entradas."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert len(builder.gates) == 8

    def test_gate_names_present(self):
        """Todos os gates esperados devem existir."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        gate_names = {g.gate_name for g in builder.gates}
        expected = {
            "gate_geometry", "gate_crs", "gate_observed_status",
            "gate_source_authority", "gate_event_or_survey_date",
            "gate_document_vector_linkage", "gate_region_match",
            "gate_phenomenon_match",
        }
        assert expected.issubset(gate_names), f"Gates faltando: {expected - gate_names}"

    def test_stats_keys_populated(self):
        """stats deve ter chaves esperadas com valores >= 0."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert "gates_pass" in builder.stats
        assert "gates_fail" in builder.stats
        assert "gates_moderate" in builder.stats
        assert builder.stats["gates_pass"] >= 0
        assert builder.stats["gates_fail"] >= 0

    def test_gates_pass_minimum(self):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve passar em pelo menos 4 gates (geometry, observed, region, phenomenon)."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.stats["gates_pass"] >= 4, (
            f"Deve passar em pelo menos 4 gates, passou em {builder.stats['gates_pass']}"
        )

    def test_gate_observed_status_passes(self):
        """gate_observed_status deve ser PASS (feições de deslizamento são observadas, não modeladas)."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        obs_gate = next(g for g in builder.gates if g.gate_name == "gate_observed_status")
        assert obs_gate.status == "PASS", f"gate_observed_status deve ser PASS, got {obs_gate.status}"

    def test_gate_region_match_passes(self):
        """gate_region_match deve ser STRONG."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        region_gate = next(g for g in builder.gates if g.gate_name == "gate_region_match")
        assert region_gate.status == "STRONG", f"gate_region_match deve ser STRONG, got {region_gate.status}"

    def test_gate_phenomenon_match_passes(self):
        """gate_phenomenon_match deve ser STRONG."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        phenom_gate = next(g for g in builder.gates if g.gate_name == "gate_phenomenon_match")
        assert phenom_gate.status == "STRONG", f"gate_phenomenon_match deve ser STRONG, got {phenom_gate.status}"

    def test_gate_event_date_does_not_pass(self):
        """gate_event_or_survey_date NÃO deve ser PASS (feições de deslizamento cumulativas/sem data)."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        date_gate = next(g for g in builder.gates if g.gate_name == "gate_event_or_survey_date")
        assert date_gate.status not in ("PASS", "STRONG"), (
            f"gate_event_or_survey_date NÃO deve ser PASS/STRONG. "
            f"feições de deslizamento são cumulativas, sem data específica. Got: {date_gate.status}"
        )


class TestV1IQPromotion:
    """Testes para decisão de promoção — resultado esperado: bloqueio temporal persiste."""

    def test_promotion_decision_is_valid(self):
        """Decisão de promoção deve ser uma string válida."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.promotion_decision in [
            "GROUND_REFERENCE_CANDIDATE",
            "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK",
            "DOCUMENTED_EVENT_BUT_VECTOR_LINK_INSUFFICIENT",
            "VECTOR_OBSERVED_BUT_EVENT_DATE_LINK_MISSING",
            "CONTEXT_ONLY",
            "NOT_USABLE",
        ]

    def test_cicatriz_area_not_ground_reference_candidate(self):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA NÃO deve virar GROUND_REFERENCE_CANDIDATE (SIG histórico 2013-2015)."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.promotion_decision != "GROUND_REFERENCE_CANDIDATE", (
            "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA NÃO pode ser GROUND_REFERENCE_CANDIDATE: "
            "feições de deslizamento são cumulativas (SIG histórico 2013-2015), "
            "sem campo de data, sem vínculo explícito com 2022-02-15."
        )

    def test_cicatriz_area_is_strong_composite_but_temporal_weak(self):
        """Decisão deve ser STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.promotion_decision == "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK", (
            f"Esperado STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK, "
            f"got {builder.dossier.promotion_decision}"
        )

    def test_can_be_ground_reference_candidate_no(self):
        """can_be_ground_reference_candidate deve ser NO."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.can_be_ground_reference_candidate == "NO", (
            f"can_be_ground_reference_candidate deve ser NO, got {builder.dossier.can_be_ground_reference_candidate}"
        )

    def test_primary_blocker_is_temporal(self):
        """Primary blocker deve ser gate relacionado a data/temporal."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        blocker = builder.dossier.primary_blocker
        assert "event" in blocker or "temporal" in blocker or "date" in blocker or "linkage" in blocker, (
            f"Bloqueio deve ser temporal/event/linkage, got: {blocker}"
        )


class TestV1IQInvariants:
    """Testes para invariantes de v1iq — todos os can_* devem ser NO sempre."""

    def test_can_be_operational_ground_truth_no(self):
        """can_be_operational_ground_truth deve ser NO — invariante absoluto."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.can_be_operational_ground_truth == "NO"

    def test_can_create_training_label_no(self):
        """can_create_training_label deve ser NO — invariante absoluto."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.can_create_training_label == "NO"

    def test_can_train_model_no(self):
        """can_train_model deve ser NO — invariante absoluto."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.can_train_model == "NO"

    def test_can_reopen_protocol_b_no(self):
        """can_reopen_protocol_b deve ser NO — invariante absoluto."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        assert builder.dossier is not None
        assert builder.dossier.can_reopen_protocol_b == "NO"


class TestV1IQOutputs:
    """Testes para geração de outputs."""

    def test_local_runs_dir_created(self):
        """local_runs/protocolo_c/v1iq/ deve existir após run."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq"
        assert local_runs_dir.exists(), "local_runs/protocolo_c/v1iq deve existir"

    def test_local_outputs_generated(self):
        """Arquivos de output locais devem ser gerados."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        local_runs_dir = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq"
        expected_files = [
            "v1iq_cicatriz_area_dossier_inventory.csv",
            "v1iq_summary.json",
        ]
        for expected_file in expected_files:
            file_path = local_runs_dir / expected_file
            assert file_path.exists(), f"{expected_file} não foi gerado"

    def test_summary_json_promotion_decision(self):
        """v1iq_summary.json deve ter promotion_decision correto."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        summary_path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_summary.json"
        assert summary_path.exists()
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["promotion_decision"] == "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"
        assert data.get("can_create_training_label") == False
        assert data.get("can_train_model") == False
        assert data.get("can_reopen_protocol_b") == False

    def test_public_registries_created(self):
        """Registries públicos devem ser criados em datasets/."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        datasets_dir = REPO_ROOT / "datasets"
        dossier_path = datasets_dir / "cicatriz_area_ground_reference_dossier.csv"
        assert dossier_path.exists(), "cicatriz_area_ground_reference_dossier.csv deve ser criado"
        matrix_path = datasets_dir / "cicatriz_area_ground_reference_gate_matrix.csv"
        assert matrix_path.exists(), "cicatriz_area_ground_reference_gate_matrix.csv deve ser criada"

    def test_dossier_registry_has_cicatriz(self):
        """Registry deve conter PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        registry_path = REPO_ROOT / "datasets" / "cicatriz_area_ground_reference_dossier.csv"
        with open(registry_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Registry deve ter pelo menos 1 registro"
        assert "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in rows[0].get("candidate_asset_name", "")

    def test_dossier_registry_promotion_decision(self):
        """Registry deve ter promotion_decision correto."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        registry_path = REPO_ROOT / "datasets" / "cicatriz_area_ground_reference_dossier.csv"
        with open(registry_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["promotion_decision"] == "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"


class TestV1IQNoPrivatePaths:
    """Testes para garantir que paths privados não aparecem em arquivos públicos."""

    def test_no_private_paths_in_registry(self):
        """Registry não deve conter paths privados (C:\\ ou username)."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        registry_path = REPO_ROOT / "datasets" / "cicatriz_area_ground_reference_dossier.csv"
        with open(registry_path, encoding="utf-8") as f:
            content = f.read()
        assert "C:\\" not in content and "C:/" not in content, (
            "Path absoluto não deve aparecer no registry"
        )
        assert "gabriela" not in content.lower(), (
            "Nome de usuário não deve aparecer no registry"
        )

    def test_no_private_paths_in_gate_matrix(self):
        """Gate matrix não deve conter paths privados."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        matrix_path = REPO_ROOT / "datasets" / "cicatriz_area_ground_reference_gate_matrix.csv"
        with open(matrix_path, encoding="utf-8") as f:
            content = f.read()
        assert "C:\\" not in content and "C:/" not in content, (
            "Path absoluto não deve aparecer na gate matrix"
        )
        assert "gabriela" not in content.lower(), (
            "Nome de usuário não deve aparecer na gate matrix"
        )


class TestV1IQStats:
    """Testes para estatísticas."""

    def test_stats_dossier_complete(self):
        """stats['dossier_complete'] deve ser True após run."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.stats["dossier_complete"] == True

    def test_stats_promotion_decision_filled(self):
        """stats['promotion_decision'] deve ser preenchido (não UNKNOWN)."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.stats["promotion_decision"] != "UNKNOWN"

    def test_stats_promotion_matches_dossier(self):
        """stats e dossier devem ter a mesma decisão de promoção."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.dossier is not None
        assert builder.stats["promotion_decision"] == builder.dossier.promotion_decision


class TestV1IQDBFValueAudit:
    """Testes para auditoria de valores reais do DBF (v1iq-R2)."""

    def test_dbf_value_audit_csv_created(self):
        """v1iq_cicatriz_area_dbf_value_audit.csv deve ser criado se DBF acessível."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente — skip de teste de valor")
        audit_csv = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_cicatriz_area_dbf_value_audit.csv"
        assert audit_csv.exists(), "v1iq_cicatriz_area_dbf_value_audit.csv deve ser criado"

    def test_attribute_summary_json_created(self):
        """v1iq_cicatriz_area_attribute_summary.json deve ser criado se DBF acessível."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        summary_json = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_cicatriz_area_attribute_summary.json"
        assert summary_json.exists(), "v1iq_cicatriz_area_attribute_summary.json deve ser criado"

    def test_source_terms_csv_created(self):
        """v1iq_cicatriz_area_source_terms.csv deve ser criado se DBF acessível."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        src_csv = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_cicatriz_area_source_terms.csv"
        assert src_csv.exists(), "v1iq_cicatriz_area_source_terms.csv deve ser criado"

    def test_provenance_decision_csv_created(self):
        """v1iq_cicatriz_area_attribute_provenance_decision.csv deve ser criado se DBF acessível."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        prov_csv = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_cicatriz_area_attribute_provenance_decision.csv"
        assert prov_csv.exists(), "v1iq_cicatriz_area_attribute_provenance_decision.csv deve ser criado"

    def test_total_records_is_444_or_audited(self):
        """Se DBF lido, total_records deve ser 444 (valor auditado localmente)."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        total = builder.dbf_audit.get("total_records", 0)
        assert total == 444, (
            f"PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA deve ter 444 feições (auditado), got {total}"
        )

    def test_expected_fields_in_audit(self):
        """Campos MUNICIPIO, UF, TIPO, FONTE, OBS devem aparecer na auditoria."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        audited_fields = [f["name"] for f in builder.dbf_audit.get("fields", [])]
        for expected in ["MUNICIPIO", "UF", "TIPO", "FONTE", "OBS"]:
            assert expected in audited_fields, (
                f"Campo {expected} deve estar presente na auditoria; encontrados: {audited_fields}"
            )

    def test_attr_prov_is_set(self):
        """self.attr_prov deve ser preenchido após run com focus_cicatriz_area=True."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run(focus_cicatriz_area=True)
        assert builder.attr_prov is not None, "attr_prov deve ser preenchido"

    def test_attr_prov_candidate_name(self):
        """attr_prov.candidate_asset_name deve ser PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.attr_prov is not None
        assert builder.attr_prov.candidate_asset_name == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"

    def test_attr_prov_promotion_valid_value(self):
        """attr_prov.promotion_decision_after_attribute_audit deve ser valor válido."""
        builder = FocusedGroundReferenceDossierBuilder(force=False)
        builder.run()
        assert builder.attr_prov is not None
        valid = {
            "GROUND_REFERENCE_CANDIDATE",
            "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK",
            "DOCUMENTED_EVENT_BUT_VECTOR_LINK_INSUFFICIENT",
        }
        assert builder.attr_prov.promotion_decision_after_attribute_audit in valid, (
            f"Valor inválido: {builder.attr_prov.promotion_decision_after_attribute_audit}"
        )

    def test_attr_prov_no_event_date_for_historical_sig(self):
        """SIG 2013-2015: has_event_or_survey_date_in_field deve ser NO se DBF acessível."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        assert builder.attr_prov is not None
        assert builder.attr_prov.has_event_or_survey_date_in_field == "NO", (
            "SIG histórico 2013-2015 não deve ter data de evento de 2022 nos atributos"
        )

    def test_attribute_summary_json_no_private_paths(self):
        """attribute_summary.json não deve conter paths privados."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        summary_path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_cicatriz_area_attribute_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            content = f.read()
        assert "C:\\" not in content and "C:/" not in content, (
            "attribute_summary.json não deve conter paths absolutos"
        )
        assert "gabriela" not in content.lower(), (
            "attribute_summary.json não deve conter nome de usuário"
        )

    def test_provenance_decision_no_private_paths(self):
        """provenance_decision.csv não deve conter paths privados."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        prov_path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_cicatriz_area_attribute_provenance_decision.csv"
        with open(prov_path, encoding="utf-8") as f:
            content = f.read()
        assert "C:\\" not in content and "C:/" not in content
        assert "gabriela" not in content.lower()

    def test_no_system_date_as_evidence(self):
        """Data de sistema de arquivos nunca deve ser usada como evidência temporal."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        # A data de evento deve vir apenas dos campos DBF, não do sistema
        # gate_event_or_survey_date só pode passar se atributos ou campo de data existirem
        if builder.dbf_audit.get("dbf_read"):
            date_gate = next(
                (g for g in builder.gates if g.gate_name == "gate_event_or_survey_date"), None
            )
            if date_gate and date_gate.status == "PASS":
                # Só aceitável se atributo confirmar
                assert "attr_event_date=True" in date_gate.evidence_detail or \
                       "has_date_field=True" in date_gate.evidence_detail, (
                    "gate_event_or_survey_date=PASS só é válido com evidência de atributo ou campo de data"
                )

    def test_attribute_evidence_strength_not_strong_for_historical_sig(self):
        """SIG histórico 2013-2015: attribute_evidence_strength deve ser MODERATE ou inferior."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        assert builder.attr_prov is not None
        strength = builder.attr_prov.attribute_evidence_strength
        assert strength in ("MODERATE", "WEAK", "NONE"), (
            f"SIG histórico sem data de evento não deve ter attribute_evidence_strength=STRONG; got {strength}"
        )

    def test_summary_json_has_attribute_audit_section(self):
        """v1iq_summary.json deve ter seção attribute_audit_R2."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        summary_path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "attribute_audit_R2" in data, "summary.json deve ter seção attribute_audit_R2"
        assert "stage" in data
        assert data["stage"] == "v1iq_R2"

    def test_ground_reference_candidate_requires_event_date_in_attributes(self):
        """GROUND_REFERENCE_CANDIDATE só se has_event_or_survey_date_in_field=YES."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        if not builder.dbf_audit.get("dbf_read"):
            pytest.skip("DBF não acessível neste ambiente")
        assert builder.attr_prov is not None
        if builder.attr_prov.promotion_decision_after_attribute_audit == "GROUND_REFERENCE_CANDIDATE":
            assert builder.attr_prov.has_event_or_survey_date_in_field == "YES", (
                "Promoção para GROUND_REFERENCE_CANDIDATE requer has_event_or_survey_date_in_field=YES"
            )

    def test_crs_epsg31983_in_summary(self):
        """CRS EPSG:31983 deve aparecer no summary JSON."""
        builder = FocusedGroundReferenceDossierBuilder(force=True)
        builder.run()
        summary_path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iq" / "v1iq_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            content = f.read()
        assert "SIRGAS_2000_UTM_Zone_23S" in content or "31983" in content or "Zone_23S" in content, (
            "CRS EPSG:31983/Zone_23S deve aparecer no summary JSON"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
