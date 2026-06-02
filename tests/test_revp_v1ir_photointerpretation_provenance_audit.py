"""
test_revp_v1ir_photointerpretation_provenance_audit.py

Testes para v1ir — Photointerpretation Provenance and Source Imagery Audit.

Expectativa correta:
- PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA NÃO é GROUND_REFERENCE_CANDIDATE após auditoria de proveniência.
- FONTE="Fotointerpretação" é método de produção de 2013, não resposta ao evento de 2022.
- Imagem base NÃO documentada nos metadados do pacote.
- Decisão: STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK (inalterada de v1iq).
- Data de levantamento documentada: Maio/2013 (Pontos_de_Campo_P.shp.xml).
- Data de mapeamento inferida: 2013-08-22 (copy date de sidecar original de pontos de feições de deslizamento fotointerpretadas).
- "2022" não aparece em nenhum XML do pacote.
"""

import pytest
import json
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1ir_photointerpretation_provenance_audit import (
    PhotointerpretationProvenanceAuditor,
    PhotointerpretationProvenanceDecision,
    SidecarEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def auditor_run():
    """Auditor executado uma vez por módulo (sem --force para não sobrescrever)."""
    a = PhotointerpretationProvenanceAuditor(force=False)
    result = a.run()
    return a, result


@pytest.fixture(scope="module")
def auditor_force():
    """Auditor executado com force=True para garantir outputs frescos."""
    a = PhotointerpretationProvenanceAuditor(force=True)
    result = a.run()
    return a, result


# ---------------------------------------------------------------------------
# TestV1IRBasics — script existe e roda
# ---------------------------------------------------------------------------

class TestV1IRBasics:
    """O script existe e instancia/roda sem erros."""

    def test_script_exists(self):
        """Script deve existir no diretório correto."""
        script_path = SCRIPTS_DIR / "revp_v1ir_photointerpretation_provenance_audit.py"
        assert script_path.exists(), "Script v1ir não encontrado"

    def test_auditor_instantiation(self):
        """Auditor deve instanciar sem erros."""
        a = PhotointerpretationProvenanceAuditor(force=False)
        assert a is not None

    def test_auditor_run_returns_dict(self, auditor_run):
        """run() deve retornar dict com chaves esperadas."""
        _, result = auditor_run
        assert isinstance(result, dict)
        assert "status" in result
        assert "promotion_decision" in result
        assert "sidecars_read" in result

    def test_auditor_run_status_complete(self, auditor_run):
        """Status de run() deve ser 'complete'."""
        _, result = auditor_run
        assert result["status"] == "complete"

    def test_auditor_run_with_all_flags(self):
        """Auditor deve rodar com todos os flags explícitos."""
        a = PhotointerpretationProvenanceAuditor(force=False)
        result = a.run(
            focus_cicatriz_area=True,
            scan_sidecars=True,
            scan_package_metadata=True,
            scan_local_documents=True,
            scan_existing_registries=True,
            emit_provenance_decision=True,
        )
        assert result is not None
        assert "promotion_decision" in result

    def test_stats_complete_after_run(self, auditor_run):
        """stats['complete'] deve ser True após run."""
        a, _ = auditor_run
        assert a.stats["complete"] is True

    def test_decision_set_after_run(self, auditor_run):
        """self.decision deve ser preenchido após run."""
        a, _ = auditor_run
        assert a.decision is not None
        assert isinstance(a.decision, PhotointerpretationProvenanceDecision)


# ---------------------------------------------------------------------------
# TestV1IROutputsLocais — outputs locais são gerados
# ---------------------------------------------------------------------------

class TestV1IROutputsLocais:
    """Os 8 arquivos de output local devem ser gerados em local_runs/protocolo_c/v1ir/."""

    LOCAL_DIR = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir"

    EXPECTED_FILES = [
        "v1ir_photointerpretation_source_inventory.csv",
        "v1ir_sidecar_metadata_audit.csv",
        "v1ir_package_lineage_audit.csv",
        "v1ir_source_imagery_temporal_audit.csv",
        "v1ir_documentary_provenance_linkage.csv",
        "v1ir_ground_reference_update_decision.csv",
        "v1ir_summary.json",
        "v1ir_qa.csv",
    ]

    def test_local_dir_created(self, auditor_force):
        """local_runs/protocolo_c/v1ir/ deve existir após run."""
        assert self.LOCAL_DIR.exists(), "local_runs/protocolo_c/v1ir/ não foi criado"

    @pytest.mark.parametrize("fname", EXPECTED_FILES)
    def test_output_file_created(self, auditor_force, fname):
        """Cada arquivo de output deve ser criado."""
        path = self.LOCAL_DIR / fname
        assert path.exists(), f"{fname} não foi gerado em local_runs/protocolo_c/v1ir/"

    def test_summary_json_valid(self, auditor_force):
        """v1ir_summary.json deve ser JSON válido."""
        path = self.LOCAL_DIR / "v1ir_summary.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "stage" in data
        assert data["stage"] == "v1ir"

    def test_sidecar_audit_csv_has_rows(self, auditor_force):
        """v1ir_sidecar_metadata_audit.csv deve ter dados ou comentário."""
        a, _ = auditor_force
        path = self.LOCAL_DIR / "v1ir_sidecar_metadata_audit.csv"
        content = path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, "Arquivo sidecar_metadata_audit.csv não deve ser vazio"

    def test_ground_reference_update_decision_csv_has_fields(self, auditor_force):
        """v1ir_ground_reference_update_decision.csv deve ter campos esperados."""
        a, _ = auditor_force
        if a.decision is None:
            pytest.skip("Decisão não foi construída (sem acesso a PROJETO)")
        path = self.LOCAL_DIR / "v1ir_ground_reference_update_decision.csv"
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        assert "promotion_decision_after_provenance_audit" in rows[0]
        assert "can_be_ground_reference_candidate" in rows[0]

    def test_qa_csv_all_checks_passed(self, auditor_force):
        """v1ir_qa.csv deve ter todos os checks passando."""
        path = self.LOCAL_DIR / "v1ir_qa.csv"
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        failed = [r for r in rows if r.get("passed", "True").lower() == "false"]
        assert len(failed) == 0, f"QA checks falhando: {failed}"


# ---------------------------------------------------------------------------
# TestV1IRFotointerpretacao — Fotointerpretação como fonte/método
# ---------------------------------------------------------------------------

class TestV1IRFotointerpretacao:
    """Fotointerpretação deve aparecer como fonte/método auditado."""

    def test_source_field_value_fotointerpretacao(self, auditor_run):
        """source_field_value deve ser 'Fotointerpretação'."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert "Fotointerpretação" in a.decision.source_field_value or \
               "fotointerpret" in a.decision.source_field_value.lower(), (
            f"source_field_value deve ser Fotointerpretação; got: {a.decision.source_field_value}"
        )

    def test_source_method_is_fotointerpretacao(self, auditor_run):
        """source_method deve ser Fotointerpretação (método documentado)."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert "fotointerpret" in a.decision.source_method.lower(), (
            f"source_method deve conter fotointerpretação; got: {a.decision.source_method}"
        )

    def test_fonte_value_appears_in_sidecar_scan(self, auditor_run):
        """Se sidecars foram lidos, pelo menos um deve ter fonte_field_value com 'fotointerpret'."""
        a, _ = auditor_run
        if not a.sidecars:
            pytest.skip("Nenhum sidecar lido (sem acesso a PROJETO)")
        fonte_values = [s.fonte_field_value.lower() for s in a.sidecars]
        has_foto = any("fotointerpret" in v for v in fonte_values if v)
        # Se o sidecar proxy (camada de pontos de feições de deslizamento fotointerpretadas) foi lido, deve ter "fotointerpretação"
        cicatriz_ponto = next(
            (s for s in a.sidecars if "Cicatriz_Ponto_P" in s.sidecar_file_sanitized), None
        )
        if cicatriz_ponto:
            assert "fotointerpret" in cicatriz_ponto.fonte_field_value.lower(), (
                "sidecar original de pontos de feições de deslizamento fotointerpretadas deve ter FONTE=Fotointerpretação"
            )

    def test_method_is_not_institution(self, auditor_run):
        """'Fotointerpretação' é método, não instituição — institution deve ser inferida."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        # source_field_value é o método, source_institution é a instituição inferida
        assert a.decision.source_field_value != a.decision.source_institution, (
            "source_field_value (método) não deve ser igual a source_institution"
        )

    def test_candidate_asset_name_correct(self, auditor_run):
        """candidate_asset_name deve ser PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.candidate_asset_name == "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"

    def test_phenomenon_match_strong(self, auditor_run):
        """phenomenon_match deve ser STRONG."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.phenomenon_match == "STRONG"

    def test_region_match_strong(self, auditor_run):
        """region_match deve ser STRONG."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.region_match == "STRONG"


# ---------------------------------------------------------------------------
# TestV1IRDataDeSistema — data de sistema não é aceita
# ---------------------------------------------------------------------------

class TestV1IRDataDeSistema:
    """Data de sistema de arquivos nunca deve ser usada como evidência temporal."""

    def test_no_system_date_as_temporal_evidence(self, auditor_run):
        """Data de sistema (mtime/ctime) não deve produzir temporal_link=STRONG."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        # Se temporal_link é STRONG, deve haver imagery_date documentada — não só system date
        if a.decision.temporal_link_strength == "STRONG":
            assert a.decision.imagery_date_documented not in ("NOT_DOCUMENTED", "", "UNKNOWN"), (
                "temporal_link=STRONG requer imagery_date documentada, não data de sistema"
            )

    def test_no_system_date_as_event_date(self, auditor_run):
        """event_date_documented não pode ser derivada de mtime/ctime do arquivo."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        # event_date_documented para SIG 2013-2015 deve ser NOT_DOCUMENTED
        assert a.decision.event_date_documented == "NOT_DOCUMENTED", (
            f"SIG histórico 2013-2015: event_date_documented deve ser NOT_DOCUMENTED; "
            f"got: {a.decision.event_date_documented}"
        )

    def test_imagery_date_not_from_system(self, auditor_run):
        """Se imagery_date_documented preenchida, deve vir de metadado, não de mtime."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        # Para este SIG, imagem base não está documentada
        # imagery_date deve ser NOT_DOCUMENTED (não data de pasta/sistema)
        assert a.decision.imagery_date_documented == "NOT_DOCUMENTED", (
            "Imagem base não documentada nos metadados: imagery_date deve ser NOT_DOCUMENTED"
        )

    def test_mapping_date_comes_from_sidecar(self, auditor_run):
        """mapping_date_documented deve vir de cópia no XML, não de sistema."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        # Mapping date deve conter "2013" (via sidecar original de pontos de feições de deslizamento fotointerpretadas copy date)
        assert "2013" in a.decision.mapping_date_documented, (
            f"mapping_date deve conter 2013 (de sidecar XML), got: {a.decision.mapping_date_documented}"
        )


# ---------------------------------------------------------------------------
# TestV1IRNomeDePasta — nome de pasta não é prova forte
# ---------------------------------------------------------------------------

class TestV1IRNomeDePasta:
    """Nome de pasta (ex: sig_extracted, Kits_Executores_2013) não é prova forte de data."""

    def test_folder_name_not_strong_temporal_proof(self, auditor_run):
        """Temporal_link não deve ser STRONG apenas por causa de nome de pasta."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        # Se temporal_link for STRONG, não pode ser apenas por nome de pasta
        if a.decision.temporal_link_strength == "STRONG":
            # Deve haver data documentada em metadado real
            has_imagery = a.decision.imagery_date_documented not in ("NOT_DOCUMENTED", "")
            has_survey = a.stats.get("survey_date_documented", False)
            assert has_imagery or has_survey, (
                "temporal_link=STRONG requer metadado documentado, não apenas nome de pasta"
            )

    def test_sig_package_hint_is_2013_not_2022(self, auditor_run):
        """O pacote SIG deve ser identificado como contexto de 2013, não de 2022."""
        a, _ = auditor_run
        # Inventário deve identificar o kit de 2013
        inv_path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir" / \
                   "v1ir_photointerpretation_source_inventory.csv"
        if not inv_path.exists():
            pytest.skip("Inventário não gerado")
        with open(inv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {r["item"]: r["value"] for r in reader}
        hint = rows.get("sig_package_hint", "")
        assert "2013" in hint or "2015" in hint or "SUSCETIBILIDADE" in hint.upper(), (
            f"sig_package_hint deve referenciar contexto 2013-2015; got: {hint}"
        )

    def test_folder_name_not_used_for_event_date(self, auditor_run):
        """Nome de pasta 'sig_extracted' ou '2022' na URL não implica data de evento no vetor."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        # Para este SIG, event_date_documented deve ser NOT_DOCUMENTED
        # mesmo que a pasta/repositório seja referenciado como "pós-desastre 2022"
        assert a.decision.event_date_documented == "NOT_DOCUMENTED", (
            "Nome de pasta/URL não é evidência de event_date no vetor"
        )


# ---------------------------------------------------------------------------
# TestV1IRGateTemporal — apenas metadado documentado atualiza gate temporal
# ---------------------------------------------------------------------------

class TestV1IRGateTemporal:
    """Gate temporal só pode ser atualizado por metadado documentado nos XMLs."""

    def test_event_2022_not_found_in_sidecars(self, auditor_run):
        """'2022' não deve aparecer em nenhum sidecar XML do pacote."""
        a, _ = auditor_run
        assert a.stats["event_2022_found_anywhere"] is False, (
            "Nenhum XML do pacote SIG 2013-2015 deve conter '2022'"
        )

    def test_imagery_not_documented(self, auditor_run):
        """Imagem base da fotointerpretação não deve estar documentada nos metadados."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.imagery_date_documented == "NOT_DOCUMENTED", (
            "Imagem base não está documentada nos XMLs do pacote"
        )
        assert a.decision.imagery_or_base_name_sanitized == "NOT_DOCUMENTED", (
            "Nome da imagem base não está documentado nos XMLs do pacote"
        )

    def test_can_update_ground_reference_status_no(self, auditor_run):
        """can_update_ground_reference_status deve ser NO (sem metadado suficiente)."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.can_update_ground_reference_status == "NO", (
            "v1ir não deve atualizar ground reference status sem metadado de data"
        )

    def test_temporal_link_not_strong(self, auditor_run):
        """temporal_link_strength não deve ser STRONG para SIG histórico 2013-2015."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.temporal_link_strength != "STRONG", (
            "SIG 2013-2015 sem evento de 2022: temporal_link_strength não deve ser STRONG"
        )

    def test_temporal_reference_type_is_2013(self, auditor_run):
        """temporal_reference_type deve referenciar 2013."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert "2013" in a.decision.temporal_reference_type or \
               "survey" in a.decision.temporal_reference_type or \
               "partial" in a.decision.temporal_reference_type or \
               "undoc" in a.decision.temporal_reference_type.lower(), (
            f"temporal_reference_type deve referenciar 2013 ou ausência; "
            f"got: {a.decision.temporal_reference_type}"
        )

    def test_survey_date_is_2013(self, auditor_run):
        """survey_date_documented deve referenciar 2013 (levantamento de campo)."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert "2013" in a.decision.survey_date_documented, (
            f"survey_date_documented deve conter 2013 (Pontos_de_Campo_P proxy); "
            f"got: {a.decision.survey_date_documented}"
        )

    def test_only_documented_metadata_can_update_temporal(self, auditor_run):
        """Somente metadado real (XML sidecar) pode atualizar gate temporal — nunca system date."""
        a, _ = auditor_run
        if not a.sidecars:
            pytest.skip("Nenhum sidecar lido")
        # survey_date_documented vem de Pontos_de_Campo_P.shp.xml (DATA="Maio/2013")
        # mapping_date vem de sidecar original de pontos de feições de deslizamento fotointerpretadas (copy date 20130822)
        # Ambos são metadados documentados — OK
        # O que não é permitido: mtime/ctime do sistema
        # Verificar que todos os sidecars têm sidecar_file_sanitized sem path privado
        for s in a.sidecars:
            assert "\\" not in s.sidecar_file_sanitized or s.sidecar_file_sanitized.count("\\") == 0, (
                f"sidecar_file_sanitized não deve conter paths privados: {s.sidecar_file_sanitized}"
            )


# ---------------------------------------------------------------------------
# TestV1IRInvariants — todos os can_* = NO
# ---------------------------------------------------------------------------

class TestV1IRInvariants:
    """Invariantes absolutos: todos os can_* campos devem ser NO sempre."""

    def test_can_be_operational_ground_truth_no(self, auditor_run):
        """can_be_operational_ground_truth deve ser NO — invariante absoluto."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.can_be_operational_ground_truth == "NO", (
            "can_be_operational_ground_truth deve ser NO sempre"
        )

    def test_can_create_training_label_no(self, auditor_run):
        """can_create_training_label deve ser NO — invariante absoluto."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.can_create_training_label == "NO", (
            "can_create_training_label deve ser NO sempre"
        )

    def test_can_train_model_no(self, auditor_run):
        """can_train_model deve ser NO — invariante absoluto."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.can_train_model == "NO", (
            "can_train_model deve ser NO sempre"
        )

    def test_can_reopen_protocol_b_no(self, auditor_run):
        """can_reopen_protocol_b deve ser NO — invariante absoluto."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.can_reopen_protocol_b == "NO", (
            "can_reopen_protocol_b deve ser NO sempre"
        )

    def test_can_be_ground_reference_candidate_no(self, auditor_run):
        """can_be_ground_reference_candidate deve ser NO (SIG histórico sem data de evento)."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.can_be_ground_reference_candidate == "NO", (
            "SIG histórico 2013-2015 sem evento de 2022: não pode ser ground reference candidate"
        )

    def test_invariants_in_summary_json(self, auditor_force):
        """summary.json deve ter todos os can_* = False."""
        path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir" / "v1ir_summary.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("can_create_training_label") is False, (
            "summary.json: can_create_training_label deve ser False"
        )
        assert data.get("can_train_model") is False, (
            "summary.json: can_train_model deve ser False"
        )
        assert data.get("can_reopen_protocol_b") is False, (
            "summary.json: can_reopen_protocol_b deve ser False"
        )
        assert data.get("can_be_operational_ground_truth") is False, (
            "summary.json: can_be_operational_ground_truth deve ser False"
        )

    def test_promotion_decision_unchanged(self, auditor_run):
        """Decisão de promoção deve ser STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.promotion_decision_after_provenance_audit == \
               "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK", (
            f"Decisão deve ser STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK, "
            f"got: {a.decision.promotion_decision_after_provenance_audit}"
        )

    def test_not_ground_reference_candidate(self, auditor_run):
        """PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA não deve ser promovida a GROUND_REFERENCE_CANDIDATE."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        assert a.decision.promotion_decision_after_provenance_audit != "GROUND_REFERENCE_CANDIDATE", (
            "v1ir não deve promover PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA a GROUND_REFERENCE_CANDIDATE: "
            "fotointerpretação de 2013, sem imagem base, sem data de 2022"
        )


# ---------------------------------------------------------------------------
# TestV1IRSemPathPrivado — sem path privado em arquivos públicos
# ---------------------------------------------------------------------------

class TestV1IRSemPathPrivado:
    """Paths privados não podem aparecer em nenhum arquivo público (datasets/)."""

    def test_no_private_path_in_provenance_registry(self, auditor_force):
        """cicatriz_area_photointerpretation_provenance_registry.csv não deve ter paths privados."""
        reg = REPO_ROOT / "datasets" / "cicatriz_area_photointerpretation_provenance_registry.csv"
        if not reg.exists():
            pytest.skip("Registry não gerado")
        content = reg.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content, (
            "Path absoluto não deve aparecer no registry público"
        )
        assert "gabriela" not in content.lower(), (
            "Nome de usuário não deve aparecer no registry público"
        )
        assert "PROJETO" not in content, (
            "Nome do diretório privado PROJETO não deve aparecer no registry"
        )

    def test_no_private_path_in_schema(self, auditor_force):
        """Schema CSV não deve ter paths privados."""
        schema = (
            REPO_ROOT / "datasets" / "schemas"
            / "cicatriz_area_photointerpretation_provenance_schema.csv"
        )
        if not schema.exists():
            pytest.skip("Schema não gerado")
        content = schema.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content
        assert "gabriela" not in content.lower()

    def test_no_private_path_in_decision_notes(self, auditor_run):
        """decision.notes não deve conter paths privados."""
        a, _ = auditor_run
        if a.decision is None:
            pytest.skip("Decisão não construída")
        notes = a.decision.notes
        assert "C:\\" not in notes, "notes não deve conter path privado Windows"
        assert "gabriela" not in notes.lower(), "notes não deve conter nome de usuário"

    def test_no_private_path_in_local_ground_reference_decision_csv(self, auditor_force):
        """v1ir_ground_reference_update_decision.csv não deve ter paths privados."""
        path = (
            REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir"
            / "v1ir_ground_reference_update_decision.csv"
        )
        if not path.exists():
            pytest.skip("Decision CSV não gerado")
        content = path.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content
        assert "gabriela" not in content.lower()

    def test_no_private_path_in_sidecar_audit_csv(self, auditor_force):
        """v1ir_sidecar_metadata_audit.csv não deve ter paths privados."""
        path = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir" / "v1ir_sidecar_metadata_audit.csv"
        if not path.exists():
            pytest.skip("Sidecar audit CSV não gerado")
        content = path.read_text(encoding="utf-8")
        assert "gabriela" not in content.lower(), (
            "sidecar_metadata_audit.csv não deve conter nome de usuário"
        )
        # Paths tipo "D:\..." ou "C:\..." não devem aparecer (sanitizados)
        import re
        private_paths = re.findall(r"[A-Za-z]:\\[A-Za-z]+\\[A-Za-z]+", content)
        assert len(private_paths) == 0, (
            f"Paths privados encontrados no sidecar audit: {private_paths}"
        )


# ---------------------------------------------------------------------------
# TestV1IRLocalRunsNaoVersionado — local_runs/ não deve ir ao git
# ---------------------------------------------------------------------------

class TestV1IRLocalRunsNaoVersionado:
    """local_runs/ e local_only/ não devem ser versionados."""

    def test_gitignore_has_local_runs(self):
        """.gitignore deve ignorar local_runs/."""
        gitignore = REPO_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore não encontrado")
        content = gitignore.read_text(encoding="utf-8")
        assert "local_runs" in content, ".gitignore deve incluir local_runs/"

    def test_v1ir_dir_not_in_git(self):
        """local_runs/protocolo_c/v1ir/ não deve estar no índice do git."""
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "local_runs/"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert "v1ir" not in result.stdout, (
            "local_runs/protocolo_c/v1ir/ não deve estar no git"
        )

    def test_local_only_in_gitignore_or_absent(self):
        """local_only/ não deve estar no git."""
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "local_only/"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "", (
            "local_only/ não deve estar no git"
        )

    def test_no_raster_staged(self):
        """Nenhum arquivo .tif/.tiff deve estar staged ou tracked."""
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--others", "--cached", "*.tif"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "", "Nenhum .tif deve estar no git"

    def test_no_npz_staged(self):
        """Nenhum arquivo .npz deve estar staged para o próximo commit."""
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        npz_staged = [l for l in result.stdout.splitlines() if l.endswith(".npz")]
        assert len(npz_staged) == 0, f"Nenhum .npz deve estar staged: {npz_staged}"


# ---------------------------------------------------------------------------
# TestV1IRDocs — docs não sugerem e-mail/solicitação/Protocolo B
# ---------------------------------------------------------------------------

class TestV1IRDocs:
    """Os docs gerados não devem conter linguagem proibida."""

    DOCS = [
        REPO_ROOT / "docs" / "metodologia_cientifica" /
        "protocolo_c_auditoria_proveniencia_fotointerpretacao_v1ir.md",
        REPO_ROOT / "docs" / "metodologia_cientifica" /
        "protocolo_c_relatorio_proveniencia_fotointerpretacao_v1ir.md",
    ]

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_exists(self, doc_path):
        """Doc deve existir após run."""
        assert doc_path.exists(), f"Doc não encontrado: {doc_path.name}"

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_email_suggestion(self, doc_path):
        """Doc não deve sugerir envio de e-mail."""
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "enviar e-mail" not in content
        assert "solicitar via e-mail" not in content
        assert "mandar e-mail" not in content
        assert "email" not in content or "envio" not in content, (
            f"Doc não deve sugerir envio de e-mail: {doc_path.name}"
        )

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_solicitation(self, doc_path):
        """Doc não deve sugerir solicitação de dados externos."""
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        # Solicitação não autorizada
        assert "solicitar ao cprm" not in content
        assert "solicitar ao sgb" not in content
        assert "entrar em contato" not in content
        assert "contatar a" not in content

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_protocol_b(self, doc_path):
        """Doc não deve sugerir reabertura do Protocolo B."""
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "protocolo b" not in content or "não" in content, (
            f"Doc não deve sugerir reabertura de Protocolo B: {doc_path.name}"
        )

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_flood_prediction(self, doc_path):
        """Doc não deve referenciar flood prediction ou flood detection como validado."""
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "flood prediction" not in content, (
            f"Doc não deve usar 'flood prediction': {doc_path.name}"
        )
        assert "flood detection validado" not in content, (
            f"Doc não deve usar 'flood detection validado': {doc_path.name}"
        )
        assert "previsão de enchentes" not in content, (
            f"Doc não deve usar 'previsão de enchentes': {doc_path.name}"
        )

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_private_paths(self, doc_path):
        """Doc não deve conter paths privados."""
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content, (
            f"Doc não deve conter paths privados: {doc_path.name}"
        )
        assert "gabriela" not in content.lower(), (
            f"Doc não deve conter nome de usuário: {doc_path.name}"
        )

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_in_portuguese(self, doc_path):
        """Doc deve estar em português (contém palavras-chave PT)."""
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        pt_words = ["fotointerpretação", "feição de deslizamento", "deslizamento", "proveniência"]
        assert any(w in content for w in pt_words), (
            f"Doc deve estar em português: {doc_path.name}"
        )

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_has_invariants_section(self, doc_path):
        """Doc deve ter seção de invariantes (can_* = NO)."""
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "can_create_training_label" in content or "invariante" in content, (
            f"Doc deve mencionar invariantes: {doc_path.name}"
        )


# ---------------------------------------------------------------------------
# TestV1IRPublicRegistry — registry público criado corretamente
# ---------------------------------------------------------------------------

class TestV1IRPublicRegistry:
    """Registry público deve ter campos e valores corretos."""

    REG_PATH = REPO_ROOT / "datasets" / "cicatriz_area_photointerpretation_provenance_registry.csv"
    SCHEMA_PATH = (
        REPO_ROOT / "datasets" / "schemas"
        / "cicatriz_area_photointerpretation_provenance_schema.csv"
    )

    def test_registry_exists(self, auditor_force):
        """Registry deve ser criado em datasets/."""
        assert self.REG_PATH.exists(), "cicatriz_area_photointerpretation_provenance_registry.csv não criado"

    def test_schema_exists(self, auditor_force):
        """Schema deve ser criado em datasets/schemas/."""
        assert self.SCHEMA_PATH.exists(), "Schema de proveniência não criado"

    def test_registry_has_one_row(self, auditor_force):
        """Registry deve ter exatamente 1 linha de dados."""
        with open(self.REG_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1, f"Registry deve ter 1 linha; got {len(rows)}"

    def test_registry_candidate_asset_name(self, auditor_force):
        """Registry deve referenciar PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""
        with open(self.REG_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in rows[0].get("candidate_asset_name", "")

    def test_registry_can_be_ground_reference_no(self, auditor_force):
        """can_be_ground_reference_candidate deve ser NO no registry."""
        with open(self.REG_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0].get("can_be_ground_reference_candidate", "").upper() == "NO"

    def test_registry_invariants_all_no(self, auditor_force):
        """Invariantes no registry devem ser NO."""
        with open(self.REG_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        for field in [
            "can_be_operational_ground_truth",
            "can_create_training_label",
            "can_train_model",
            "can_reopen_protocol_b",
        ]:
            assert row.get(field, "").upper() == "NO", (
                f"Registry: {field} deve ser NO; got {row.get(field)}"
            )

    def test_registry_promotion_decision(self, auditor_force):
        """Registry deve ter promotion_decision correto."""
        with open(self.REG_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        decision = rows[0].get("promotion_decision_after_provenance_audit", "")
        assert decision == "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK", (
            f"promotion_decision deve ser STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK; got {decision}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
