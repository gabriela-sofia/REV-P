"""
test_revp_v1ir_official_documented_event_unit_builder.py

Testes para v1ir — Official Documented Event Unit Ground Reference Builder.

Expectativas:
- Script lê PDFs CPRM de v1if e extrai unidades de evento documentadas.
- Cada unidade tem fonte, data, localidade, fenômeno e excerto verificável.
- Relatório documental NÃO vira vetor observado.
- Localidade sem coordenada NÃO vira ponto inventado.
- can_* = NO sempre (invariantes absolutos).
- Sem paths privados em arquivos públicos.
"""

import pytest
import json
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1ir_official_documented_event_unit_builder import (
    OfficialDocumentedEventUnitBuilder,
    DocumentedEventUnit,
    GateResult,
    TextExtractionLog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def builder_run():
    """Builder executado uma vez por módulo (sem --force)."""
    b = OfficialDocumentedEventUnitBuilder(force=False)
    result = b.run()
    return b, result


@pytest.fixture(scope="module")
def builder_force():
    """Builder com force=True para outputs frescos."""
    b = OfficialDocumentedEventUnitBuilder(force=True)
    result = b.run()
    return b, result


# ---------------------------------------------------------------------------
# TestV1IREventUnitBasics — script existe e roda
# ---------------------------------------------------------------------------

class TestV1IREventUnitBasics:
    """O script existe, instancia e roda sem erros."""

    def test_script_exists(self):
        """Script deve existir."""
        p = SCRIPTS_DIR / "revp_v1ir_official_documented_event_unit_builder.py"
        assert p.exists(), "Script não encontrado"

    def test_builder_instantiation(self):
        """Builder deve instanciar sem erros."""
        b = OfficialDocumentedEventUnitBuilder(force=False)
        assert b is not None

    def test_run_returns_dict(self, builder_run):
        """run() deve retornar dict com chaves esperadas."""
        _, result = builder_run
        assert isinstance(result, dict)
        assert "status" in result
        assert "documents_found" in result
        assert "event_units_created" in result

    def test_run_status_complete(self, builder_run):
        """Status de run() deve ser 'complete'."""
        _, result = builder_run
        assert result["status"] == "complete"

    def test_stats_complete(self, builder_run):
        """stats['complete'] deve ser True após run."""
        b, _ = builder_run
        assert b.stats["complete"] is True

    def test_run_with_all_flags(self):
        """Builder deve aceitar todos os flags sem erro."""
        b = OfficialDocumentedEventUnitBuilder(force=False)
        result = b.run(
            scan_v1if_documents=True,
            extract_light_text=True,
            extract_event_units=True,
            classify_spatial_precision=True,
            emit_ground_reference_candidates=True,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# TestV1IREventUnitOutputs — outputs locais são gerados
# ---------------------------------------------------------------------------

class TestV1IREventUnitOutputs:
    """Os 8 arquivos de output local devem existir após run."""

    LOCAL_DIR = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir"

    EXPECTED_FILES = [
        "v1ir_document_inventory.csv",
        "v1ir_text_extraction_log.csv",
        "v1ir_documented_event_units.csv",
        "v1ir_spatial_precision_audit.csv",
        "v1ir_temporal_precision_audit.csv",
        "v1ir_ground_reference_candidate_decision.csv",
        "v1ir_summary.json",
        "v1ir_qa.csv",
    ]

    def test_output_dir_exists(self, builder_force):
        """local_runs/protocolo_c/v1ir/ deve existir."""
        assert self.LOCAL_DIR.exists()

    @pytest.mark.parametrize("fname", EXPECTED_FILES)
    def test_output_file_created(self, builder_force, fname):
        """Cada arquivo de output deve existir."""
        assert (self.LOCAL_DIR / fname).exists(), f"{fname} não gerado"

    def test_summary_json_stage(self, builder_force):
        """summary.json deve ter stage='v1ir_event_units'."""
        with open(self.LOCAL_DIR / "v1ir_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["stage"] == "v1ir_event_units"

    def test_summary_json_invariants_false(self, builder_force):
        """summary.json deve ter todos os can_* False."""
        with open(self.LOCAL_DIR / "v1ir_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        inv = data.get("invariants", {})
        assert inv.get("can_create_training_label") is False
        assert inv.get("can_train_model") is False
        assert inv.get("can_reopen_protocol_b") is False
        assert inv.get("can_be_operational_ground_truth") is False

    def test_event_units_csv_has_header(self, builder_force):
        """v1ir_documented_event_units.csv deve ter header correto."""
        p = self.LOCAL_DIR / "v1ir_documented_event_units.csv"
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "documented_event_unit_id" in reader.fieldnames
            assert "can_create_training_label" in reader.fieldnames

    def test_qa_all_passed(self, builder_force):
        """v1ir_qa.csv deve ter todos os checks True."""
        p = self.LOCAL_DIR / "v1ir_qa.csv"
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        failed = [r for r in rows if r.get("passed", "True") != "True"]
        assert len(failed) == 0, f"QA checks falhando: {failed}"

    def test_document_inventory_has_pdfs(self, builder_force):
        """v1ir_document_inventory.csv deve listar PDFs."""
        p = self.LOCAL_DIR / "v1ir_document_inventory.csv"
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0, "Inventário não deve ser vazio"
        pdf_names = [r.get("document_name_sanitized", "") for r in rows]
        assert any("CPRM" in n or "ANEXO" in n for n in pdf_names), (
            "Inventário deve conter documentos CPRM/ANEXO"
        )


# ---------------------------------------------------------------------------
# TestV1IREventUnitRegistry — registry/schema criados se houver candidatos
# ---------------------------------------------------------------------------

class TestV1IREventUnitRegistry:
    """Registry público e schema devem ser criados."""

    REG = REPO_ROOT / "datasets" / "official_documented_event_unit_registry.csv"
    GATE = REPO_ROOT / "datasets" / "official_documented_event_ground_reference_gate_matrix.csv"
    SCHEMA = REPO_ROOT / "datasets" / "schemas" / "official_documented_event_unit_schema.csv"
    GATE_SCHEMA = REPO_ROOT / "datasets" / "schemas" / "official_documented_event_ground_reference_gate_matrix_schema.csv"

    def test_registry_exists(self, builder_force):
        """Registry deve ser criado em datasets/."""
        assert self.REG.exists(), "official_documented_event_unit_registry.csv não encontrado"

    def test_gate_matrix_exists(self, builder_force):
        """Gate matrix deve ser criada em datasets/."""
        assert self.GATE.exists()

    def test_schema_exists(self, builder_force):
        """Schema do registry deve ser criado."""
        assert self.SCHEMA.exists()

    def test_gate_schema_exists(self, builder_force):
        """Schema da gate matrix deve ser criado."""
        assert self.GATE_SCHEMA.exists()

    def test_registry_has_rows(self, builder_force):
        """Registry deve ter ao menos 1 linha."""
        with open(self.REG, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    def test_registry_invariants_all_no(self, builder_force):
        """Todos os can_* no registry devem ser NO."""
        with open(self.REG, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for field in ["can_be_operational_ground_truth",
                          "can_create_training_label",
                          "can_train_model",
                          "can_reopen_protocol_b"]:
                assert row.get(field, "").upper() == "NO", (
                    f"Registry: {field} deve ser NO; got {row.get(field)}"
                )

    def test_registry_cprm_source(self, builder_force):
        """Registry: source_institution deve ser CPRM."""
        with open(self.REG, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        institutions = {r.get("source_institution", "") for r in rows}
        assert "CPRM" in institutions, "Registry deve ter CPRM como instituição"

    def test_gate_matrix_has_pass_entries(self, builder_force):
        """Gate matrix deve ter entradas PASS."""
        with open(self.GATE, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        passes = [r for r in rows if r.get("source_official") == "PASS"]
        assert len(passes) > 0, "Gate matrix deve ter entradas com source_official=PASS"


# ---------------------------------------------------------------------------
# TestV1IRDataDeSistema — data de sistema não é aceita
# ---------------------------------------------------------------------------

class TestV1IRDataDeSistema:
    """Data de sistema de arquivos não deve ser usada como evidence."""

    def test_event_dates_are_from_documents(self, builder_run):
        """Datas de evento devem vir dos documentos, não do sistema."""
        b, _ = builder_run
        for unit in b.event_units:
            if unit.event_date:
                # Deve ser formato DD/MM/YYYY de documento, não mtime
                import re
                assert re.match(r"\d{2}/\d{2}/\d{4}", unit.event_date), (
                    f"event_date deve ser DD/MM/YYYY de documento: {unit.event_date}"
                )

    def test_event_dates_post_2022_event(self, builder_run):
        """Datas de evento devem ser pós 15/02/2022."""
        b, _ = builder_run
        for unit in b.event_units:
            if unit.event_date and "/" in unit.event_date:
                parts = unit.event_date.split("/")
                if len(parts) == 3:
                    dd, mm, yyyy = parts
                    assert yyyy == "2022", f"Ano deve ser 2022: {unit.event_date}"
                    assert int(mm) >= 2, f"Mês deve ser >= 02: {unit.event_date}"
                    if mm == "02":
                        assert int(dd) >= 15, f"Dia deve ser >= 15 em fevereiro: {unit.event_date}"

    def test_no_pre_event_dates(self, builder_run):
        """Nenhuma data anterior ao evento (2022-02-15) deve ser aceita."""
        b, _ = builder_run
        for unit in b.event_units:
            if unit.event_date and "/" in unit.event_date:
                parts = unit.event_date.split("/")
                if len(parts) == 3:
                    dd, mm, yyyy = parts
                    # Ano 2013 ou 2015 (do SIG histórico) não deve aparecer
                    assert yyyy not in ("2013", "2014", "2015", "2016"), (
                        f"Data de SIG histórico não deve aparecer como event_date: {unit.event_date}"
                    )


# ---------------------------------------------------------------------------
# TestV1IRLocalidadeSemCoordenada — localidade sem coord não vira ponto
# ---------------------------------------------------------------------------

class TestV1IRLocalidadeSemCoordenada:
    """Localidade sem coordenada documentada não gera coordenada inventada."""

    def test_coordinate_only_if_documented(self, builder_run):
        """coordinate_available=YES apenas se coordenada está no documento."""
        b, _ = builder_run
        for unit in b.event_units:
            if unit.coordinate_available == "YES":
                # Deve ter lat e lon preenchidos
                assert unit.coordinate_lat, (
                    f"coordinate_available=YES mas lat vazio: {unit.documented_event_unit_id}"
                )
                assert unit.coordinate_lon, (
                    f"coordinate_available=YES mas lon vazio: {unit.documented_event_unit_id}"
                )
                # Lat deve estar no intervalo de Petrópolis/RJ
                try:
                    lat = float(unit.coordinate_lat)
                    assert -23.5 < lat < -21.0, (
                        f"Lat fora do intervalo esperado para RJ: {lat}"
                    )
                except ValueError:
                    pass

    def test_no_invented_coordinates(self, builder_run):
        """coordinate_source nunca deve ser 'INFERRED' ou 'INTERPOLATED'."""
        b, _ = builder_run
        for unit in b.event_units:
            assert "INFERRED" not in unit.coordinate_source.upper(), (
                f"coordinate_source não pode ser INFERRED: {unit.documented_event_unit_id}"
            )
            assert "INTERPOLATED" not in unit.coordinate_source.upper(), (
                f"coordinate_source não pode ser INTERPOLATED"
            )

    def test_no_coordinate_if_only_neighborhood(self, builder_run):
        """Unidade com apenas bairro (NEIGHBORHOOD) não deve ter coordinate_available=YES."""
        b, _ = builder_run
        for unit in b.event_units:
            if (unit.spatial_precision == "NEIGHBORHOOD"
                    and unit.coordinate_available == "YES"):
                # Se NEIGHBORHOOD + coordinate_available=YES → deve ter lat/lon reais
                # (por exemplo, se o texto do doc tinha coord mesmo com bairro)
                assert unit.coordinate_lat and unit.coordinate_lon, (
                    f"NEIGHBORHOOD com coordinate_available=YES mas sem lat/lon: {unit.documented_event_unit_id}"
                )

    def test_spatial_precision_values_valid(self, builder_run):
        """spatial_precision deve ter valores válidos."""
        valid = {"EXACT_COORDINATE", "STREET_OR_LOCALITY", "NEIGHBORHOOD",
                 "MUNICIPAL_ONLY", "UNKNOWN"}
        b, _ = builder_run
        for unit in b.event_units:
            assert unit.spatial_precision in valid, (
                f"spatial_precision inválido: {unit.spatial_precision}"
            )


# ---------------------------------------------------------------------------
# TestV1IRRelatorioNaoViraVetor — relatório não vira vetor observado
# ---------------------------------------------------------------------------

class TestV1IRRelatorioNaoViraVetor:
    """O relatório documental não pode ser promovido a vetor observado."""

    def test_no_unit_is_operational_ground_truth(self, builder_run):
        """Nenhuma unidade pode ser ground truth operacional."""
        b, _ = builder_run
        for unit in b.event_units:
            assert unit.can_be_operational_ground_truth == "NO", (
                f"can_be_operational_ground_truth deve ser NO: {unit.documented_event_unit_id}"
            )

    def test_candidate_status_is_documentary(self, builder_run):
        """Candidatos devem ser CANDIDATE_DOCUMENTARY_ONLY ou CANDIDATE_WITH_DOCUMENTED_COORDINATE."""
        b, _ = builder_run
        valid_candidate = {
            "CANDIDATE_WITH_DOCUMENTED_COORDINATE",
            "CANDIDATE_DOCUMENTARY_ONLY",
            "INSUFFICIENT_EVIDENCE",
        }
        for unit in b.event_units:
            assert unit.ground_reference_candidate_status in valid_candidate, (
                f"Status inválido: {unit.ground_reference_candidate_status}"
            )

    def test_no_vector_claim_in_summary(self, builder_force):
        """summary.json não deve afirmar que relatório é vetor observado."""
        with open(REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir" / "v1ir_summary.json",
                  encoding="utf-8") as f:
            data = json.load(f)
        # next_step não deve afirmar que cria label
        next_step = data.get("next_step_if_warranted", "")
        assert "label automático" not in next_step.lower() or "não" in next_step.lower(), (
            "next_step não deve afirmar criação de label automático"
        )

    def test_excerpt_is_text_not_coordinate(self, builder_run):
        """document_excerpt deve ser texto descritivo, não uma coordenada pura."""
        b, _ = builder_run
        for unit in b.event_units:
            if unit.document_excerpt_sanitized:
                # Não deve ser apenas números de coordenada
                assert not unit.document_excerpt_sanitized.strip().startswith("-2"), (
                    f"Excerpt não deve ser coordenada pura: {unit.documented_event_unit_id}"
                )


# ---------------------------------------------------------------------------
# TestV1IRGroundRefRequirements — ground ref exige fonte+data+fenômeno+loc
# ---------------------------------------------------------------------------

class TestV1IRGroundRefRequirements:
    """Ground reference candidate exige todos os critérios."""

    def test_candidates_have_official_source(self, builder_run):
        """Todos os candidatos devem ter fonte oficial."""
        b, _ = builder_run
        candidates = [u for u in b.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
        for c in candidates:
            assert c.source_institution in ("CPRM", "SGB", "DRM"), (
                f"Candidato deve ter fonte oficial: {c.documented_event_unit_id}"
            )

    def test_candidates_have_date_or_window(self, builder_run):
        """Candidatos devem ter data ou janela temporal."""
        b, _ = builder_run
        candidates = [u for u in b.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
        for c in candidates:
            assert c.event_date or c.event_window, (
                f"Candidato deve ter event_date ou event_window: {c.documented_event_unit_id}"
            )

    def test_candidates_have_phenomenon(self, builder_run):
        """Candidatos devem ter fenômeno explícito."""
        b, _ = builder_run
        candidates = [u for u in b.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
        for c in candidates:
            assert c.phenomenon_group != "UNKNOWN", (
                f"Candidato deve ter phenomenon_group: {c.documented_event_unit_id}"
            )

    def test_candidates_have_location(self, builder_run):
        """Candidatos devem ter localidade explícita."""
        b, _ = builder_run
        candidates = [u for u in b.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
        for c in candidates:
            assert c.locality_text_sanitized or c.municipality, (
                f"Candidato deve ter localidade: {c.documented_event_unit_id}"
            )

    def test_candidates_have_excerpt(self, builder_run):
        """Candidatos devem ter trecho documental."""
        b, _ = builder_run
        candidates = [u for u in b.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
        for c in candidates:
            assert c.document_excerpt_sanitized, (
                f"Candidato deve ter excerpt verificável: {c.documented_event_unit_id}"
            )

    def test_insufficient_units_not_candidate(self, builder_run):
        """Unidades com INSUFFICIENT_EVIDENCE devem ter can_be_ground_reference_candidate=NO."""
        b, _ = builder_run
        insufficient = [u for u in b.event_units
                        if u.ground_reference_candidate_status == "INSUFFICIENT_EVIDENCE"]
        for u in insufficient:
            assert u.can_be_ground_reference_candidate == "NO", (
                f"INSUFFICIENT_EVIDENCE: can_be_ground_reference_candidate deve ser NO"
            )

    def test_petropolis_2022_cprm_units_found(self, builder_run):
        """Deve haver ao menos 5 candidatos documentais do evento Petrópolis 2022."""
        b, _ = builder_run
        candidates = [u for u in b.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
        cprm_2022 = [c for c in candidates if "CPRM" in c.source_institution]
        assert len(cprm_2022) >= 5, (
            f"Esperados >= 5 candidatos CPRM/2022; encontrados: {len(cprm_2022)}"
        )


# ---------------------------------------------------------------------------
# TestV1IRInvariants — can_* = NO sempre
# ---------------------------------------------------------------------------

class TestV1IRInvariants:
    """Invariantes absolutos: todos os can_* = NO."""

    def test_can_create_training_label_never_yes(self, builder_run):
        """can_create_training_label deve ser NO em todas as unidades."""
        b, _ = builder_run
        for u in b.event_units:
            assert u.can_create_training_label == "NO", (
                f"can_create_training_label deve ser NO: {u.documented_event_unit_id}"
            )

    def test_can_train_model_never_yes(self, builder_run):
        """can_train_model deve ser NO em todas as unidades."""
        b, _ = builder_run
        for u in b.event_units:
            assert u.can_train_model == "NO", (
                f"can_train_model deve ser NO: {u.documented_event_unit_id}"
            )

    def test_can_reopen_protocol_b_never_yes(self, builder_run):
        """can_reopen_protocol_b deve ser NO em todas as unidades."""
        b, _ = builder_run
        for u in b.event_units:
            assert u.can_reopen_protocol_b == "NO", (
                f"can_reopen_protocol_b deve ser NO: {u.documented_event_unit_id}"
            )

    def test_can_be_operational_ground_truth_never_yes(self, builder_run):
        """can_be_operational_ground_truth deve ser NO em todas as unidades."""
        b, _ = builder_run
        for u in b.event_units:
            assert u.can_be_operational_ground_truth == "NO", (
                f"can_be_operational_ground_truth deve ser NO: {u.documented_event_unit_id}"
            )


# ---------------------------------------------------------------------------
# TestV1IRSemPathPrivado — sem path privado em públicos
# ---------------------------------------------------------------------------

class TestV1IRSemPathPrivado:
    """Nenhum path privado nos arquivos públicos."""

    def test_no_private_path_in_registry(self, builder_force):
        """Registry não deve conter paths privados."""
        reg = REPO_ROOT / "datasets" / "official_documented_event_unit_registry.csv"
        if not reg.exists():
            pytest.skip("Registry não gerado")
        content = reg.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content
        assert "gabriela" not in content.lower()
        assert "PROJETO" not in content

    def test_no_private_path_in_gate_matrix(self, builder_force):
        """Gate matrix não deve conter paths privados."""
        gate = REPO_ROOT / "datasets" / "official_documented_event_ground_reference_gate_matrix.csv"
        if not gate.exists():
            pytest.skip("Gate matrix não gerada")
        content = gate.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content
        assert "gabriela" not in content.lower()

    def test_no_private_path_in_event_units_csv(self, builder_force):
        """v1ir_documented_event_units.csv não deve conter paths privados."""
        p = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir" / "v1ir_documented_event_units.csv"
        content = p.read_text(encoding="utf-8")
        assert "gabriela" not in content.lower()
        import re
        private = re.findall(r"[A-Za-z]:\\[A-Za-z]+\\[A-Za-z]+", content)
        assert len(private) == 0, f"Paths privados encontrados: {private}"

    def test_no_private_path_in_summary_json(self, builder_force):
        """v1ir_summary.json não deve conter paths privados."""
        p = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ir" / "v1ir_summary.json"
        content = p.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content
        assert "gabriela" not in content.lower()


# ---------------------------------------------------------------------------
# TestV1IRDocs — docs não sugerem e-mail/Protocolo B/flood prediction
# ---------------------------------------------------------------------------

class TestV1IRDocs:
    """Os docs não contêm linguagem proibida."""

    DOCS = [
        REPO_ROOT / "docs" / "metodologia_cientifica" /
        "protocolo_c_eventos_documentados_oficiais_v1ir.md",
        REPO_ROOT / "docs" / "metodologia_cientifica" /
        "protocolo_c_relatorio_eventos_documentados_oficiais_v1ir.md",
    ]

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_exists(self, doc_path):
        assert doc_path.exists(), f"Doc não encontrado: {doc_path.name}"

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_email_suggestion(self, doc_path):
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "enviar e-mail" not in content
        assert "solicitar via e-mail" not in content

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_solicitation(self, doc_path):
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "solicitar ao cprm" not in content
        assert "entrar em contato" not in content
        assert "contatar a" not in content

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_protocol_b(self, doc_path):
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        # Pode mencionar protocolo b se deixar claro que não foi iniciado
        if "protocolo b" in content:
            assert "não" in content or "não iniciado" in content, (
                f"Doc menciona Protocolo B sem negar: {doc_path.name}"
            )

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_flood_prediction(self, doc_path):
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "flood prediction" not in content
        assert "flood detection validado" not in content
        assert "previsão de enchentes" not in content

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_no_private_paths(self, doc_path):
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8")
        assert "C:\\" not in content and "C:/" not in content
        assert "gabriela" not in content.lower()

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_in_portuguese(self, doc_path):
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        pt_words = ["relatório", "vistoria", "deslizamento", "localidade", "fenômeno"]
        assert any(w in content for w in pt_words)

    @pytest.mark.parametrize("doc_path", DOCS)
    def test_doc_states_invariants(self, doc_path):
        if not doc_path.exists():
            pytest.skip(f"Doc não encontrado: {doc_path.name}")
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "can_create_training_label" in content or "invariant" in content


# ---------------------------------------------------------------------------
# TestV1IRPhenomena — fenômenos corretos encontrados
# ---------------------------------------------------------------------------

class TestV1IRPhenomena:
    """Fenômenos corretos devem ser encontrados nos documentos."""

    def test_movement_of_mass_found(self, builder_run):
        """MOVEMENT_OF_MASS deve estar entre os fenômenos."""
        b, _ = builder_run
        phenom = list(b.stats.get("phenomena_found", []))
        assert "MOVEMENT_OF_MASS" in phenom, (
            f"MOVEMENT_OF_MASS não encontrado. Fenômenos: {phenom}"
        )

    def test_most_units_are_movement_of_mass(self, builder_run):
        """Maioria das unidades deve ser MOVEMENT_OF_MASS."""
        b, _ = builder_run
        mm = [u for u in b.event_units if u.phenomenon_group == "MOVEMENT_OF_MASS"]
        assert len(mm) >= len(b.event_units) // 2, (
            f"Maioria deve ser MOVEMENT_OF_MASS; encontrados {len(mm)}/{len(b.event_units)}"
        )

    def test_phenomenon_groups_valid(self, builder_run):
        """Todos os phenomenon_group devem ser valores válidos."""
        valid = {"MOVEMENT_OF_MASS", "FLOODING", "EROSION", "RISK_AREA_MIXED", "UNKNOWN"}
        b, _ = builder_run
        for u in b.event_units:
            assert u.phenomenon_group in valid, (
                f"phenomenon_group inválido: {u.phenomenon_group}"
            )

    def test_cprm_region_is_petropolis(self, builder_run):
        """region deve ser PET para todos os documentos CPRM."""
        b, _ = builder_run
        for u in b.event_units:
            if u.source_institution == "CPRM":
                assert u.region == "PET", (
                    f"region deve ser PET para CPRM: {u.documented_event_unit_id}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
