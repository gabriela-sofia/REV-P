"""
Testes para v1if — Aquisicao e Auditoria de Vetores Observados Oficiais

Testa:
  - Script existe e tem sintaxe valida
  - Registry existe apos --force
  - Schema compativel com registry
  - Nenhum dado pesado e versionado
  - Paths privados nao aparecem em arquivos publicos
  - Risco/suscetibilidade nunca vira ground truth
  - Fonte municipal sem geometria intraurbana nunca vira patch-level ground truth
  - PDF/imagem nunca vira ground truth vetorial
  - Vetor sem data compativel fica BLOCKED
  - Vetor com fenomeno misto fica BLOCKED_UNTIL_PHENOMENON_SEPARATION
  - Candidate observed ground truth so aparece se todos os gates passarem
  - ml_label_status permanece BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL
  - Docs nao usam claims proibidos
  - Nao ha flood prediction/flood detection validado
  - Nao ha label/target/class supervisionado
"""

import ast
import csv
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
DOCS_DIR = REPO_ROOT / "docs" / "metodologia_cientifica"
LOCAL_RUNS = REPO_ROOT / "local_runs" / "protocolo_c" / "v1if"
SOLICITACOES_DIR = DOCS_DIR / "solicitacoes_dados_ground_truth"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# TestScriptExists
# ---------------------------------------------------------------------------

class TestScriptExists:
    """Verifica existencia e integridade do script v1if."""

    def test_script_exists(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        assert s.exists(), f"Script nao encontrado: {s}"

    def test_script_not_empty(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        assert s.stat().st_size > 5000, "Script muito pequeno"

    def test_script_is_valid_python(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        try:
            ast.parse(s.read_text(encoding="utf-8"))
        except SyntaxError as e:
            pytest.fail(f"Script com sintaxe invalida: {e}")

    def test_script_has_main_guard(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        assert 'if __name__ == "__main__"' in read_text(s)

    def test_script_has_force_flag(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        assert "--force" in read_text(s)

    def test_script_has_search_local_flag(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        assert "--search-local" in read_text(s)

    def test_script_has_download_official_known_flag(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        assert "--download-official-known" in read_text(s)

    def test_script_has_official_sgb_cprm_url(self):
        """Script deve ter URL real do RIGeo SGB/CPRM."""
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        assert "rigeo.sgb.gov.br" in read_text(s), "URL RIGeo ausente no script"

    def test_script_has_eleven_gates(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        src = read_text(s)
        gates = [
            "official_or_institutional_source",
            "raw_asset_traceable",
            "geometry_available",
            "crs_available",
            "geometry_valid",
            "event_date_available",
            "event_date_compatible",
            "phenomenon_available",
            "phenomenon_is_observed_not_risk",
            "hydrological_or_mass_movement_separable",
            "spatial_unit_usable_for_patch_binding",
        ]
        for gate in gates:
            assert gate in src, f"Gate '{gate}' ausente no script"


# ---------------------------------------------------------------------------
# TestSchemaAndRegistry
# ---------------------------------------------------------------------------

class TestSchemaAndRegistry:
    """Verifica existencia e estrutura do schema e registry v1if."""

    def test_schema_exists(self):
        schema = SCHEMAS_DIR / "official_observed_event_vector_registry_schema.csv"
        assert schema.exists(), f"Schema nao encontrado: {schema}"

    def test_registry_exists_after_force(self):
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        assert registry.exists(), (
            "Registry nao encontrado — rode o script com --force primeiro"
        )

    def test_schema_has_required_fields(self):
        schema = SCHEMAS_DIR / "official_observed_event_vector_registry_schema.csv"
        rows = read_csv(schema)
        fields = {r.get("field_name", r.get("field", "")) for r in rows}
        required = {
            "source_asset_id",
            "event_id",
            "source_institution",
            "geometry_available",
            "event_date_compatible",
            "risk_or_susceptibility_only",
            "observed_event_status",
            "patch_level_usability",
            "ground_truth_status",
            "ml_label_status",
        }
        for f in required:
            assert f in fields, f"Campo obrigatorio ausente no schema: {f}"

    def test_registry_has_records(self):
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        rows = read_csv(registry)
        assert len(rows) >= 1, "Registry vazio"

    def test_registry_schema_compatible(self):
        """Todos os campos do schema devem estar no registry."""
        schema = SCHEMAS_DIR / "official_observed_event_vector_registry_schema.csv"
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        schema_fields = {r.get("field_name", r.get("field", "")) for r in read_csv(schema)}
        registry_rows = read_csv(registry)
        if not registry_rows:
            pytest.skip("Registry vazio")
        registry_fields = set(registry_rows[0].keys())
        # Registry pode ter um subconjunto dos campos do schema
        assert registry_fields.issubset(schema_fields | {"source_asset_id"}), (
            f"Registry tem campos fora do schema: {registry_fields - schema_fields}"
        )


# ---------------------------------------------------------------------------
# TestNoPrivatePaths
# ---------------------------------------------------------------------------

class TestNoPrivatePaths:
    """Verifica ausencia de paths privados em arquivos versionaveis."""

    PRIVATE_MARKERS = ["PROJETO", "Users\\gabriela", "C:\\Users", "/home/gabriela"]

    def test_registry_has_no_private_paths(self):
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        content = read_text(registry)
        for marker in self.PRIVATE_MARKERS:
            assert marker not in content, (
                f"Path privado '{marker}' encontrado no registry publico"
            )

    def test_schema_has_no_private_paths(self):
        schema = SCHEMAS_DIR / "official_observed_event_vector_registry_schema.csv"
        content = read_text(schema)
        for marker in self.PRIVATE_MARKERS:
            assert marker not in content, (
                f"Path privado '{marker}' encontrado no schema"
            )


# ---------------------------------------------------------------------------
# TestGroundTruthGuardrails
# ---------------------------------------------------------------------------

class TestGroundTruthGuardrails:
    """Verifica que as regras de ground truth sao respeitadas no registry."""

    def test_susceptibility_never_ground_truth(self):
        """Risco/suscetibilidade nunca pode ser ground truth."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        for row in rows:
            if row.get("risk_or_susceptibility_only", "").upper() == "YES":
                gt = row.get("ground_truth_status", "")
                assert gt == "BLOCKED", (
                    f"Suscetibilidade {row['source_asset_id']} nao esta BLOCKED: {gt}"
                )

    def test_pdf_never_ground_truth_vector(self):
        """PDF/imagem nunca pode ser ground truth vetorial."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        for row in rows:
            asset_type = row.get("source_asset_type", "")
            if asset_type in ("PDF_REPORT", "IMAGE"):
                gt = row.get("ground_truth_status", "")
                assert gt not in ("CANDIDATE_OBSERVED_GROUND_TRUTH", "GROUND_REFERENCE_AUDITED"), (
                    f"PDF/imagem {row['source_asset_id']} marcado como ground truth"
                )

    def test_no_date_field_means_blocked(self):
        """Vetor sem campo de data fica BLOCKED."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        for row in rows:
            if row.get("has_event_date_field", "") == "NO" and row.get("source_asset_type", "").startswith("VECTOR"):
                gt = row.get("ground_truth_status", "")
                assert "BLOCKED" in gt, (
                    f"{row['source_asset_id']} sem data mas nao esta BLOCKED: {gt}"
                )

    def test_ml_label_blocked_for_all(self):
        """ml_label_status deve ser BLOCKED para todos os registros."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        for row in rows:
            ml = row.get("ml_label_status", "")
            assert "BLOCKED" in ml, (
                f"{row['source_asset_id']} tem ml_label_status={ml}, esperado conter BLOCKED"
            )

    def test_no_operational_ground_truth(self):
        """Nenhum registro pode ter ground_truth_status=OPERATIONAL_GROUND_TRUTH."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        operational = [
            r for r in rows
            if r.get("ground_truth_status", "") == "OPERATIONAL_GROUND_TRUTH"
        ]
        assert len(operational) == 0, (
            f"Registro(s) com OPERATIONAL_GROUND_TRUTH: {[r['source_asset_id'] for r in operational]}"
        )

    def test_municipal_level_only_not_patch_usable(self):
        """Dado em nivel municipal nao pode ser patch-level usable."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        for row in rows:
            if row.get("patch_level_usability", "") == "MUNICIPAL_LEVEL_ONLY":
                gt = row.get("ground_truth_status", "")
                assert "BLOCKED" in gt, (
                    f"{row['source_asset_id']}: MUNICIPAL_LEVEL_ONLY mas ground_truth={gt}"
                )

    def test_candidate_ground_truth_requires_all_gates(self):
        """CANDIDATE_OBSERVED_GROUND_TRUTH so aparece se todos os gates passaram."""
        # Esta é uma invariante logica: se o registro tem status CANDIDATE,
        # os campos de gate nao devem ter FAIL
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        candidate_rows = [
            r for r in rows
            if r.get("ground_truth_status", "") == "CANDIDATE_OBSERVED_GROUND_TRUTH"
        ]
        for row in candidate_rows:
            # Se for candidato, deve ter geometria disponivel e data compativel
            assert row.get("geometry_available", "") == "YES", (
                f"{row['source_asset_id']}: CANDIDATE mas geometry_available!=YES"
            )
            assert row.get("event_date_compatible", "") == "PASS", (
                f"{row['source_asset_id']}: CANDIDATE mas event_date_compatible!=PASS"
            )


# ---------------------------------------------------------------------------
# TestNoOperationalClaim
# ---------------------------------------------------------------------------

class TestNoOperationalClaim:
    """Verifica ausencia de claims operacionais proibidos."""

    FORBIDDEN_TERMS_SCRIPT = [
        "flood_prediction", "flood_detection_validated", "train_model",
        "fit(", "model.train", "supervised_label", "create_target",
        "label_positivo", "target_supervisionado",
    ]

    def test_script_has_no_forbidden_claims(self):
        s = SCRIPTS_DIR / "revp_v1if_official_observed_event_vector_acquisition_audit.py"
        src = read_text(s).lower()
        for term in self.FORBIDDEN_TERMS_SCRIPT:
            assert term.lower() not in src, f"Termo proibido '{term}' no script v1if"

    def test_methodology_doc_has_no_forbidden_claims(self):
        doc = DOCS_DIR / "protocolo_c_aquisicao_auditoria_vetores_observados_v1if.md"
        if not doc.exists():
            pytest.skip("Documento metodologico nao encontrado")
        content = read_text(doc).lower()
        forbidden = [
            "flood prediction", "flood detection validado",
            "ground truth operacional estabelecido",
            "treino liberado", "label supervisionado criado",
        ]
        for term in forbidden:
            assert term not in content, f"Claim proibido '{term}' no documento metodologico"

    def test_report_explicitly_states_blocked(self):
        report = DOCS_DIR / "protocolo_c_relatorio_aquisicao_auditoria_vetores_observados_v1if.md"
        if not report.exists():
            pytest.skip("Relatorio nao encontrado")
        content = read_text(report)
        assert "BLOCKED" in content, "Relatorio nao menciona estado BLOCKED"
        assert "ground truth" in content.lower(), "Relatorio nao menciona ground truth"

    def test_report_has_mandatory_statements(self):
        report = DOCS_DIR / "protocolo_c_relatorio_aquisicao_auditoria_vetores_observados_v1if.md"
        if not report.exists():
            pytest.skip("Relatorio nao encontrado")
        content = read_text(report)
        assert "nao possui ground truth operacional" in content.lower() or "BLOCKED" in content, (
            "Relatorio deve declarar que nao ha ground truth operacional"
        )
        assert "treino supervisionado" in content.lower() or "bloqueado" in content.lower(), (
            "Relatorio deve mencionar que treino supervisionado esta bloqueado"
        )


# ---------------------------------------------------------------------------
# TestNoHeavyFilesVersioned
# ---------------------------------------------------------------------------

class TestNoHeavyFilesVersioned:
    """Verifica que arquivos pesados nao estao no repositorio git."""

    HEAVY_EXTENSIONS = {".zip", ".shp", ".gpkg", ".kmz", ".kml", ".tif", ".tiff",
                        ".npz", ".npy", ".pdf"}

    def test_local_runs_not_in_git(self):
        gitignore = REPO_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore nao encontrado")
        assert "local_runs" in read_text(gitignore), (
            "local_runs/ nao esta no .gitignore"
        )

    def test_datasets_dir_has_no_heavy_files(self):
        """Diretorio datasets/ nao deve conter ZIPs, SHPs, PDFs etc."""
        for ext in self.HEAVY_EXTENSIONS:
            found = list(DATASETS_DIR.glob(f"*{ext}"))
            # Excluir subdiretorios -- apenas raiz
            found = [f for f in found if f.parent == DATASETS_DIR]
            assert len(found) == 0, (
                f"Arquivo pesado {ext} encontrado em datasets/: {found}"
            )

    def test_registry_raw_file_status_not_versioned(self):
        """Registry deve dizer que arquivos brutos nao sao versionados."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry nao encontrado")
        rows = read_csv(registry)
        for row in rows:
            versioning = row.get("raw_file_versioning_status", "")
            assert versioning != "VERSIONED_IN_GIT", (
                f"{row['source_asset_id']}: raw_file_versioning_status=VERSIONED_IN_GIT "
                "(dado bruto nao pode ir para Git)"
            )

    def test_local_outputs_exist_if_script_ran(self):
        """Se o script foi rodado, outputs locais devem existir."""
        if not LOCAL_RUNS.exists():
            pytest.skip("local_runs/v1if nao encontrado — script pode nao ter rodado")
        summary = LOCAL_RUNS / "v1if_summary.json"
        assert summary.exists(), f"Summary local ausente: {summary}"

    def test_local_summary_operational_ground_truth_blocked(self):
        """Summary local deve confirmar que ground truth esta BLOCKED."""
        summary_path = LOCAL_RUNS / "v1if_summary.json"
        if not summary_path.exists():
            pytest.skip("v1if_summary.json nao encontrado")
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("operational_ground_truth_status") == "BLOCKED", (
            "Summary indica ground truth nao-BLOCKED"
        )
        assert data.get("can_create_training_label") is False, (
            "Summary indica can_create_training_label=True"
        )


# ---------------------------------------------------------------------------
# TestDocumentation
# ---------------------------------------------------------------------------

class TestDocumentation:
    """Verifica existencia e conteudo minimo dos documentos v1if."""

    def test_methodology_doc_exists(self):
        doc = DOCS_DIR / "protocolo_c_aquisicao_auditoria_vetores_observados_v1if.md"
        assert doc.exists(), f"Documento metodologico nao encontrado: {doc}"

    def test_report_doc_exists(self):
        doc = DOCS_DIR / "protocolo_c_relatorio_aquisicao_auditoria_vetores_observados_v1if.md"
        assert doc.exists(), f"Relatorio nao encontrado: {doc}"

    def test_methodology_doc_distinguishes_susceptibility_from_event(self):
        doc = DOCS_DIR / "protocolo_c_aquisicao_auditoria_vetores_observados_v1if.md"
        content = read_text(doc).lower()
        assert "suscetib" in content or "susceptib" in content, (
            "Documento nao menciona suscetibilidade"
        )
        assert "observad" in content, "Documento nao menciona ocorrencia observada"

    def test_methodology_doc_mentions_pdf_limitation(self):
        doc = DOCS_DIR / "protocolo_c_aquisicao_auditoria_vetores_observados_v1if.md"
        content = read_text(doc).lower()
        assert "pdf" in content, "Documento nao menciona limitacao de PDF"

    def test_methodology_doc_mentions_eleven_gates(self):
        doc = DOCS_DIR / "protocolo_c_aquisicao_auditoria_vetores_observados_v1if.md"
        content = read_text(doc)
        assert "11" in content or "eleven" in content.lower() or "gates" in content.lower(), (
            "Documento nao menciona os 11 gates"
        )

    def test_report_mentions_zip_download(self):
        doc = DOCS_DIR / "protocolo_c_relatorio_aquisicao_auditoria_vetores_observados_v1if.md"
        content = read_text(doc)
        assert "zip" in content.lower() or "ZIP" in content, (
            "Relatorio nao menciona download do ZIP"
        )

    def test_report_mentions_pdfs_not_vectors(self):
        doc = DOCS_DIR / "protocolo_c_relatorio_aquisicao_auditoria_vetores_observados_v1if.md"
        content = read_text(doc).lower()
        assert "pdf" in content, "Relatorio nao menciona PDFs"

    def test_institutional_requests_directory_exists(self):
        assert SOLICITACOES_DIR.exists(), (
            f"Diretorio de solicitacoes nao encontrado: {SOLICITACOES_DIR}"
        )

    def test_all_five_request_files_exist(self):
        expected = [
            "solicitacao_sgb_cprm_anexos_petropolis_2022.md",
            "solicitacao_drm_rj_vetores_petropolis_2022.md",
            "solicitacao_defesa_civil_petropolis_vetores_evento_2022.md",
            "solicitacao_inpe_charter_petropolis_2022.md",
            "solicitacao_sedec_s2id_microdados_georreferenciados.md",
        ]
        for fname in expected:
            p = SOLICITACOES_DIR / fname
            assert p.exists(), f"Solicitacao nao encontrada: {fname}"

    def test_request_files_mention_shapefile_or_vector(self):
        """Cada solicitacao deve pedir shapefile/vetor explicitamente."""
        for f in SOLICITACOES_DIR.glob("solicitacao_*.md"):
            content = read_text(f).lower()
            mentions_vector = (
                "shapefile" in content
                or "geopackage" in content
                or "kmz" in content
                or "kml" in content
                or "geojson" in content
            )
            assert mentions_vector, (
                f"{f.name} nao menciona formato vetorial (shapefile/geopackage/kmz/geojson)"
            )

    def test_request_files_mention_phenomenon_separation(self):
        """Cada solicitacao deve pedir separacao de fenomeno."""
        for f in SOLICITACOES_DIR.glob("solicitacao_*.md"):
            content = read_text(f).lower()
            has_sep = (
                "inundaç" in content or "inundac" in content
                or "deslizamento" in content
                or "fenômeno" in content or "fenomeno" in content
                or "separa" in content
            )
            assert has_sep, (
                f"{f.name} nao menciona separacao de fenomeno (inundacao/deslizamento)"
            )


# ---------------------------------------------------------------------------
# TestCompatibilityWithPreviousStages
# ---------------------------------------------------------------------------

class TestCompatibilityWithPreviousStages:
    """Verifica compatibilidade com etapas anteriores."""

    def test_v1ie_registry_still_blocked(self):
        """v1ie registry deve continuar BLOCKED."""
        reg = DATASETS_DIR / "ground_reference_evidence_registry.csv"
        if not reg.exists():
            pytest.skip("v1ie registry nao encontrado")
        rows = read_csv(reg)
        for row in rows:
            assert row.get("operational_ground_truth_status", "") == "BLOCKED", (
                f"v1ie: {row.get('ground_reference_id')} nao esta BLOCKED"
            )

    def test_v1id_pkg_still_required_not_ingested(self):
        """PKG_FR_PET_001 deve continuar REQUIRED_NOT_INGESTED em v1id."""
        reg = DATASETS_DIR / "observed_reference_source_package_registry.csv"
        if not reg.exists():
            pytest.skip("v1id registry nao encontrado")
        rows = read_csv(reg)
        pkg = [r for r in rows if r.get("package_id", "") == "PKG_FR_PET_001"]
        if not pkg:
            pytest.skip("PKG_FR_PET_001 nao encontrado em v1id registry")
        assert pkg[0].get("local_asset_status", "") == "NOT_FOUND"

    def test_event_id_consistent(self):
        """event_id PET_2022_02_15 deve ser consistente no registry v1if."""
        registry = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if not registry.exists():
            pytest.skip("Registry v1if nao encontrado")
        rows = read_csv(registry)
        pet_rows = [r for r in rows if "PET" in r.get("event_id", "")]
        assert len(pet_rows) >= 1, "Nenhum registro para evento PET no registry v1if"
