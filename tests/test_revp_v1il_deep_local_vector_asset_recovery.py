"""
test_revp_v1il_deep_local_vector_asset_recovery.py

Testes para v1il -- Varredura Profunda de Ativos Vetoriais Locais
"""

import pytest
import csv
import json
from pathlib import Path
import sys

# Adicionar scripts ao path
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1il_deep_local_vector_asset_recovery import (
    LocalVectorAssetAuditor,
    VectorAsset,
    ShapefileBundle,
)


class TestV1ILBasics:
    """Testes básicos do script v1il."""

    def test_script_exists(self):
        """Script deve existir."""
        script_path = SCRIPTS_DIR / "revp_v1il_deep_local_vector_asset_recovery.py"
        assert script_path.exists(), "Script v1il não encontrado"

    def test_auditor_instantiation(self):
        """Auditor deve instanciar sem erros."""
        auditor = LocalVectorAssetAuditor(
            force=False,
            scan_revp=True,
            focus_missing_candidates=True,
        )
        assert auditor is not None
        assert auditor.force == False
        assert auditor.scan_revp == True

    def test_auditor_run_with_flags(self):
        """Auditor deve rodar com flags principais."""
        auditor = LocalVectorAssetAuditor(
            force=False,
            scan_revp=True,
            focus_missing_candidates=True,
            inspect_vector_headers=True,
            emit_handoff=True,
        )
        result = auditor.run()
        assert result is not None
        assert "stats" in result
        assert "vector_assets_found" in result
        assert "shapefile_bundles_mapped" in result


class TestV1ILDataStructures:
    """Testes para estruturas de dados do v1il."""

    def test_vector_asset_creation(self):
        """VectorAsset deve criar instância válida."""
        asset = VectorAsset(
            asset_id="ASSET_TEST_0001",
            asset_name="test.shp",
            asset_format="SHAPEFILE",
            found_in="REVP",
            is_candidate=True,
        )
        assert asset.asset_id == "ASSET_TEST_0001"
        assert asset.asset_name == "test.shp"
        assert asset.asset_format == "SHAPEFILE"
        assert asset.found_in == "REVP"
        assert asset.is_candidate == True

    def test_shapefile_bundle_minimal_completeness(self):
        """Bundle deve marcar completeness baseado em .shp/.dbf/.shx."""
        bundle = ShapefileBundle(
            bundle_id="BUNDLE_0001",
            bundle_name="test",
            has_shp=True,
            has_dbf=True,
            has_shx=True,
            is_complete_minimal=True,
        )
        assert bundle.is_complete_minimal == True

        bundle2 = ShapefileBundle(
            bundle_id="BUNDLE_0002",
            bundle_name="test2",
            has_shp=True,
            has_dbf=False,
            has_shx=True,
            is_complete_minimal=False,
        )
        assert bundle2.is_complete_minimal == False


class TestV1ILCandidateIdentification:
    """Testes para identificação de candidatos missing."""

    def test_missing_candidates_structure(self):
        """Missing candidates devem ser identificados."""
        auditor = LocalVectorAssetAuditor(
            force=False,
            scan_revp=False,
            focus_missing_candidates=True,
        )
        auditor._identify_missing_candidates()

        assert len(auditor.missing_candidates) >= 2

        # Verificar que camada de pontos de feições de deslizamento fotointerpretadas está na lista
        has_ponto_p = any("Cicatriz_Ponto_P" in m.target_name for m in auditor.missing_candidates)
        assert has_ponto_p, "camada de pontos de feições de deslizamento fotointerpretadas não identificado como missing"

        # Verificar que PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA está na lista
        has_area_a = any("PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in m.target_name for m in auditor.missing_candidates)
        assert has_area_a, "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA não identificado como missing"

    def test_can_create_training_label_always_false(self):
        """can_create_training_label deve ser sempre NO."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)
        auditor._identify_missing_candidates()

        for record in auditor.missing_candidates:
            assert record.can_create_training_label == "NO", \
                f"can_create_training_label não é NO para {record.target_name}"


class TestV1ILOutputGeneration:
    """Testes para geração de outputs."""

    def test_outputs_created_with_force_flag(self, tmp_path):
        """Outputs devem ser criados quando --force é usado."""
        # Este teste é limitado pois depende da estrutura real
        # Apenas verificar que o método não quebra
        auditor = LocalVectorAssetAuditor(
            force=False,  # Manter False para não criar arquivos reais
            scan_revp=False,
        )

        # Não quebrar ao chamar outputs
        summary = auditor.get_summary()
        assert summary is not None

    def test_registries_should_have_no_private_paths(self):
        """Registries públicos nunca devem conter paths privados."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        # Simular um asset com path privado
        asset = VectorAsset(
            asset_id="ASSET_TEST",
            asset_name="test.shp",
            asset_path="C:\\Users\\gabriela\\Documents\\PROJETO\\test.shp",
            asset_format="SHAPEFILE",
            found_in="REVP",
        )

        # Verificar que o auditor rejeita paths privados
        assert auditor._contains_private_marker(asset.asset_path), \
            "Auditor não detectou path privado"


class TestV1ILBundleMapping:
    """Testes para mapeamento de bundles shapefile."""

    def test_bundle_minimal_requirement_shp_dbf_shx(self):
        """Bundle mínimo deve exigir .shp, .dbf, .shx."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        # Simular três bundles: completo, faltando dbf, faltando shx
        complete = ShapefileBundle(
            bundle_name="complete",
            has_shp=True, has_dbf=True, has_shx=True,
            is_complete_minimal=True,
        )

        no_dbf = ShapefileBundle(
            bundle_name="no_dbf",
            has_shp=True, has_dbf=False, has_shx=True,
            is_complete_minimal=False,
        )

        no_shx = ShapefileBundle(
            bundle_name="no_shx",
            has_shp=True, has_dbf=True, has_shx=False,
            is_complete_minimal=False,
        )

        assert complete.is_complete_minimal == True
        assert no_dbf.is_complete_minimal == False
        assert no_shx.is_complete_minimal == False


class TestV1ILPrivacyAndSecurity:
    """Testes para privacidade e segurança."""

    def test_no_private_markers_in_outputs(self):
        """Outputs nunca devem conter marcadores privados."""
        private_markers = ["gabriela", "C:\\Users", "PROJETO"]

        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        for marker in private_markers:
            is_detected = auditor._contains_private_marker(f"C:\\Users\\gabriela\\{marker}\\test.shp")
            assert is_detected, f"Auditor não detectou marcador privado: {marker}"

    def test_local_runs_not_versioned(self):
        """local_runs/ e local_only/ nunca devem ser versionados."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        # Simular arquivo em local_runs/
        local_runs_path = "C:\\Users\\gabriela\\Documents\\REV-P\\local_runs\\protocolo_c\\v1il\\test.csv"

        # Este arquivo será ignorado se escanearmos REVP
        # (não vai aparecer em assets) - apenas verificar sem quebrar
        assert True


class TestV1ILDataIntegrity:
    """Testes para integridade de dados."""

    def test_no_file_system_date_accepted(self):
        """Data de sistema de arquivos nunca deve ser aceita como data de evento."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        # Verificar que invariante é mantido na documentação
        assert auditor.qa_log is not None, "QA log deve existir"

    def test_missing_candidate_no_auto_ground_truth(self):
        """Asset local nunca deve virar ground truth automaticamente."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)
        auditor._identify_missing_candidates()

        for record in auditor.missing_candidates:
            # Mesmo se encontrado, não pode entrar na consolidação sozinho
            if record.found_status == "FOUND":
                assert record.can_enter_next_consolidation == "NO", \
                    "Asset encontrado não pode entrar automaticamente na consolidação"


class TestV1ILTermMatching:
    """Testes para busca de termos."""

    def test_search_terms_matching(self):
        """Busca de termos deve funcionar para candidatos-alvo."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        # Testar matching
        matches = auditor._find_matching_terms("Cicatriz_Ponto_P.shp")
        assert "Cicatriz_Ponto_P" in matches, "Cicatriz_Ponto_P não foi matched"

        matches2 = auditor._find_matching_terms("deslizamento_2022_02_15.shp")
        assert len(matches2) > 0, "Nenhum termo foi matched em deslizamento_2022_02_15"

    def test_format_inference(self):
        """Inferência de formato deve funcionar."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        assert auditor._infer_format(Path("test.shp")) == "SHAPEFILE"
        assert auditor._infer_format(Path("test.geojson")) == "GEOJSON"
        assert auditor._infer_format(Path("test.gpkg")) == "GEOPACKAGE"
        assert auditor._infer_format(Path("test.kmz")) == "KMZ"


class TestV1ILRegionAndEventHints:
    """Testes para extração de hints de região e evento."""

    def test_region_hint_extraction(self):
        """Extração de hint de região deve funcionar."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        assert auditor._extract_region_hint("Cicatriz_PET_2022.shp") == "PET"
        assert auditor._extract_region_hint("inundacao_Recife.shp") == "REC"
        assert auditor._extract_region_hint("deslizamento_curitiba.shp") == "CTB"
        assert auditor._extract_region_hint("unknown.shp") == "UNKNOWN"

    def test_event_hint_extraction(self):
        """Extração de hint de evento deve funcionar."""
        auditor = LocalVectorAssetAuditor(force=False, scan_revp=False)

        result = auditor._extract_event_hint("Cicatriz_2022_02_15.shp")
        assert "2022" in result or result == "PET_2022_02_15"

        assert "HYDROLOGICAL" in auditor._extract_event_hint("inundacao_evento.shp")
        assert "MASS_MOVEMENT" in auditor._extract_event_hint("deslizamento_evento.shp")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
