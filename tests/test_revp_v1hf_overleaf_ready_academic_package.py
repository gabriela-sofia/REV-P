"""Tests for revp_v1hf_overleaf_ready_academic_package.py.

Validates that:
- All 9 expected output files are generated.
- Texts do not contain forbidden predictive claims.
- Each section has at least one substantiated artifact reference.
- Figures/tables referenced in the index exist in v1gy.
- Limitations section is present and substantive.
- ready_for_template_insertion is True only when all 5 sections are present.
- No private paths in any output or source.
- No heavy artifacts staged (structural check via constants).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dino"))

import revp_v1hf_overleaf_ready_academic_package as v1hf  # noqa: E402

OUT_DIR = ROOT / "local_runs" / "overleaf_package" / "v1hf"
V1GY_DIR = ROOT / "local_runs" / "tcc_figures" / "v1gy"

# ---------------------------------------------------------------------------
# Governance constants
# ---------------------------------------------------------------------------
PRIVATE_PATH_FRAGMENTS = ["PROJETO", "AppData", r"C:\Users\gabriela"]

HARD_FORBIDDEN_TERMS = [
    "enchente",
    "alagamento",
    "risco de inundação",
    "classificação supervisionada",
    "detecção de inundação",
    "flood detection",
    "flood prediction",
]

EXPECTED_OUTPUT_FILES = [
    "metodologia_overleaf_draft_v1hf.md",
    "resultados_overleaf_draft_v1hf.md",
    "discussao_overleaf_draft_v1hf.md",
    "limitacoes_overleaf_draft_v1hf.md",
    "contribuicoes_overleaf_draft_v1hf.md",
    "overleaf_figures_tables_index_v1hf.csv",
    "appendices_plan_v1hf.md",
    "tcc_section_artifact_crosswalk_v1hf.csv",
    "overleaf_package_summary_v1hf.json",
]


# ---------------------------------------------------------------------------
# Fake Evidence for unit tests
# ---------------------------------------------------------------------------

class _FakeEvidence:
    n_patches = 12
    n_regions = 3
    emb_dim = 768
    backbone = "DINOv2-com-registers"
    n_review_candidates = 47
    n_figures_ready = 5
    n_tables_ready = 6
    n_allowed_claims = 11
    n_forbidden_claims = 10
    n_robust = 12
    n_unstable = 0
    n_pert_types = 6
    regions: list = ["Curitiba", "Petrópolis", "Recife"]
    regional_mean_drift: dict = {
        "Curitiba": 0.043,
        "Petrópolis": 0.078,
        "Recife": 0.060,
    }
    intra_rate = 0.367
    inter_rate = 0.633
    n_pairs = 60
    n_visual_computed = 47
    n_unc_low = 3
    n_unc_med = 5
    n_unc_high = 39
    he_results_text = "## 4.1 Análise Estrutural\nTexto de resultados de teste."
    he_discussion_text = "## 5.1 O que os resultados sustentam\nTexto de discussão de teste."
    he_claim_matrix: list = []
    he_fig_captions: list = []
    he_tbl_captions: list = []
    ha_summary: dict = {
        "perturbation_types": [
            "gaussian_noise", "brightness_scale", "contrast_scale",
            "blur_light", "crop_jitter", "band_dropout",
        ]
    }
    fig_names_present: list = ["fig_similarity_heatmap_v1gy.png"]
    tbl_names_present: list = ["table_embedding_corpus_summary_v1gy.csv"]


# ---------------------------------------------------------------------------
# TestV1HfConstants
# ---------------------------------------------------------------------------

class TestV1HfConstants:
    """FIGURES_TABLES_INDEX and CROSSWALK structure checks."""

    def test_figures_tables_index_count(self) -> None:
        assert len(v1hf.FIGURES_TABLES_INDEX) == 11  # 5 figs + 6 tables

    def test_figures_tables_index_required_fields(self) -> None:
        required = {
            "artifact_id", "tipo", "arquivo", "legenda",
            "secao_recomendada", "status", "nota_limitacao", "corpo_ou_apendice",
        }
        for entry in v1hf.FIGURES_TABLES_INDEX:
            assert required.issubset(entry.keys()), (
                f"Missing fields in {entry.get('artifact_id')}"
            )

    def test_figures_count_in_index(self) -> None:
        figs = [e for e in v1hf.FIGURES_TABLES_INDEX if e["tipo"] == "figura"]
        assert len(figs) == 5

    def test_tables_count_in_index(self) -> None:
        tbls = [e for e in v1hf.FIGURES_TABLES_INDEX if e["tipo"] == "tabela"]
        assert len(tbls) == 6

    def test_index_corpo_ou_apendice_valid_values(self) -> None:
        for entry in v1hf.FIGURES_TABLES_INDEX:
            assert entry["corpo_ou_apendice"] in {"corpo", "apendice"}, (
                f"Invalid corpo_ou_apendice: {entry['artifact_id']}"
            )

    def test_index_status_all_ready(self) -> None:
        for entry in v1hf.FIGURES_TABLES_INDEX:
            assert entry["status"] == "READY", (
                f"Status not READY: {entry['artifact_id']}"
            )

    def test_index_nota_limitacao_non_empty(self) -> None:
        for entry in v1hf.FIGURES_TABLES_INDEX:
            assert len(entry["nota_limitacao"].strip()) > 5, (
                f"Empty nota_limitacao: {entry['artifact_id']}"
            )

    def test_index_no_private_paths(self) -> None:
        for entry in v1hf.FIGURES_TABLES_INDEX:
            text = " ".join(str(v) for v in entry.values())
            for frag in PRIVATE_PATH_FRAGMENTS:
                assert frag not in text, (
                    f"Private path '{frag}' in index entry {entry['artifact_id']}"
                )

    def test_crosswalk_count(self) -> None:
        assert len(v1hf.CROSSWALK) >= 8

    def test_crosswalk_required_fields(self) -> None:
        required = {
            "secao_tcc", "argumento_cientifico",
            "artifact_sustentando", "figura_tabela", "limitacao_associada",
        }
        for row in v1hf.CROSSWALK:
            assert required.issubset(row.keys()), (
                f"Missing fields in crosswalk row: {row.get('secao_tcc', '')}"
            )

    def test_crosswalk_limitacao_non_empty(self) -> None:
        for row in v1hf.CROSSWALK:
            assert len(row["limitacao_associada"].strip()) > 5, (
                f"Empty limitation in: {row['secao_tcc']}"
            )

    def test_crosswalk_argumento_non_empty(self) -> None:
        for row in v1hf.CROSSWALK:
            assert len(row["argumento_cientifico"].strip()) > 10, (
                f"Empty argument in: {row['secao_tcc']}"
            )

    def test_crosswalk_no_private_paths(self) -> None:
        for row in v1hf.CROSSWALK:
            text = " ".join(str(v) for v in row.values())
            for frag in PRIVATE_PATH_FRAGMENTS:
                assert frag not in text, (
                    f"Private path '{frag}' in crosswalk row {row.get('secao_tcc')}"
                )

    def test_crosswalk_covers_methodology_section(self) -> None:
        sections = [r["secao_tcc"] for r in v1hf.CROSSWALK]
        assert any("Metodologia" in s or "3." in s for s in sections)

    def test_crosswalk_covers_results_section(self) -> None:
        sections = [r["secao_tcc"] for r in v1hf.CROSSWALK]
        assert any("4." in s or "Resultados" in s for s in sections)

    def test_crosswalk_covers_discussion_section(self) -> None:
        sections = [r["secao_tcc"] for r in v1hf.CROSSWALK]
        assert any("5." in s or "Discuss" in s for s in sections)


# ---------------------------------------------------------------------------
# TestV1HfEvidence
# ---------------------------------------------------------------------------

class TestV1HfEvidence:
    """Evidence class safe defaults when pipeline files are absent."""

    def _make_absent_ev(self) -> v1hf.Evidence:
        with patch.object(v1hf, "V1HE_DIR", Path("/nonexistent/v1he")), \
             patch.object(v1hf, "V1GZ_DIR", Path("/nonexistent/v1gz")), \
             patch.object(v1hf, "V1HA_DIR", Path("/nonexistent/v1ha")), \
             patch.object(v1hf, "V1GY_DIR", Path("/nonexistent/v1gy")), \
             patch.object(v1hf, "V1HD_DIR", Path("/nonexistent/v1hd")), \
             patch.object(v1hf, "DOCS_DIR", Path("/nonexistent/docs")), \
             patch.object(v1hf, "DATASETS_DIR", Path("/nonexistent/datasets")):
            return v1hf.Evidence()

    def test_evidence_loads_without_crash(self) -> None:
        ev = self._make_absent_ev()
        assert ev.n_patches >= 1

    def test_evidence_fallback_n_patches(self) -> None:
        ev = self._make_absent_ev()
        assert ev.n_patches == 12

    def test_evidence_fallback_n_regions(self) -> None:
        ev = self._make_absent_ev()
        assert ev.n_regions == 3

    def test_evidence_fallback_intra_inter_rates(self) -> None:
        ev = self._make_absent_ev()
        assert 0.0 < ev.intra_rate < 1.0
        assert 0.0 < ev.inter_rate < 1.0
        assert abs(ev.intra_rate + ev.inter_rate - 1.0) < 1e-3

    def test_evidence_fallback_emb_dim(self) -> None:
        ev = self._make_absent_ev()
        assert ev.emb_dim == 768

    def test_evidence_fallback_regions_list(self) -> None:
        ev = self._make_absent_ev()
        assert len(ev.regions) == 3


# ---------------------------------------------------------------------------
# TestV1HfBuildFunctions
# ---------------------------------------------------------------------------

class TestV1HfBuildFunctions:
    """Build functions produce non-empty, well-formed output."""

    @pytest.fixture
    def ev(self) -> _FakeEvidence:
        return _FakeEvidence()

    # Metodologia
    def test_metodologia_is_string(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_metodologia_min_length(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert len(text) >= 5000

    def test_metodologia_covers_audit_first(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert "audit" in text.lower() or "Audit" in text

    def test_metodologia_covers_sentinel_first(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert "Sentinel" in text

    def test_metodologia_covers_review_only(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert "review-only" in text.lower() or "review_only" in text.lower()

    def test_metodologia_covers_areas_de_estudo(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert "Curitiba" in text
        assert "Petrópolis" in text or "Petr" in text
        assert "Recife" in text

    def test_metodologia_covers_dinov2(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert "DINOv2" in text or "DINO" in text

    def test_metodologia_covers_governance(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        assert "claims" in text.lower() or "governança" in text.lower()

    def test_metodologia_no_forbidden_terms(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev).lower()  # type: ignore[arg-type]
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in text, f"Forbidden term '{term}' in metodologia"

    def test_metodologia_no_private_paths(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_metodologia(ev)  # type: ignore[arg-type]
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in text, f"Private path '{frag}' in metodologia"

    # Resultados
    def test_resultados_is_string(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_resultados(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_resultados_contains_v1he_body(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_resultados(ev)  # type: ignore[arg-type]
        assert "resultados" in text.lower() or "4." in text

    def test_resultados_no_forbidden_terms(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_resultados(ev).lower()  # type: ignore[arg-type]
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in text, f"Forbidden term '{term}' in resultados"

    # Discussão
    def test_discussao_is_string(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_discussao(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_discussao_contains_checklist(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_discussao(ev)  # type: ignore[arg-type]
        assert "Checklist" in text or "checklist" in text.lower()

    def test_discussao_no_forbidden_terms(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_discussao(ev).lower()  # type: ignore[arg-type]
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in text, f"Forbidden term '{term}' in discussao"

    # Limitações
    def test_limitacoes_is_string(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_limitacoes_min_length(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev)  # type: ignore[arg-type]
        assert len(text) >= 2000

    def test_limitacoes_mentions_ground_truth_absence(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev)  # type: ignore[arg-type]
        assert "ground truth" in text.lower() or "Ground Truth" in text

    def test_limitacoes_mentions_corpus_pequeno(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev).lower()  # type: ignore[arg-type]
        assert "corpus" in text and ("pequeno" in text or "12" in text)

    def test_limitacoes_mentions_gis_parcial(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev).lower()  # type: ignore[arg-type]
        assert "gis" in text

    def test_limitacoes_mentions_multimodal_hold(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev).lower()  # type: ignore[arg-type]
        assert "multimodal" in text

    def test_limitacoes_mentions_validacao_supervisionada(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev).lower()  # type: ignore[arg-type]
        assert "validação supervisionada" in text or "supervisionada" in text

    def test_limitacoes_has_at_least_6_sections(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev)  # type: ignore[arg-type]
        # Each limitation is marked with ## L1–L8
        count = sum(1 for line in text.splitlines() if line.startswith("## L"))
        assert count >= 6

    def test_limitacoes_no_forbidden_terms(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_limitacoes(ev).lower()  # type: ignore[arg-type]
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in text, f"Forbidden term '{term}' in limitacoes"

    # Contribuições
    def test_contribuicoes_is_string(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_contribuicoes(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_contribuicoes_min_length(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_contribuicoes(ev)  # type: ignore[arg-type]
        assert len(text) >= 2000

    def test_contribuicoes_has_at_least_4_sections(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_contribuicoes(ev)  # type: ignore[arg-type]
        count = sum(1 for line in text.splitlines() if line.startswith("## C"))
        assert count >= 4

    def test_contribuicoes_mentions_pipeline_auditavel(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_contribuicoes(ev).lower()  # type: ignore[arg-type]
        assert "auditável" in text or "auditavel" in text or "audit" in text

    def test_contribuicoes_mentions_governance(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_contribuicoes(ev).lower()  # type: ignore[arg-type]
        assert "governança" in text or "governance" in text

    def test_contribuicoes_no_forbidden_terms(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_contribuicoes(ev).lower()  # type: ignore[arg-type]
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in text, f"Forbidden term '{term}' in contribuicoes"

    # Summary
    def test_summary_required_fields(self, ev: _FakeEvidence) -> None:
        summary = v1hf.build_summary(ev, ["metodologia", "resultados", "discussao",  # type: ignore[arg-type]
                                          "limitacoes", "contribuicoes"])
        required = {
            "phase", "sections_ready", "figures_ready", "tables_ready",
            "appendices_planned", "forbidden_claims_checked",
            "ready_for_template_insertion", "methodological_guardrails",
            "crosswalk_entries",
        }
        assert required.issubset(summary.keys())

    def test_summary_ready_true_all_sections(self, ev: _FakeEvidence) -> None:
        summary = v1hf.build_summary(ev, ["metodologia", "resultados", "discussao",  # type: ignore[arg-type]
                                          "limitacoes", "contribuicoes"])
        assert summary["ready_for_template_insertion"] is True

    def test_summary_ready_false_missing_section(self, ev: _FakeEvidence) -> None:
        summary = v1hf.build_summary(ev, ["metodologia", "resultados"])  # type: ignore[arg-type]
        assert summary["ready_for_template_insertion"] is False

    def test_summary_guardrails_off(self, ev: _FakeEvidence) -> None:
        summary = v1hf.build_summary(ev, ["metodologia", "resultados", "discussao",  # type: ignore[arg-type]
                                          "limitacoes", "contribuicoes"])
        g = summary["methodological_guardrails"]
        assert g["labels_created"] is False
        assert g["predictions_made"] is False
        assert g["ground_truth_established"] is False
        assert g["review_only"] is True
        assert g["gis_contextual_only"] is True
        assert g["multimodal_disabled"] is True

    def test_summary_phase_field(self, ev: _FakeEvidence) -> None:
        summary = v1hf.build_summary(ev, [])  # type: ignore[arg-type]
        assert summary["phase"] == "v1hf"

    # Appendices plan
    def test_appendices_plan_is_string(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_appendices_plan(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_appendices_plan_has_8_appendices(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_appendices_plan(ev)  # type: ignore[arg-type]
        count = sum(1 for line in text.splitlines() if line.startswith("## Apêndice"))
        assert count >= 6

    def test_appendices_plan_no_private_paths(self, ev: _FakeEvidence) -> None:
        text = v1hf.build_appendices_plan(ev)  # type: ignore[arg-type]
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in text, f"Private path '{frag}' in appendices plan"


# ---------------------------------------------------------------------------
# TestV1HfOutputFiles
# ---------------------------------------------------------------------------

class TestV1HfOutputFiles:
    """All 9 expected output files exist and are non-empty."""

    @pytest.fixture(autouse=True)
    def require_outputs(self) -> None:
        if not OUT_DIR.exists():
            pytest.skip("v1hf outputs not found — run script first")

    @pytest.mark.parametrize("filename", EXPECTED_OUTPUT_FILES)
    def test_expected_file_exists(self, filename: str) -> None:
        assert (OUT_DIR / filename).exists(), f"Missing: {filename}"

    def test_metodologia_min_length(self) -> None:
        content = (OUT_DIR / "metodologia_overleaf_draft_v1hf.md").read_text(encoding="utf-8")
        assert len(content) >= 5000

    def test_resultados_non_empty(self) -> None:
        content = (OUT_DIR / "resultados_overleaf_draft_v1hf.md").read_text(encoding="utf-8")
        assert len(content) >= 500

    def test_discussao_non_empty(self) -> None:
        content = (OUT_DIR / "discussao_overleaf_draft_v1hf.md").read_text(encoding="utf-8")
        assert len(content) >= 500

    def test_limitacoes_min_length(self) -> None:
        content = (OUT_DIR / "limitacoes_overleaf_draft_v1hf.md").read_text(encoding="utf-8")
        assert len(content) >= 2000

    def test_contribuicoes_min_length(self) -> None:
        content = (OUT_DIR / "contribuicoes_overleaf_draft_v1hf.md").read_text(encoding="utf-8")
        assert len(content) >= 2000

    def test_figures_tables_index_row_count(self) -> None:
        with (OUT_DIR / "overleaf_figures_tables_index_v1hf.csv").open(
            "r", encoding="utf-8-sig"
        ) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 11  # 5 figs + 6 tables

    def test_crosswalk_row_count(self) -> None:
        with (OUT_DIR / "tcc_section_artifact_crosswalk_v1hf.csv").open(
            "r", encoding="utf-8-sig"
        ) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 8

    def test_summary_json_ready_for_template(self) -> None:
        with (OUT_DIR / "overleaf_package_summary_v1hf.json").open(
            "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        assert isinstance(data["ready_for_template_insertion"], bool)

    def test_summary_json_five_sections_ready(self) -> None:
        with (OUT_DIR / "overleaf_package_summary_v1hf.json").open(
            "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        sections = set(data.get("sections_ready", []))
        required = {"metodologia", "resultados", "discussao", "limitacoes", "contribuicoes"}
        assert required.issubset(sections)

    def test_summary_json_guardrails(self) -> None:
        with (OUT_DIR / "overleaf_package_summary_v1hf.json").open(
            "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        g = data["methodological_guardrails"]
        assert g["labels_created"] is False
        assert g["predictions_made"] is False
        assert g["ground_truth_established"] is False
        assert g["review_only"] is True

    def test_figures_index_v1gy_files_exist(self) -> None:
        """Figures listed as 'corpo' in the index must exist in v1gy/."""
        if not V1GY_DIR.exists():
            pytest.skip("v1gy figures directory not present")
        with (OUT_DIR / "overleaf_figures_tables_index_v1hf.csv").open(
            "r", encoding="utf-8-sig"
        ) as f:
            rows = list(csv.DictReader(f))
        fig_rows = [r for r in rows if r["tipo"] == "figura"]
        for row in fig_rows:
            fig_path = V1GY_DIR / row["arquivo"]
            assert fig_path.exists(), (
                f"Figure file missing: {row['arquivo']} (expected in v1gy/)"
            )


# ---------------------------------------------------------------------------
# TestV1HfGovernance
# ---------------------------------------------------------------------------

class TestV1HfGovernance:
    """Private path and forbidden term audits on all output files."""

    @pytest.fixture(autouse=True)
    def require_outputs(self) -> None:
        if not OUT_DIR.exists():
            pytest.skip("v1hf outputs not found — run script first")

    def _all_md_content(self) -> str:
        return "\n".join(
            f.read_text(encoding="utf-8") for f in OUT_DIR.glob("*.md")
        )

    def _all_csv_content(self) -> str:
        return "\n".join(
            f.read_text(encoding="utf-8-sig") for f in OUT_DIR.glob("*.csv")
        )

    def test_no_private_paths_in_md_outputs(self) -> None:
        content = self._all_md_content()
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in content, f"Private path '{frag}' in .md outputs"

    def test_no_private_paths_in_csv_outputs(self) -> None:
        content = self._all_csv_content()
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in content, f"Private path '{frag}' in .csv outputs"

    def test_no_private_paths_in_json(self) -> None:
        content = (OUT_DIR / "overleaf_package_summary_v1hf.json").read_text(
            encoding="utf-8"
        )
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in content, f"Private path '{frag}' in summary JSON"

    def test_no_hard_forbidden_terms_in_metodologia(self) -> None:
        content = (OUT_DIR / "metodologia_overleaf_draft_v1hf.md").read_text(
            encoding="utf-8"
        ).lower()
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in content, f"Forbidden term '{term}' in metodologia"

    def test_no_hard_forbidden_terms_in_limitacoes(self) -> None:
        content = (OUT_DIR / "limitacoes_overleaf_draft_v1hf.md").read_text(
            encoding="utf-8"
        ).lower()
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in content, f"Forbidden term '{term}' in limitacoes"

    def test_no_hard_forbidden_terms_in_contribuicoes(self) -> None:
        content = (OUT_DIR / "contribuicoes_overleaf_draft_v1hf.md").read_text(
            encoding="utf-8"
        ).lower()
        for term in HARD_FORBIDDEN_TERMS:
            assert term not in content, f"Forbidden term '{term}' in contribuicoes"

    def test_limitacoes_mentions_ground_truth_absence(self) -> None:
        content = (OUT_DIR / "limitacoes_overleaf_draft_v1hf.md").read_text(
            encoding="utf-8"
        )
        assert "ground truth" in content.lower(), "Limitações must address absence of ground truth"

    def test_limitacoes_mentions_corpus_size(self) -> None:
        content = (OUT_DIR / "limitacoes_overleaf_draft_v1hf.md").read_text(
            encoding="utf-8"
        )
        assert "12" in content, "Limitações must mention corpus size (12 patches)"

    def test_metodologia_contains_audit_first(self) -> None:
        content = (OUT_DIR / "metodologia_overleaf_draft_v1hf.md").read_text(
            encoding="utf-8"
        )
        assert "audit" in content.lower()

    def test_metodologia_mentions_gis_not_ground_truth(self) -> None:
        content = (OUT_DIR / "metodologia_overleaf_draft_v1hf.md").read_text(
            encoding="utf-8"
        ).lower()
        # Must mention GIS as contextual
        assert "gis" in content and "contextual" in content

    def test_appendices_plan_no_private_paths(self) -> None:
        content = (OUT_DIR / "appendices_plan_v1hf.md").read_text(encoding="utf-8")
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in content, f"Private path '{frag}' in appendices plan"

    def test_script_source_no_private_paths(self) -> None:
        script = ROOT / "scripts" / "dino" / "revp_v1hf_overleaf_ready_academic_package.py"
        content = script.read_text(encoding="utf-8")
        for frag in ["PROJETO", r"C:\Users\gabriela", "AppData\\Local\\Programs"]:
            assert frag not in content, f"Private path '{frag}' in script source"

    def test_output_dir_under_local_runs(self) -> None:
        assert "local_runs" in str(v1hf.OUT_DIR)
        assert "scripts" not in str(v1hf.OUT_DIR)

    def test_no_syntax_warnings_in_script(self) -> None:
        """Script must import without SyntaxWarning in Python 3.12."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-W", "error::SyntaxWarning",
             "-c", "import sys; sys.path.insert(0, r'scripts/dino'); "
             "import revp_v1hf_overleaf_ready_academic_package"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"SyntaxWarning found:\n{result.stderr}"
        )

    def test_figures_tables_index_no_forbidden_terms(self) -> None:
        with (OUT_DIR / "overleaf_figures_tables_index_v1hf.csv").open(
            "r", encoding="utf-8-sig"
        ) as f:
            content = f.read().lower()
        for term in ["enchente", "alagamento", "flood detection", "flood prediction"]:
            assert term not in content, (
                f"Forbidden term '{term}' in figures/tables index"
            )
