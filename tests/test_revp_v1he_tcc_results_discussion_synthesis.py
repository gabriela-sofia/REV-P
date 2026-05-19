"""Tests for revp_v1he_tcc_results_discussion_synthesis.py.

Validates that:
- Constants (FIGURE_CAPTIONS, TABLE_CAPTIONS, CLAIM_MATRIX) have correct structure.
- Evidence class loads with safe defaults when pipeline outputs are absent.
- Build functions produce non-empty text grounded in real numbers.
- Summary JSON contains required fields and correct ready_for_overleaf logic.
- Output files exist, are non-empty, and pass governance audits:
    * No private paths (PROJETO, gabriela, Documents, AppData)
    * No forbidden predictive claims
    * Methodological guardrails enforced
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

import revp_v1he_tcc_results_discussion_synthesis as v1he  # noqa: E402

OUT_DIR = ROOT / "local_runs" / "tcc_synthesis" / "v1he"

# ---------------------------------------------------------------------------
# Private path fragments to ban from versionable outputs
# ---------------------------------------------------------------------------
PRIVATE_PATH_FRAGMENTS = ["PROJETO", "gabriela", "AppData", "Documents\\REV-P"]

# Hard forbidden predictive terms (in output content, not code)
HARD_FORBIDDEN_OUTPUT = [
    "enchente",
    "alagamento",
    "risco de inundação",
    "predição",
    "classificação supervisionada",
    "ground truth",
    "detecção de inundação",
    "label",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEvidence:
    """Minimal Evidence-like object for testing build functions in isolation."""
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
    n_perturbation_types = 6
    n_neighbor_pairs = 60
    n_intra = 22
    n_inter = 38
    intra_rate = 0.367
    inter_rate = 0.633
    n_visual_computed = 47
    hd_uncertainty: dict = {"low": 3, "medium": 5, "high": 39}
    hd_usable: dict = {"yes": 0, "conditional": 47, "no": 0}
    regional_mean_drift: dict = {
        "Curitiba": 0.043,
        "Petrópolis": 0.078,
        "Recife": 0.060,
    }
    gy_medoids: list = [
        {"regiao": "Curitiba", "medoid": "CUR_00357", "outliers": "CUR_00350"},
        {"regiao": "Petrópolis", "medoid": "PET_00104", "outliers": "PET_00016"},
        {"regiao": "Recife", "medoid": "REC_00205", "outliers": "REC_00019"},
    ]
    ha_summary: dict = {"perturbation_types": ["noise", "flip", "rotate", "crop", "blur", "band_drop"]}
    gz_summary: dict = {
        "allowed_claims_count": 11,
        "forbidden_claims_count": 10,
        "corpus_size": 12,
    }
    hb_manifest: list = []
    hc_summary: dict = {}
    hd_summary: dict = {}
    hd_examples: list = []
    gy_corpus: list = []
    gy_coverage: list = []
    gy_review_cats: list = []


# ---------------------------------------------------------------------------
# TestV1HeConstants
# ---------------------------------------------------------------------------

class TestV1HeConstants:
    """FIGURE_CAPTIONS, TABLE_CAPTIONS and CLAIM_MATRIX structure checks."""

    def test_figure_captions_count(self) -> None:
        assert len(v1he.FIGURE_CAPTIONS) == 5

    def test_figure_captions_required_fields(self) -> None:
        required = {"figure_id", "filename", "caption_pt", "tcc_section",
                    "claim_scope", "limitation_note", "forbidden_terms_checked"}
        for fig in v1he.FIGURE_CAPTIONS:
            assert required.issubset(fig.keys()), f"Missing fields in {fig.get('figure_id')}"

    def test_figure_captions_forbidden_terms_checked_yes(self) -> None:
        for fig in v1he.FIGURE_CAPTIONS:
            assert fig["forbidden_terms_checked"] == "yes", (
                f"{fig['figure_id']} does not mark forbidden_terms_checked=yes"
            )

    def test_figure_captions_no_private_paths(self) -> None:
        for fig in v1he.FIGURE_CAPTIONS:
            text = " ".join(str(v) for v in fig.values())
            for frag in PRIVATE_PATH_FRAGMENTS:
                assert frag not in text, f"Private path '{frag}' in figure caption {fig['figure_id']}"

    def test_table_captions_count(self) -> None:
        assert len(v1he.TABLE_CAPTIONS) == 6

    def test_table_captions_required_fields(self) -> None:
        required = {"table_id", "filename", "caption_pt", "tcc_section",
                    "claim_scope", "limitation_note"}
        for tbl in v1he.TABLE_CAPTIONS:
            assert required.issubset(tbl.keys()), f"Missing fields in {tbl.get('table_id')}"

    def test_table_captions_no_private_paths(self) -> None:
        for tbl in v1he.TABLE_CAPTIONS:
            text = " ".join(str(v) for v in tbl.values())
            for frag in PRIVATE_PATH_FRAGMENTS:
                assert frag not in text, f"Private path '{frag}' in table caption {tbl['table_id']}"

    def test_claim_matrix_count(self) -> None:
        assert len(v1he.CLAIM_MATRIX) == 10

    def test_claim_matrix_required_fields(self) -> None:
        required = {"claim_allowed", "result_supporting_it", "evidence_artifact",
                    "figure_or_table", "limitation", "blocked_overclaim"}
        for row in v1he.CLAIM_MATRIX:
            assert required.issubset(row.keys()), f"Missing fields in claim: {row.get('claim_allowed', '')[:40]}"

    def test_claim_matrix_blocked_overclaim_starts_with_nao(self) -> None:
        for row in v1he.CLAIM_MATRIX:
            assert row["blocked_overclaim"].startswith("NÃO"), (
                f"blocked_overclaim must start with 'NÃO': {row['blocked_overclaim'][:50]}"
            )

    def test_claim_matrix_no_private_paths(self) -> None:
        for row in v1he.CLAIM_MATRIX:
            text = " ".join(str(v) for v in row.values())
            for frag in PRIVATE_PATH_FRAGMENTS:
                assert frag not in text, f"Private path '{frag}' in claim matrix"

    def test_claim_matrix_limitation_non_empty(self) -> None:
        for row in v1he.CLAIM_MATRIX:
            assert len(row["limitation"].strip()) > 5, (
                f"Limitation too short for: {row['claim_allowed'][:40]}"
            )


# ---------------------------------------------------------------------------
# TestV1HeEvidence
# ---------------------------------------------------------------------------

class TestV1HeEvidence:
    """Evidence class safe defaults when pipeline files are absent."""

    def test_evidence_loads_without_crash(self) -> None:
        """Evidence must not raise even if all input files are missing."""
        with patch.object(v1he, "V1GZ_DIR", Path("/nonexistent/v1gz")), \
             patch.object(v1he, "V1HA_DIR", Path("/nonexistent/v1ha")), \
             patch.object(v1he, "V1GY_DIR", Path("/nonexistent/v1gy")), \
             patch.object(v1he, "V1HB_DIR", Path("/nonexistent/v1hb")), \
             patch.object(v1he, "V1HC_DIR", Path("/nonexistent/v1hc")), \
             patch.object(v1he, "V1HD_DIR", Path("/nonexistent/v1hd")):
            ev = v1he.Evidence()
        assert ev.n_patches >= 1  # defaults, not zero

    def test_evidence_fallback_n_patches(self) -> None:
        with patch.object(v1he, "V1GZ_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HA_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1GY_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HB_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HC_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HD_DIR", Path("/nonexistent")):
            ev = v1he.Evidence()
        assert ev.n_patches == 12  # hard-coded safe default

    def test_evidence_fallback_n_regions(self) -> None:
        with patch.object(v1he, "V1GZ_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HA_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1GY_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HB_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HC_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HD_DIR", Path("/nonexistent")):
            ev = v1he.Evidence()
        assert ev.n_regions == 3

    def test_evidence_fallback_intra_rate(self) -> None:
        with patch.object(v1he, "V1GZ_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HA_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1GY_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HB_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HC_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HD_DIR", Path("/nonexistent")):
            ev = v1he.Evidence()
        assert 0.0 < ev.intra_rate < 1.0
        assert 0.0 < ev.inter_rate < 1.0

    def test_evidence_intra_inter_sum_to_one(self) -> None:
        with patch.object(v1he, "V1GZ_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HA_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1GY_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HB_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HC_DIR", Path("/nonexistent")), \
             patch.object(v1he, "V1HD_DIR", Path("/nonexistent")):
            ev = v1he.Evidence()
        assert abs(ev.intra_rate + ev.inter_rate - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestV1HeBuildFunctions
# ---------------------------------------------------------------------------

class TestV1HeBuildFunctions:
    """Build functions produce non-empty, well-formed output."""

    @pytest.fixture
    def ev(self) -> _FakeEvidence:
        return _FakeEvidence()

    def test_results_section_is_string(self, ev: _FakeEvidence) -> None:
        text = v1he.build_results_section(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_results_section_min_length(self, ev: _FakeEvidence) -> None:
        text = v1he.build_results_section(ev)  # type: ignore[arg-type]
        assert len(text) >= 3000

    def test_results_section_contains_patch_count(self, ev: _FakeEvidence) -> None:
        text = v1he.build_results_section(ev)  # type: ignore[arg-type]
        assert "12" in text

    def test_results_section_contains_inter_rate(self, ev: _FakeEvidence) -> None:
        text = v1he.build_results_section(ev)  # type: ignore[arg-type]
        assert "63" in text  # 63.3% inter rate

    def test_results_section_starts_with_section_header(self, ev: _FakeEvidence) -> None:
        text = v1he.build_results_section(ev)  # type: ignore[arg-type]
        assert "# Seção 4" in text or "## 4.1" in text

    def test_results_section_no_forbidden_terms(self, ev: _FakeEvidence) -> None:
        text = v1he.build_results_section(ev).lower()  # type: ignore[arg-type]
        for term in ["enchente", "alagamento", "risco de inundação",
                     "classificação supervisionada", "detecção de inundação"]:
            assert term not in text, f"Forbidden term '{term}' in results draft"

    def test_discussion_section_is_string(self, ev: _FakeEvidence) -> None:
        text = v1he.build_discussion_section(ev)  # type: ignore[arg-type]
        assert isinstance(text, str)

    def test_discussion_section_min_length(self, ev: _FakeEvidence) -> None:
        text = v1he.build_discussion_section(ev)  # type: ignore[arg-type]
        assert len(text) >= 3000

    def test_discussion_section_has_limitations(self, ev: _FakeEvidence) -> None:
        text = v1he.build_discussion_section(ev)  # type: ignore[arg-type]
        assert "limitaç" in text.lower() or "Limitações" in text

    def test_discussion_section_no_forbidden_terms(self, ev: _FakeEvidence) -> None:
        text = v1he.build_discussion_section(ev).lower()  # type: ignore[arg-type]
        for term in ["enchente", "alagamento", "risco de inundação",
                     "classificação supervisionada", "detecção de inundação"]:
            assert term not in text, f"Forbidden term '{term}' in discussion draft"

    def test_build_summary_required_fields(self, ev: _FakeEvidence) -> None:
        summary = v1he.build_summary(ev)  # type: ignore[arg-type]
        required = {"phase", "ready_for_overleaf", "total_figures_ready",
                    "total_tables_ready", "total_claims_ready",
                    "total_forbidden_claims_blocked", "methodological_guardrails",
                    "corpus", "neighbor_topology"}
        assert required.issubset(summary.keys())

    def test_build_summary_guardrails_off(self, ev: _FakeEvidence) -> None:
        summary = v1he.build_summary(ev)  # type: ignore[arg-type]
        guards = summary["methodological_guardrails"]
        assert guards["labels_created"] is False
        assert guards["predictions_made"] is False
        assert guards["ground_truth_established"] is False
        assert guards["review_only"] is True
        assert guards["all_forbidden_claims_blocked"] is True

    def test_build_summary_ready_for_overleaf_true(self, ev: _FakeEvidence) -> None:
        summary = v1he.build_summary(ev)  # type: ignore[arg-type]
        assert summary["ready_for_overleaf"] is True

    def test_build_summary_ready_for_overleaf_false_few_figures(self, ev: _FakeEvidence) -> None:
        ev.n_figures_ready = 2
        summary = v1he.build_summary(ev)  # type: ignore[arg-type]
        assert summary["ready_for_overleaf"] is False

    def test_build_summary_ready_for_overleaf_false_unstable(self, ev: _FakeEvidence) -> None:
        ev.n_unstable = 1
        ev.n_robust = 11  # one less than n_patches=12 → triggers not-ready
        summary = v1he.build_summary(ev)  # type: ignore[arg-type]
        assert summary["ready_for_overleaf"] is False

    def test_build_overleaf_plan_is_string(self, ev: _FakeEvidence) -> None:
        plan = v1he.build_overleaf_plan(ev)  # type: ignore[arg-type]
        assert isinstance(plan, str)

    def test_build_overleaf_plan_min_length(self, ev: _FakeEvidence) -> None:
        plan = v1he.build_overleaf_plan(ev)  # type: ignore[arg-type]
        assert len(plan) >= 2000

    def test_build_overleaf_plan_uses_ev_data(self, ev: _FakeEvidence) -> None:
        plan = v1he.build_overleaf_plan(ev)  # type: ignore[arg-type]
        assert "12" in plan  # n_patches
        assert "768" in plan  # emb_dim

    def test_build_overleaf_plan_no_private_paths(self, ev: _FakeEvidence) -> None:
        plan = v1he.build_overleaf_plan(ev)  # type: ignore[arg-type]
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in plan, f"Private path '{frag}' in overleaf plan"


# ---------------------------------------------------------------------------
# TestV1HeOutputFiles
# ---------------------------------------------------------------------------

class TestV1HeOutputFiles:
    """Output files must exist, be non-empty, and pass content checks."""

    @pytest.fixture(autouse=True)
    def require_outputs(self) -> None:
        if not OUT_DIR.exists():
            pytest.skip("v1he outputs not found — run script first")

    def test_results_draft_exists(self) -> None:
        assert (OUT_DIR / "results_section_draft_v1he.md").exists()

    def test_discussion_draft_exists(self) -> None:
        assert (OUT_DIR / "discussion_section_draft_v1he.md").exists()

    def test_figure_captions_csv_exists(self) -> None:
        assert (OUT_DIR / "figure_captions_final_v1he.csv").exists()

    def test_table_captions_csv_exists(self) -> None:
        assert (OUT_DIR / "table_captions_final_v1he.csv").exists()

    def test_claim_matrix_csv_exists(self) -> None:
        assert (OUT_DIR / "claim_result_limitation_matrix_v1he.csv").exists()

    def test_summary_json_exists(self) -> None:
        assert (OUT_DIR / "tcc_results_discussion_summary_v1he.json").exists()

    def test_overleaf_plan_md_exists(self) -> None:
        assert (OUT_DIR / "overleaf_insert_plan_v1he.md").exists()

    def test_results_draft_non_empty(self) -> None:
        content = (OUT_DIR / "results_section_draft_v1he.md").read_text(encoding="utf-8")
        assert len(content) >= 3000

    def test_discussion_draft_non_empty(self) -> None:
        content = (OUT_DIR / "discussion_section_draft_v1he.md").read_text(encoding="utf-8")
        assert len(content) >= 3000

    def test_figure_captions_csv_row_count(self) -> None:
        with (OUT_DIR / "figure_captions_final_v1he.csv").open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_table_captions_csv_row_count(self) -> None:
        with (OUT_DIR / "table_captions_final_v1he.csv").open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6

    def test_claim_matrix_csv_row_count(self) -> None:
        with (OUT_DIR / "claim_result_limitation_matrix_v1he.csv").open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_summary_json_ready_for_overleaf_is_bool(self) -> None:
        with (OUT_DIR / "tcc_results_discussion_summary_v1he.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data["ready_for_overleaf"], bool)

    def test_summary_json_phase_field(self) -> None:
        with (OUT_DIR / "tcc_results_discussion_summary_v1he.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["phase"] == "v1he"

    def test_summary_json_guardrails_no_labels(self) -> None:
        with (OUT_DIR / "tcc_results_discussion_summary_v1he.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        guards = data["methodological_guardrails"]
        assert guards["labels_created"] is False
        assert guards["predictions_made"] is False
        assert guards["ground_truth_established"] is False


# ---------------------------------------------------------------------------
# TestV1HeGovernance
# ---------------------------------------------------------------------------

class TestV1HeGovernance:
    """Private path and forbidden term audits on all output files."""

    @pytest.fixture(autouse=True)
    def require_outputs(self) -> None:
        if not OUT_DIR.exists():
            pytest.skip("v1he outputs not found — run script first")

    def _all_md_content(self) -> str:
        texts = []
        for md in OUT_DIR.glob("*.md"):
            texts.append(md.read_text(encoding="utf-8"))
        return "\n".join(texts)

    def _all_csv_content(self) -> str:
        texts = []
        for csv_path in OUT_DIR.glob("*.csv"):
            texts.append(csv_path.read_text(encoding="utf-8-sig"))
        return "\n".join(texts)

    def test_no_private_paths_in_md_files(self) -> None:
        content = self._all_md_content()
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in content, f"Private path '{frag}' found in .md outputs"

    def test_no_private_paths_in_csv_files(self) -> None:
        content = self._all_csv_content()
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in content, f"Private path '{frag}' found in .csv outputs"

    def test_no_private_paths_in_json(self) -> None:
        json_path = OUT_DIR / "tcc_results_discussion_summary_v1he.json"
        content = json_path.read_text(encoding="utf-8")
        for frag in PRIVATE_PATH_FRAGMENTS:
            assert frag not in content, f"Private path '{frag}' found in summary JSON"

    def test_no_hard_forbidden_terms_in_results_draft(self) -> None:
        content = (OUT_DIR / "results_section_draft_v1he.md").read_text(encoding="utf-8").lower()
        for term in ["enchente", "alagamento", "risco de inundação",
                     "classificação supervisionada", "detecção de inundação"]:
            assert term not in content, f"Forbidden term '{term}' in results draft"

    def test_no_hard_forbidden_terms_in_discussion_draft(self) -> None:
        content = (OUT_DIR / "discussion_section_draft_v1he.md").read_text(encoding="utf-8").lower()
        for term in ["enchente", "alagamento", "risco de inundação",
                     "classificação supervisionada", "detecção de inundação"]:
            assert term not in content, f"Forbidden term '{term}' in discussion draft"

    def test_no_hard_forbidden_terms_in_claim_matrix(self) -> None:
        with (OUT_DIR / "claim_result_limitation_matrix_v1he.csv").open("r", encoding="utf-8-sig") as f:
            content = f.read().lower()
        for term in ["enchente", "alagamento", "risco de inundação",
                     "detecção de inundação"]:
            assert term not in content, f"Forbidden term '{term}' in claim matrix CSV"

    def test_results_draft_contains_note_nao_claim(self) -> None:
        """Results draft must include a methodological caveat note."""
        content = (OUT_DIR / "results_section_draft_v1he.md").read_text(encoding="utf-8")
        assert "Nenhum claim preditivo" in content or "nenhum claim" in content.lower()

    def test_discussion_draft_contains_limitation_section(self) -> None:
        content = (OUT_DIR / "discussion_section_draft_v1he.md").read_text(encoding="utf-8")
        assert "Limitações" in content or "limitações" in content

    def test_overleaf_plan_contains_nao_inserir_section(self) -> None:
        """Overleaf plan must include a 'What NOT to insert' section."""
        content = (OUT_DIR / "overleaf_insert_plan_v1he.md").read_text(encoding="utf-8")
        assert "Não Inserir" in content or "não inserir" in content.lower()

    def test_summary_json_visual_review_completed(self) -> None:
        with (OUT_DIR / "tcc_results_discussion_summary_v1he.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("visual_review_status") == "COMPLETED"

    def test_summary_json_robustness_fully_robust(self) -> None:
        with (OUT_DIR / "tcc_results_discussion_summary_v1he.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("robustness_status") == "FULLY_ROBUST"

    def test_claim_matrix_all_blocked_overclaims_start_with_nao(self) -> None:
        with (OUT_DIR / "claim_result_limitation_matrix_v1he.csv").open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["blocked_overclaim"].startswith("NÃO"), (
                f"blocked_overclaim does not start with 'NÃO': {row['blocked_overclaim'][:50]}"
            )

    def test_script_has_no_private_path_in_source(self) -> None:
        script = (ROOT / "scripts" / "dino" / "revp_v1he_tcc_results_discussion_synthesis.py")
        content = script.read_text(encoding="utf-8")
        for frag in ["PROJETO", r"C:\\Users\\gabriela", "AppData"]:
            assert frag not in content, f"Private path fragment '{frag}' in script source"

    def test_output_dir_is_under_local_runs(self) -> None:
        """Output dir must be inside local_runs/, never in scripts/ or tests/."""
        assert "local_runs" in str(v1he.OUT_DIR)
        assert "scripts" not in str(v1he.OUT_DIR)
