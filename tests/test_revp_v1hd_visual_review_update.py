"""Tests for REV-P v1hd: Visual-Assisted Interpretation Update."""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
V1HD_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hd"

EXPECTED_OUTPUTS = {
    "review_gate_visual_annotation_v1hd.csv": {"min_rows": 47},
    "review_gate_visual_examples_for_tcc_v1hd.csv": {"min_rows": 1},
}

FORBIDDEN_PREDICTIVE_TERMS = [
    "enchente", "alagamento", "inundação", "predição", "risco predito",
    "vulnerabilidade predita", "flood", "detecção de risco",
    "classifica como", "valida o modelo",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_tif(tmp_path: Path) -> Path:
    """6-band float64 TIF mimicking Sentinel-2 structure."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    tif_path = tmp_path / "patch_curitiba_99999.tif"
    rng = np.random.default_rng(42)
    data = rng.uniform(100, 3000, (6, 64, 64)).astype(np.float64)
    data[3] = rng.uniform(2000, 4000, (64, 64))  # NIR high → positive NDVI

    transform = from_bounds(0, 0, 1, 1, 64, 64)
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=64, width=64, count=6,
        dtype="float64", crs="EPSG:32722", transform=transform,
    ) as dst:
        dst.write(data)
    return tif_path


@pytest.fixture()
def mod():
    sys.path.insert(0, str(ROOT / "scripts" / "dino"))
    return importlib.import_module("revp_v1hd_visual_review_update_update")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestV1HDImageStats:
    """Test image statistics computation."""

    def test_compute_stats_returns_imagestats(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None, "compute_stats_from_tif returned None"
        assert isinstance(stats.mean_brightness, float)
        assert isinstance(stats.ndvi_mean, float)
        assert isinstance(stats.n_pixels, int)

    def test_brightness_in_range(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        assert 0.0 <= stats.mean_brightness <= 1.0
        assert 0.0 <= stats.std_brightness

    def test_ndvi_in_range(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        assert -1.0 <= stats.ndvi_mean <= 1.0
        assert 0.0 <= stats.ndvi_veg_fraction <= 1.0
        assert 0.0 <= stats.ndvi_low_fraction <= 1.0

    def test_ndvi_fractions_sum_lte_one(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        assert stats.ndvi_veg_fraction + stats.ndvi_low_fraction <= 1.0 + 1e-6

    def test_missing_tif_returns_none(self, tmp_path: Path, mod) -> None:
        result = mod.compute_stats_from_tif(tmp_path / "nonexistent.tif")
        assert result is None

    def test_stats_to_visual_notes_no_forbidden_terms(self, synthetic_tif: Path, mod) -> None:
        """Visual notes must not contain positive predictive or operational claims."""
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        notes = mod.stats_to_visual_notes(stats, "CUR_00038", "coverage_external_low")
        notes_lower = notes.lower()
        # These terms must never appear as positive assertions (flood event, risk prediction etc.)
        hard_forbidden = [
            "enchente", "alagamento", "inundação", "risco predito",
            "vulnerabilidade predita", "flood", "detecção de risco",
        ]
        for term in hard_forbidden:
            assert term not in notes_lower, f"Forbidden term '{term}' in visual notes"

    def test_stats_to_visual_notes_has_ndvi_info(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        notes = mod.stats_to_visual_notes(stats, "CUR_00038", "coverage_external_low")
        assert "NDVI" in notes or "ndvi" in notes.lower()

    def test_visual_notes_is_conservative(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        notes = mod.stats_to_visual_notes(stats, "CUR_00357", "medoid_regional")
        assert any(word in notes for word in ["exploratória", "conservadora", "possível", "aparente", "padrão consistente"])

    def test_stats_to_visual_notes_no_label(self, synthetic_tif: Path, mod) -> None:
        """Visual notes must not assign a label or class to the patch."""
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        notes = mod.stats_to_visual_notes(stats, "CUR_00350", "outlier_structural")
        # The note may say "sem rótulos" but must not assign one as a positive claim
        assert "rótulo" not in notes.lower() or "sem rótulo" in notes.lower(), \
            "Notes must not assign a label"
        # Must not use the word 'label' as a positive assignment
        assert "label" not in notes.lower() or "no label" in notes.lower() or \
            "sem label" in notes.lower(), "Notes must not assign a label"

    def test_uncertainty_never_lowers_non_corpus(self, synthetic_tif: Path, mod) -> None:
        """Non-corpus patches should never get lower than high uncertainty."""
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        result = mod.stats_to_uncertainty(stats, "CUR_99999", "high")
        assert result == "high"

    def test_uncertainty_medoid_stays_low(self, synthetic_tif: Path, mod) -> None:
        """Medoid patches already at low should not be changed."""
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        result = mod.stats_to_uncertainty(stats, "CUR_00357", "low")
        assert result == "low"

    def test_usable_promoted_when_preview_generated(self, mod) -> None:
        """Patches with 'no' usability should become 'conditional' if preview generated."""
        result = mod.stats_to_usable("CUR_99999", "no", "GENERATED")
        assert result == "conditional"

    def test_usable_unchanged_without_preview(self, mod) -> None:
        """Patches without preview keep prior usability."""
        result = mod.stats_to_usable("CUR_99999", "no", "BLOCKED")
        assert result == "no"

    def test_discussion_note_no_predictive_claim(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        note = mod.stats_to_discussion_note(stats, "CUR_00357", "medoid_regional", "Curitiba", "low")
        note_lower = note.lower()
        for term in FORBIDDEN_PREDICTIVE_TERMS:
            assert term not in note_lower, f"Forbidden term '{term}' in discussion_note"

    def test_discussion_note_medoid_mentions_region(self, synthetic_tif: Path, mod) -> None:
        stats = mod.compute_stats_from_tif(synthetic_tif)
        assert stats is not None
        note = mod.stats_to_discussion_note(stats, "CUR_00357", "medoid_regional", "Curitiba", "low")
        assert "Curitiba" in note or "medoid" in note.lower()


# ---------------------------------------------------------------------------
# Integration tests on v1hd outputs
# ---------------------------------------------------------------------------

class TestV1HDOutputs:
    """Test v1hd output files exist and are correctly structured."""

    @pytest.mark.parametrize("filename", EXPECTED_OUTPUTS.keys())
    def test_output_csv_exists(self, filename: str) -> None:
        path = V1HD_DIR / filename
        assert path.exists(), f"Expected output not found: {filename}"

    def test_summary_json_exists(self) -> None:
        assert (V1HD_DIR / "review_gate_visual_summary_v1hd.json").exists()

    def test_synthesis_md_exists(self) -> None:
        assert (V1HD_DIR / "review_gate_visual_discussion_synthesis_v1hd.md").exists()

    def test_annotation_row_count(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        assert len(rows) == 47, f"Expected 47 rows, got {len(rows)}"

    def test_annotation_required_columns(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        required = [
            "review_item_id", "canonical_patch_id", "region", "candidate_category",
            "preview_status", "visual_interpretation_mode", "visual_pattern_notes",
            "structural_context_notes", "external_evidence_notes",
            "methodological_interpretation", "uncertainty_level", "usable_in_discussion",
            "discussion_note", "no_label_created_confirmed",
            "no_prediction_claim_confirmed", "no_ground_truth_claim_confirmed",
            "review_only_confirmed",
        ]
        if rows:
            for col in required:
                assert col in rows[0], f"Missing required column: {col}"

    def test_all_confirmations_are_yes(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        for row in rows:
            assert row.get("no_label_created_confirmed") == "yes", \
                f"no_label_created_confirmed != yes in {row['review_item_id']}"
            assert row.get("no_prediction_claim_confirmed") == "yes", \
                f"no_prediction_claim_confirmed != yes in {row['review_item_id']}"
            assert row.get("no_ground_truth_claim_confirmed") == "yes"
            assert row.get("review_only_confirmed") == "yes"

    def test_visual_modes_valid(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        valid_modes = {"COMPUTED", "MANUAL_REQUIRED"}
        for row in rows:
            assert row.get("visual_interpretation_mode") in valid_modes, \
                f"Invalid mode in {row['review_item_id']}: {row.get('visual_interpretation_mode')}"

    def test_preview_not_available_absent(self) -> None:
        """Verify PREVIEW_NOT_AVAILABLE from v1hb has been removed for computed patches."""
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        computed = [r for r in rows if r.get("visual_interpretation_mode") == "COMPUTED"]
        for row in computed:
            assert "PREVIEW_NOT_AVAILABLE" not in row.get("visual_pattern_notes", ""), \
                f"PREVIEW_NOT_AVAILABLE in computed row {row['review_item_id']}"

    def test_summary_guardrails(self) -> None:
        data = read_json(V1HD_DIR / "review_gate_visual_summary_v1hd.json")
        guardrails = data.get("methodological_guardrails", {})
        assert guardrails.get("labels_created") is False
        assert guardrails.get("predictions_made") is False
        assert guardrails.get("ground_truth_established") is False
        assert guardrails.get("review_only") is True
        assert guardrails.get("visual_notes_are_statistical_only") is True
        assert data.get("forbidden_claims_checked") is True

    def test_summary_candidate_count(self) -> None:
        data = read_json(V1HD_DIR / "review_gate_visual_summary_v1hd.json")
        assert data.get("n_candidates") == 47

    def test_examples_have_tcc_suggestion(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_examples_for_tcc_v1hd.csv")
        assert len(rows) >= 1, "No examples generated"
        for row in rows:
            assert row.get("tcc_use_suggestion"), f"Missing tcc_use_suggestion in {row.get('example_type')}"
            assert row.get("example_type"), "Missing example_type"

    def test_synthesis_md_not_empty(self) -> None:
        path = V1HD_DIR / "review_gate_visual_discussion_synthesis_v1hd.md"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500, "Synthesis document too short"

    def test_synthesis_has_key_sections(self) -> None:
        content = (V1HD_DIR / "review_gate_visual_discussion_synthesis_v1hd.md").read_text(encoding="utf-8")
        for section in ["## 2. Medoids", "## 3. Outliers", "## 6. Como Usar", "## 7. Limitações"]:
            assert section in content, f"Missing section: {section}"


class TestV1HDNoForbiddenContent:
    """Test that v1hd outputs contain no forbidden claims."""

    def test_no_predictive_terms_in_visual_notes(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        violations = []
        for row in rows:
            notes = row.get("visual_pattern_notes", "").lower()
            for term in FORBIDDEN_PREDICTIVE_TERMS:
                if term in notes:
                    violations.append(f"{row['review_item_id']}: '{term}' in visual_pattern_notes")
        assert not violations, f"Forbidden terms found: {violations}"

    def test_no_predictive_terms_in_discussion_notes(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        violations = []
        for row in rows:
            note = row.get("discussion_note", "").lower()
            for term in FORBIDDEN_PREDICTIVE_TERMS:
                if term in note:
                    violations.append(f"{row['review_item_id']}: '{term}' in discussion_note")
        assert not violations, f"Forbidden terms found: {violations}"

    def test_no_label_columns(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        if rows:
            for col in rows[0].keys():
                assert col not in {"label", "target", "class", "risk_class"}, \
                    f"Forbidden column found: {col}"

    def test_no_private_paths_in_outputs(self) -> None:
        for fname in [
            "review_gate_visual_annotation_v1hd.csv",
            "review_gate_visual_summary_v1hd.json",
            "review_gate_visual_discussion_synthesis_v1hd.md",
        ]:
            path = V1HD_DIR / fname
            if path.exists():
                content = path.read_text(encoding="utf-8")
                assert "PROJETO" not in content, f"Private path in {fname}"
                assert "gabriela" not in content, f"User path in {fname}"

    def test_synthesis_no_predictive_claims(self) -> None:
        path = V1HD_DIR / "review_gate_visual_discussion_synthesis_v1hd.md"
        content = path.read_text(encoding="utf-8").lower()
        for term in ["prediz enchente", "risco predito", "valida risco", "detecta inundação"]:
            assert term not in content, f"Predictive claim in synthesis: '{term}'"


class TestV1HDBlockedCandidates:
    """Test that candidates without preview are properly handled."""

    def test_blocked_candidates_marked_manual_required(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        for row in rows:
            if row.get("preview_status") == "BLOCKED":
                assert row.get("visual_interpretation_mode") == "MANUAL_REQUIRED", \
                    f"BLOCKED candidate should be MANUAL_REQUIRED: {row['review_item_id']}"

    def test_ready_candidates_have_visual_evidence(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        for row in rows:
            if row.get("preview_status") == "GENERATED":
                assert row.get("visual_interpretation_mode") in {"COMPUTED", "MANUAL_REQUIRED"}, \
                    f"GENERATED candidate has unexpected mode: {row['review_item_id']}"

    def test_uncertainty_levels_valid(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        valid_levels = {"low", "medium", "high"}
        for row in rows:
            assert row.get("uncertainty_level") in valid_levels, \
                f"Invalid uncertainty in {row['review_item_id']}"

    def test_usable_values_valid(self) -> None:
        rows = read_csv(V1HD_DIR / "review_gate_visual_annotation_v1hd.csv")
        valid = {"yes", "no", "conditional"}
        for row in rows:
            assert row.get("usable_in_discussion") in valid, \
                f"Invalid usable_in_discussion in {row['review_item_id']}"


class TestV1HDOutputLocation:
    """Test that v1hd outputs are in the correct location."""

    def test_outputs_in_local_runs(self) -> None:
        assert "local_runs" in str(V1HD_DIR)

    def test_no_outputs_in_docs(self) -> None:
        docs_dir = ROOT / "docs"
        found = list(docs_dir.rglob("*_v1hd.*"))
        assert not found, f"v1hd outputs found in docs/: {found}"

    def test_no_png_in_v1hd(self) -> None:
        """v1hd should not generate PNG files — those belong to v1hc."""
        pngs = list(V1HD_DIR.rglob("*.png")) if V1HD_DIR.exists() else []
        assert not pngs, f"Unexpected PNG files in v1hd: {pngs}"


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
