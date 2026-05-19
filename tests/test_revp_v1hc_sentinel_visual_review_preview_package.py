"""Tests for REV-P v1hc: Sentinel Visual Review Preview Package."""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
V1HC_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hc"
FIGURES_DIR = V1HC_DIR / "figures"

EXPECTED_CSV_OUTPUTS = {
    "visual_review_preview_manifest_v1hc.csv": {"min_rows": 47},
    "visual_review_readiness_v1hc.csv": {"min_rows": 47},
    "visual_review_patch_index_v1hc.csv": {"min_rows": 47},
}
EXPECTED_CONTACT_SHEETS = [
    "contact_sheet_medoids_v1hc.png",
    "contact_sheet_outliers_v1hc.png",
    "contact_sheet_low_external_coverage_v1hc.png",
    "contact_sheet_all_review_candidates_v1hc.png",
]
FORBIDDEN_TERMS = {
    "prediction", "predictive", "detect", "detection",
    "classify", "classification", "risk", "vulnerability",
    "ground truth", "ground-truth", "label", "accuracy",
    "performance", "train", "supervised", "target", "causal",
}


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
# Fixture: minimal synthetic 6-band TIF
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_tif(tmp_path: Path) -> Path:
    """Create a minimal 6-band float64 TIF mimicking Sentinel-2 bands."""
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.transform import from_bounds

    tif_path = tmp_path / "patch_curitiba_99999.tif"
    rng = np.random.default_rng(42)
    data = rng.uniform(100, 3000, (6, 64, 64)).astype(np.float64)
    # Make B8 (NIR, band index 3) higher to produce positive NDVI
    data[3] = rng.uniform(2000, 4000, (64, 64))

    transform = from_bounds(0, 0, 1, 1, 64, 64)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=6,
        dtype="float64",
        crs="EPSG:32722",
        transform=transform,
    ) as dst:
        dst.write(data)
    return tif_path


# ---------------------------------------------------------------------------
# Unit tests using synthetic TIF
# ---------------------------------------------------------------------------

class TestV1HCPreviewFunctions:
    """Test preview generation functions with synthetic data."""

    def test_normalize_band_basic(self) -> None:
        """Test band normalization outputs [0,1] range."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        arr = np.array([[100.0, 500.0, 1000.0], [0.0, 200.0, 3000.0]])
        result = mod.normalize_band(arr)
        valid = result[result > 0]
        assert result.min() >= 0.0, "Output below 0"
        assert result.max() <= 1.0, "Output above 1"

    def test_normalize_band_zero_array(self) -> None:
        """Test normalization of all-zero array returns zeros."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        arr = np.zeros((10, 10), dtype=np.float32)
        result = mod.normalize_band(arr)
        assert result.sum() == 0.0

    def test_load_rgb_ndvi_shape(self, synthetic_tif: Path) -> None:
        """Test RGB and NDVI output shapes match TIF spatial dimensions."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        rgb, ndvi = mod.load_rgb_ndvi(synthetic_tif)
        assert rgb.ndim == 3, "RGB should be 3D (H, W, 3)"
        assert rgb.shape[2] == 3, "RGB last dim should be 3"
        assert ndvi.ndim == 2, "NDVI should be 2D"
        assert rgb.shape[:2] == ndvi.shape, "RGB and NDVI spatial dims should match"

    def test_rgb_values_in_range(self, synthetic_tif: Path) -> None:
        """Test RGB values are in [0, 1] after normalization."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        rgb, _ = mod.load_rgb_ndvi(synthetic_tif)
        assert rgb.min() >= 0.0 - 1e-6
        assert rgb.max() <= 1.0 + 1e-6

    def test_ndvi_range(self, synthetic_tif: Path) -> None:
        """Test NDVI values are within [-1, 1]."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        _, ndvi = mod.load_rgb_ndvi(synthetic_tif)
        assert ndvi.min() >= -1.0 - 1e-6
        assert ndvi.max() <= 1.0 + 1e-6

    def test_generate_preview_creates_file(self, synthetic_tif: Path, tmp_path: Path) -> None:
        """Test that preview PNG is created for a valid TIF."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        record = mod.PatchRecord(
            review_item_id="HRE001",
            canonical_patch_id="CUR_99999",
            region="Curitiba",
            candidate_category="coverage_external_low",
            uncertainty_level="high",
            usable_in_discussion="no",
        )
        out = mod.generate_patch_preview(record, synthetic_tif, tmp_path)
        assert out is not None, "generate_patch_preview returned None"
        assert out.exists(), f"Preview file not created: {out}"
        assert out.stat().st_size > 1000, "Preview file suspiciously small"

    def test_preview_filename_has_no_private_path(self, synthetic_tif: Path, tmp_path: Path) -> None:
        """Test that preview filename does not contain private PROJETO path."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        record = mod.PatchRecord(
            review_item_id="HRE001",
            canonical_patch_id="CUR_99999",
            region="Curitiba",
            candidate_category="coverage_external_low",
            uncertainty_level="high",
            usable_in_discussion="no",
        )
        out = mod.generate_patch_preview(record, synthetic_tif, tmp_path)
        assert out is not None
        # Check the filename itself (not the dir) contains no private path fragments
        assert "PROJETO" not in out.name, "Private path leaked into output filename"
        assert "Documents" not in out.name, "Private path leaked into output filename"

    def test_missing_tif_returns_none(self, tmp_path: Path) -> None:
        """Test that a missing TIF returns None without raising."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        missing = tmp_path / "nonexistent.tif"
        record = mod.PatchRecord(
            review_item_id="HRE001",
            canonical_patch_id="CUR_99999",
            region="Curitiba",
            candidate_category="coverage_external_low",
            uncertainty_level="high",
            usable_in_discussion="no",
        )
        # Should fail gracefully (rasterio will raise, function should return None)
        result = mod.generate_patch_preview(record, missing, tmp_path)
        assert result is None

    def test_resolve_tif_path_curitiba(self, tmp_path: Path) -> None:
        """Test TIF path resolution for Curitiba prefix."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        (tmp_path / "patch_curitiba_00038.tif").touch()
        result = mod.resolve_tif_path("CUR_00038", "Curitiba", tmp_path)
        assert result is not None
        assert result.name == "patch_curitiba_00038.tif"

    def test_resolve_tif_path_petropolis(self, tmp_path: Path) -> None:
        """Test TIF path resolution for Petropolis prefix."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        (tmp_path / "patch_petropolis_00016.tif").touch()
        result = mod.resolve_tif_path("PET_00016", "Petrópolis", tmp_path)
        assert result is not None
        assert result.name == "patch_petropolis_00016.tif"

    def test_resolve_missing_returns_none(self, tmp_path: Path) -> None:
        """Test that resolve returns None when TIF does not exist."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        result = mod.resolve_tif_path("CUR_99999", "Curitiba", tmp_path)
        assert result is None

    def test_contact_sheet_with_previews(
        self, synthetic_tif: Path, tmp_path: Path
    ) -> None:
        """Test contact sheet is created when previews are available."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        # Generate a real preview first
        record = mod.PatchRecord(
            review_item_id="HRE001",
            canonical_patch_id="CUR_99999",
            region="Curitiba",
            candidate_category="coverage_external_low",
            uncertainty_level="high",
            usable_in_discussion="no",
        )
        preview_path = mod.generate_patch_preview(record, synthetic_tif, tmp_path)
        assert preview_path is not None

        cs_path = tmp_path / "contact_sheet_test.png"
        ok = mod.build_contact_sheet(
            [record],
            {"CUR_99999": preview_path},
            cs_path,
            "Test Contact Sheet",
            ncols=1,
        )
        assert ok, "Contact sheet build returned False"
        assert cs_path.exists(), "Contact sheet file not created"
        assert cs_path.stat().st_size > 5000, "Contact sheet too small"

    def test_contact_sheet_empty_returns_false(self, tmp_path: Path) -> None:
        """Test contact sheet returns False when no previews available."""
        sys.path.insert(0, str(ROOT / "scripts" / "dino"))
        mod = importlib.import_module("revp_v1hc_sentinel_visual_review_preview_package")

        record = mod.PatchRecord(
            review_item_id="HRE001",
            canonical_patch_id="CUR_99999",
            region="Curitiba",
            candidate_category="coverage_external_low",
            uncertainty_level="high",
            usable_in_discussion="no",
        )
        cs_path = tmp_path / "empty_sheet.png"
        ok = mod.build_contact_sheet([record], {}, cs_path, "Empty", ncols=1)
        assert not ok, "Contact sheet should return False with no previews"


# ---------------------------------------------------------------------------
# Integration tests on actual v1hc outputs
# ---------------------------------------------------------------------------

class TestV1HCOutputs:
    """Test v1hc output files exist and have valid structure."""

    @pytest.mark.parametrize("filename", EXPECTED_CSV_OUTPUTS.keys())
    def test_output_csv_exists(self, filename: str) -> None:
        path = V1HC_DIR / filename
        assert path.exists(), f"Expected output not found: {filename}"

    def test_summary_json_exists(self) -> None:
        path = V1HC_DIR / "visual_review_preview_summary_v1hc.json"
        assert path.exists(), "Summary JSON not found"

    @pytest.mark.parametrize("filename", EXPECTED_CSV_OUTPUTS.keys())
    def test_csv_row_count(self, filename: str) -> None:
        path = V1HC_DIR / filename
        rows = read_csv(path)
        min_rows = EXPECTED_CSV_OUTPUTS[filename]["min_rows"]
        assert len(rows) >= min_rows, f"{filename}: expected >= {min_rows} rows, got {len(rows)}"

    def test_manifest_required_columns(self) -> None:
        rows = read_csv(V1HC_DIR / "visual_review_preview_manifest_v1hc.csv")
        required = [
            "review_item_id", "canonical_patch_id", "region",
            "candidate_category", "tif_found", "tif_path_status",
            "preview_status", "preview_file", "visual_review_scope",
            "limitations_note",
        ]
        if rows:
            for col in required:
                assert col in rows[0], f"Missing column: {col}"

    def test_readiness_valid_statuses(self) -> None:
        rows = read_csv(V1HC_DIR / "visual_review_readiness_v1hc.csv")
        valid_statuses = {"READY_FOR_VISUAL_REVIEW", "TIF_NOT_FOUND", "PREVIEW_FAILED", "BLOCKED"}
        for row in rows:
            status = row.get("readiness_status", "")
            assert status in valid_statuses, f"Invalid readiness status: {status}"

    def test_no_blocked_without_reason(self) -> None:
        """Test that blocked candidates have a documented reason."""
        manifest = read_csv(V1HC_DIR / "visual_review_preview_manifest_v1hc.csv")
        readiness = read_csv(V1HC_DIR / "visual_review_readiness_v1hc.csv")

        blocked_ids = {r["review_item_id"] for r in readiness if r["readiness_status"] == "BLOCKED"}
        for row in manifest:
            if row["review_item_id"] in blocked_ids:
                assert row.get("limitations_note"), \
                    f"Blocked candidate {row['review_item_id']} has no limitations_note"

    def test_summary_guardrails(self) -> None:
        data = read_json(V1HC_DIR / "visual_review_preview_summary_v1hc.json")
        guardrails = data.get("methodological_guardrails", {})
        assert guardrails.get("labels_created") is False
        assert guardrails.get("predictions_made") is False
        assert guardrails.get("ground_truth_established") is False
        assert guardrails.get("review_only") is True
        assert guardrails.get("private_paths_in_outputs") is False

    def test_summary_candidate_count(self) -> None:
        data = read_json(V1HC_DIR / "visual_review_preview_summary_v1hc.json")
        assert data.get("n_candidates", 0) == 47

    def test_at_least_one_preview_generated(self) -> None:
        data = read_json(V1HC_DIR / "visual_review_preview_summary_v1hc.json")
        n = data.get("n_previews_generated", 0)
        assert n >= 1, "No previews were generated"

    def test_contact_sheets_exist_when_previews_generated(self) -> None:
        data = read_json(V1HC_DIR / "visual_review_preview_summary_v1hc.json")
        n = data.get("n_previews_generated", 0)
        if n == 0:
            pytest.skip("No previews generated; contact sheets skipped")
        for sheet_name in EXPECTED_CONTACT_SHEETS:
            sheet_path = FIGURES_DIR / sheet_name
            assert sheet_path.exists(), f"Contact sheet not found: {sheet_name}"


class TestV1HCNoForbiddenContent:
    """Test that outputs contain no forbidden claims or labels."""

    def test_no_forbidden_terms_in_visual_scope(self) -> None:
        rows = read_csv(V1HC_DIR / "visual_review_preview_manifest_v1hc.csv")
        violations = []
        for row in rows:
            scope = row.get("visual_review_scope", "").lower()
            for term in FORBIDDEN_TERMS:
                if term in scope and "no " + term not in scope and "not " + term not in scope:
                    violations.append(f"{row['review_item_id']}: '{term}' in visual_review_scope")
        assert not violations, f"Forbidden terms found: {violations}"

    def test_no_private_paths_in_manifest(self) -> None:
        """Test that the committed manifest contains no private PROJETO paths."""
        rows = read_csv(V1HC_DIR / "visual_review_preview_manifest_v1hc.csv")
        for row in rows:
            for col in ["preview_file", "visual_review_scope", "limitations_note"]:
                val = row.get(col, "")
                assert "PROJETO" not in val, f"Private path in {col}: {row['review_item_id']}"
                assert "gabriela" not in val, f"User path in {col}: {row['review_item_id']}"

    def test_no_private_paths_in_summary_json(self) -> None:
        content = (V1HC_DIR / "visual_review_preview_summary_v1hc.json").read_text(encoding="utf-8")
        assert "PROJETO" not in content, "Private PROJETO path in summary JSON"
        assert "gabriela" not in content, "User path in summary JSON"

    def test_patch_index_no_labels(self) -> None:
        rows = read_csv(V1HC_DIR / "visual_review_patch_index_v1hc.csv")
        for row in rows:
            for col, val in row.items():
                assert "label" not in col.lower() or col == "no_label_created_confirmed", \
                    f"Unexpected label column: {col}"

    def test_summary_no_forbidden_claims(self) -> None:
        data = read_json(V1HC_DIR / "visual_review_preview_summary_v1hc.json")
        text = json.dumps(data).lower()
        strong_forbidden = ["flood prediction", "flood detection", "risk label", "vulnerability score"]
        for term in strong_forbidden:
            assert term not in text, f"Forbidden claim in summary: '{term}'"


class TestV1HCOutputLocation:
    """Test that outputs are in the correct local_runs location."""

    def test_all_outputs_in_local_runs(self) -> None:
        assert "local_runs" in str(V1HC_DIR)

    def test_figures_in_v1hc_figures_dir(self) -> None:
        assert "local_runs" in str(FIGURES_DIR)
        assert "v1hc" in str(FIGURES_DIR)

    def test_no_outputs_in_docs(self) -> None:
        docs_dir = ROOT / "docs"
        for pattern in ["*_v1hc.png", "*_v1hc.csv"]:
            found = list(docs_dir.rglob(pattern))
            assert not found, f"v1hc outputs found in docs/: {found}"

    def test_no_outputs_in_scripts(self) -> None:
        scripts_dir = ROOT / "scripts"
        for pattern in ["*_v1hc.png"]:
            found = list(scripts_dir.rglob(pattern))
            assert not found, f"v1hc PNG found in scripts/: {found}"


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
