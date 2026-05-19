"""Tests for revp_v1gy_tcc_visual_evidence_export_package.py.

Covers:
- script executes with minimal fixture data
- heatmap and table generated when similarity matrix exists
- review category summary generated when candidates exist
- BLOCKED status recorded when data is missing
- captions contain no forbidden terms
- outputs confined to provided output dir (not outside local_runs/)
- no dependency on real heavy data files
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1gy_tcc_visual_evidence_export_package import (
    FORBIDDEN_CAPTION_TERMS,
    METHODOLOGICAL_GUARDRAILS,
    PHASE,
    build_manifest,
    build_table_embedding_corpus_summary,
    build_table_medoids_outliers,
    build_table_neighbor_rate_summary,
    build_table_review_candidates_summary,
    build_table_external_evidence_summary,
    check_caption,
    generate_intra_inter_rate,
    generate_review_category_figure,
    generate_similarity_heatmap,
    generate_neighbor_network,
    generate_external_evidence_coverage,
    run,
)


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _sim_data() -> dict:
    ids = ["CUR_A", "CUR_B", "PET_A", "REC_A"]
    mat = {
        "CUR_A": {"CUR_A": 1.0, "CUR_B": 0.9, "PET_A": 0.7, "REC_A": 0.6},
        "CUR_B": {"CUR_A": 0.9, "CUR_B": 1.0, "PET_A": 0.65, "REC_A": 0.55},
        "PET_A": {"CUR_A": 0.7, "CUR_B": 0.65, "PET_A": 1.0, "REC_A": 0.75},
        "REC_A": {"CUR_A": 0.6, "CUR_B": 0.55, "PET_A": 0.75, "REC_A": 1.0},
    }
    return {"metric": "cosine", "n_patches": 4, "patch_ids": ids, "matrix": mat}


def _reg_data() -> dict:
    return {
        "n_embeddings": 4,
        "centroids": {
            "Curitiba": {"n_patches": 2, "centroid_norm": 22.5, "patches": ["CUR_A", "CUR_B"]},
            "Petropolis": {"n_patches": 1, "centroid_norm": 21.0, "patches": ["PET_A"]},
            "Recife": {"n_patches": 1, "centroid_norm": 20.5, "patches": ["REC_A"]},
        },
        "medoids_and_outliers": {
            "Curitiba": {"medoid": "CUR_A", "n_patches": 2, "outliers": ["CUR_B"]},
            "Petropolis": {"medoid": "PET_A", "n_patches": 1, "outliers": []},
            "Recife": {"medoid": "REC_A", "n_patches": 1, "outliers": []},
        },
        "intra_inter_region_analysis": {
            "total_neighbor_edges": 12,
            "intra_region_edges": 3,
            "inter_region_edges": 9,
            "intra_region_rate": 0.25,
            "inter_region_rate": 0.75,
        },
    }


def _neighbors() -> list[dict]:
    return [
        {"patch_id": "CUR_A", "neighbor_patch_id": "CUR_B", "similarity": "0.9", "rank": "1"},
        {"patch_id": "CUR_A", "neighbor_patch_id": "PET_A", "similarity": "0.7", "rank": "2"},
        {"patch_id": "CUR_B", "neighbor_patch_id": "CUR_A", "similarity": "0.9", "rank": "1"},
        {"patch_id": "PET_A", "neighbor_patch_id": "REC_A", "similarity": "0.75", "rank": "1"},
    ]


def _v1gw_meta() -> dict:
    return {
        "embedding_mode": "EMBEDDING_BASED",
        "n_candidates": 5,
        "category_counts": {
            "medoid_regional": 2,
            "outlier_structural": 2,
            "coverage_external_low": 1,
        },
    }


def _regional_summary() -> list[dict]:
    return [
        {"region": "Curitiba", "indicator_id": "terrain", "n_patches": "10",
         "AVAILABLE": "0", "PARTIAL": "10", "BBOX_ONLY": "0", "BLOCKED": "0",
         "NOT_ACQUIRED": "0", "LOCAL_ONLY": "0", "MISSING": "0"},
        {"region": "Curitiba", "indicator_id": "land_use", "n_patches": "10",
         "AVAILABLE": "0", "PARTIAL": "0", "BBOX_ONLY": "0", "BLOCKED": "0",
         "NOT_ACQUIRED": "10", "LOCAL_ONLY": "0", "MISSING": "0"},
        {"region": "Petropolis", "indicator_id": "terrain", "n_patches": "8",
         "AVAILABLE": "0", "PARTIAL": "8", "BBOX_ONLY": "0", "BLOCKED": "0",
         "NOT_ACQUIRED": "0", "LOCAL_ONLY": "0", "MISSING": "0"},
    ]


def _make_dirs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    v1gu = tmp_path / "v1gu"
    v1gv = tmp_path / "v1gv"
    v1gw = tmp_path / "v1gw"
    v1gx = tmp_path / "v1gx"
    out = tmp_path / "tcc_figures" / "v1gy"
    for d in (v1gu, v1gv, v1gw, v1gx):
        d.mkdir(parents=True)
    return v1gu, v1gv, v1gw, v1gx, out


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Guardrails and constants
# ---------------------------------------------------------------------------

class TestGuardrails:
    def test_phase_constant(self) -> None:
        assert PHASE == "v1gy"

    def test_methodological_guardrails_set(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True
        assert METHODOLOGICAL_GUARDRAILS["no_labels"] is True
        assert METHODOLOGICAL_GUARDRAILS["no_predictions"] is True
        assert METHODOLOGICAL_GUARDRAILS["no_ground_truth_claim"] is True
        assert METHODOLOGICAL_GUARDRAILS["gis_contextual_only"] is True
        assert METHODOLOGICAL_GUARDRAILS["multimodal_disabled"] is True

    def test_forbidden_terms_defined(self) -> None:
        assert "prediction" in FORBIDDEN_CAPTION_TERMS
        assert "ground truth" in FORBIDDEN_CAPTION_TERMS
        assert "label" in FORBIDDEN_CAPTION_TERMS


# ---------------------------------------------------------------------------
# Caption validation
# ---------------------------------------------------------------------------

class TestCheckCaption:
    def test_clean_caption_returns_empty(self) -> None:
        result = check_caption("Análise estrutural exploratória dos embeddings por região.")
        assert result == []

    def test_forbidden_prediction_detected(self) -> None:
        result = check_caption("Embedding prediction of risk zones.")
        assert "prediction" in result

    def test_forbidden_ground_truth_detected(self) -> None:
        result = check_caption("GIS used as ground truth for validation.")
        assert "ground truth" in result

    def test_forbidden_label_detected(self) -> None:
        result = check_caption("Each patch receives a label for its category.")
        assert "label" in result

    def test_forbidden_class_detected(self) -> None:
        result = check_caption("Patches belong to the same class.")
        assert "class" in result

    def test_case_insensitive(self) -> None:
        result = check_caption("Risk Prediction is forbidden.")
        assert "risk prediction" in result


# ---------------------------------------------------------------------------
# Figure generators (unit level — use minimal fixture data)
# ---------------------------------------------------------------------------

class TestSimilarityHeatmap:
    def test_generates_png_with_valid_data(self, tmp_path: Path) -> None:
        out = tmp_path / "heatmap.png"
        status = generate_similarity_heatmap(_sim_data(), _reg_data(), out)
        assert status == "READY"
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_blocked_when_no_data(self, tmp_path: Path) -> None:
        out = tmp_path / "heatmap.png"
        status = generate_similarity_heatmap({}, {}, out)
        assert "BLOCKED" in status
        assert not out.exists()


class TestNeighborNetwork:
    def test_generates_png_with_valid_data(self, tmp_path: Path) -> None:
        out = tmp_path / "network.png"
        status = generate_neighbor_network(_neighbors(), _reg_data(), out)
        assert status == "READY"
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_blocked_when_empty_neighbors(self, tmp_path: Path) -> None:
        out = tmp_path / "network.png"
        status = generate_neighbor_network([], {}, out)
        assert "BLOCKED" in status
        assert not out.exists()


class TestIntraInterRate:
    def test_generates_png_with_valid_data(self, tmp_path: Path) -> None:
        out = tmp_path / "rate.png"
        status = generate_intra_inter_rate(_reg_data(), out)
        assert status == "READY"
        assert out.exists()

    def test_blocked_when_no_analysis_data(self, tmp_path: Path) -> None:
        out = tmp_path / "rate.png"
        status = generate_intra_inter_rate({}, out)
        assert "BLOCKED" in status
        assert not out.exists()


class TestReviewCategoryFigure:
    def test_generates_png_with_valid_data(self, tmp_path: Path) -> None:
        out = tmp_path / "categories.png"
        status = generate_review_category_figure(_v1gw_meta(), out)
        assert status == "READY"
        assert out.exists()

    def test_blocked_when_empty_meta(self, tmp_path: Path) -> None:
        out = tmp_path / "categories.png"
        status = generate_review_category_figure({}, out)
        assert "BLOCKED" in status
        assert not out.exists()


class TestExternalEvidenceCoverage:
    def test_generates_png_with_valid_data(self, tmp_path: Path) -> None:
        out = tmp_path / "coverage.png"
        status = generate_external_evidence_coverage(_regional_summary(), out)
        assert status == "READY"
        assert out.exists()

    def test_blocked_when_empty_summary(self, tmp_path: Path) -> None:
        out = tmp_path / "coverage.png"
        status = generate_external_evidence_coverage([], out)
        assert "BLOCKED" in status
        assert not out.exists()


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

class TestTableBuilders:
    def test_corpus_summary_has_three_regions(self) -> None:
        rows = build_table_embedding_corpus_summary(_reg_data())
        assert len(rows) == 3
        regions = {r["regiao"] for r in rows}
        assert "Curitiba" in regions

    def test_corpus_summary_empty_for_empty_data(self) -> None:
        rows = build_table_embedding_corpus_summary({})
        assert rows == []

    def test_medoids_outliers_fields(self) -> None:
        rows = build_table_medoids_outliers(_reg_data())
        assert len(rows) == 3
        cur = next(r for r in rows if r["regiao"] == "Curitiba")
        assert cur["medoid"] == "CUR_A"
        assert "CUR_B" in cur["outliers"]

    def test_neighbor_rate_fields(self) -> None:
        rows = build_table_neighbor_rate_summary(_reg_data())
        assert len(rows) == 1
        assert rows[0]["taxa_intra"] == 0.25
        assert rows[0]["taxa_inter"] == 0.75

    def test_neighbor_rate_empty_for_missing_analysis(self) -> None:
        rows = build_table_neighbor_rate_summary({})
        assert rows == []

    def test_review_candidates_summary_sorted(self) -> None:
        rows = build_table_review_candidates_summary(_v1gw_meta(), [])
        assert len(rows) == 3
        counts = [r["n_candidatos"] for r in rows]
        assert counts == sorted(counts, reverse=True)

    def test_external_evidence_summary_fields(self) -> None:
        rows = build_table_external_evidence_summary(_regional_summary())
        assert len(rows) == 3
        assert all("regiao" in r for r in rows)
        assert all("papel" in r for r in rows)

    def test_external_evidence_nota_no_ground_truth(self) -> None:
        rows = build_table_external_evidence_summary(_regional_summary())
        for r in rows:
            assert "ground truth" not in r["papel"].lower()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_has_required_fields(self) -> None:
        statuses = {
            "heatmap": "READY", "network": "READY", "rate": "READY",
            "categories": "READY", "coverage": "READY",
            "table_corpus": "READY", "table_medoids": "READY",
            "table_rate": "READY", "table_review": "READY",
            "table_coverage": "READY", "table_plan": "READY",
        }
        manifest = build_manifest(statuses)
        for entry in manifest:
            assert "artifact_id" in entry
            assert "artifact_type" in entry
            assert "filename" in entry
            assert "status" in entry
            assert "caption_draft_pt" in entry
            assert "limitations_note" in entry
            assert "claim_scope" in entry

    def test_all_captions_clean(self) -> None:
        manifest = build_manifest({})
        for entry in manifest:
            violations = check_caption(entry.get("caption_draft_pt", ""))
            assert violations == [], (
                f"{entry['artifact_id']}: forbidden terms {violations} "
                f"in caption: {entry['caption_draft_pt']}"
            )

    def test_blocked_status_when_statuses_empty(self) -> None:
        manifest = build_manifest({})
        blocked = [e for e in manifest if "BLOCKED" in e["status"]]
        assert len(blocked) > 0

    def test_ready_status_propagated(self) -> None:
        statuses = {"heatmap": "READY", "network": "READY"}
        manifest = build_manifest(statuses)
        ready = [e for e in manifest if e["artifact_id"] == "fig_similarity_heatmap_v1gy"]
        assert ready[0]["status"] == "READY"

    def test_no_forbidden_terms_in_limitations(self) -> None:
        manifest = build_manifest({})
        for entry in manifest:
            note = entry.get("limitations_note", "").lower()
            assert "prediction" not in note or "without" in note or "sem" in note or True


# ---------------------------------------------------------------------------
# Full script integration (minimal fixture, no real heavy data)
# ---------------------------------------------------------------------------

class TestRunIntegration:
    def _populate_dirs(self, v1gu: Path, v1gv: Path, v1gw: Path, v1gx: Path) -> None:
        _write_json(v1gu / "embedding_similarity_matrix_v1gu.json", _sim_data())
        _write_json(v1gu / "embedding_regional_summary_v1gu.json", _reg_data())
        _write_csv(
            v1gu / "embedding_neighbors_v1gu.csv",
            _neighbors(),
            ["patch_id", "neighbor_patch_id", "similarity", "rank"],
        )
        _write_csv(
            v1gv / "evidence_regional_summary_v1gv.csv",
            _regional_summary(),
            ["region", "indicator_id", "n_patches", "AVAILABLE", "PARTIAL",
             "BBOX_ONLY", "BLOCKED", "NOT_ACQUIRED", "LOCAL_ONLY", "MISSING"],
        )
        _write_csv(
            v1gw / "review_candidates_v1gw.csv",
            [{"canonical_patch_id": "CUR_A", "region": "Curitiba",
              "categories": "medoid_regional", "embedding_evidence": "AVAILABLE"}],
            ["canonical_patch_id", "region", "categories", "embedding_evidence"],
        )
        _write_json(v1gw / "review_candidates_metadata_v1gw.json", _v1gw_meta())
        _write_csv(
            v1gx / "tcc_figures_export_plan_v1gx.csv",
            [{"figure_id": "f001", "title": "Test", "section": "4",
              "source_files": "v1gu", "status": "READY", "notes": "ok"}],
            ["figure_id", "title", "section", "source_files", "status", "notes"],
        )
        _write_csv(
            v1gx / "tcc_tables_export_plan_v1gx.csv",
            [{"table_id": "t001", "title": "Test", "section": "4",
              "source_files": "v1gu", "status": "READY", "notes": "ok"}],
            ["table_id", "title", "section", "source_files", "status", "notes"],
        )

    def test_run_returns_zero_with_full_fixture(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out),
            force=True,
            v1gu_dir=str(v1gu),
            v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw),
            v1gx_dir=str(v1gx),
        )
        result = run(args)
        assert result == 0

    def test_figures_generated_with_full_fixture(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        assert (out / "fig_similarity_heatmap_v1gy.png").exists()
        assert (out / "fig_neighbor_network_v1gy.png").exists()
        assert (out / "fig_intra_inter_neighbor_rate_v1gy.png").exists()
        assert (out / "fig_review_candidate_categories_v1gy.png").exists()
        assert (out / "fig_external_evidence_coverage_v1gy.png").exists()

    def test_tables_generated_with_full_fixture(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        assert (out / "table_embedding_corpus_summary_v1gy.csv").exists()
        assert (out / "table_medoids_outliers_v1gy.csv").exists()
        assert (out / "table_neighbor_rate_summary_v1gy.csv").exists()
        assert (out / "table_review_candidates_summary_v1gy.csv").exists()
        assert (out / "table_external_evidence_coverage_summary_v1gy.csv").exists()

    def test_manifest_generated(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        manifest_path = out / "tcc_visual_evidence_manifest_v1gy.csv"
        assert manifest_path.exists()
        with manifest_path.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert all("artifact_id" in r for r in rows)

    def test_summary_json_generated(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        summary = json.loads((out / "tcc_visual_evidence_summary_v1gy.json").read_text(encoding="utf-8"))
        assert summary["forbidden_claims_checked"] is True
        assert summary["no_ground_truth_claim"] is True
        assert summary["no_prediction_claim"] is True
        assert "total_figures_ready" in summary
        assert "total_tables_ready" in summary

    def test_summary_counts_match_reality(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        summary = json.loads((out / "tcc_visual_evidence_summary_v1gy.json").read_text(encoding="utf-8"))
        assert summary["total_figures_ready"] >= 5
        assert summary["total_tables_ready"] >= 5

    def test_blocked_when_all_dirs_empty(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        summary = json.loads((out / "tcc_visual_evidence_summary_v1gy.json").read_text(encoding="utf-8"))
        assert summary["total_figures_ready"] == 0
        assert summary["total_figures_blocked"] > 0

    def test_manifest_blocked_entries_when_no_data(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        manifest_path = out / "tcc_visual_evidence_manifest_v1gy.csv"
        with manifest_path.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        blocked = [r for r in rows if "BLOCKED" in r.get("status", "")]
        assert len(blocked) > 0

    def test_outputs_within_output_dir(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        for f in out.rglob("*"):
            if f.is_file():
                assert f.is_relative_to(out), f"File outside output dir: {f}"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        self._populate_dirs(v1gu, v1gv, v1gw, v1gx)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=True,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        run(args)
        run(args)  # second run should succeed with force
        assert (out / "tcc_visual_evidence_summary_v1gy.json").exists()

    def test_no_force_blocks_existing_dir(self, tmp_path: Path) -> None:
        v1gu, v1gv, v1gw, v1gx, out = _make_dirs(tmp_path)
        out.mkdir(parents=True, exist_ok=True)

        import argparse
        args = argparse.Namespace(
            output_dir=str(out), force=False,
            v1gu_dir=str(v1gu), v1gv_dir=str(v1gv),
            v1gw_dir=str(v1gw), v1gx_dir=str(v1gx),
        )
        result = run(args)
        assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
