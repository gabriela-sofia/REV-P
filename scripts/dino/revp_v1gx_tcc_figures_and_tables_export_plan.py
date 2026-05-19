"""REV-P v1gx: TCC figures and tables export plan.

Organizes results into TCC-ready artifacts:
- inventory of required figures
- inventory of required tables
- mapping file-source → TCC section
- status per artifact: READY, NEEDS_LOCAL_OUTPUT, NEEDS_MANUAL_REVIEW, BLOCKED
- no complex figure generation; clear blocking when data unavailable

Allowed claims: documentation of what is ready vs blocked
Forbidden: over-committing to figures/tables without data; false readiness claims
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1gx"
PHASE_NAME = "TCC_FIGURES_AND_TABLES_EXPORT_PLAN"

DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gx"

# TCC-ready artifacts plan
TCC_FIGURES: list[dict[str, str]] = [
    {
        "figure_id": "f001_sentinel_patch_distribution",
        "title": "Spatial distribution of Sentinel patches across regions",
        "section": "3. Methodology | 3.1 Data and Patches",
        "source_files": "manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest",
        "format": "map (static image or interactive map)",
        "status": "READY",
        "notes": "Derived from input manifest; requires cartographic rendering",
    },
    {
        "figure_id": "f002_dino_embedding_pca",
        "title": "PCA projection of DINO embeddings by region",
        "section": "4. Results | 4.1 Embedding Structural Analysis",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_similarity_matrix_v1gu.json",
        "format": "scatter plot (2D or 3D)",
        "status": "NEEDS_LOCAL_OUTPUT",
        "notes": (
            "v1gu produced 12 real embeddings (768-dim, DINOv2-with-registers, "
            "4 per region: Curitiba/Petropolis/Recife). "
            "Similarity matrix available. Requires matplotlib PCA rendering."
        ),
    },
    {
        "figure_id": "f003_similarity_heatmap",
        "title": "Patch-patch cosine similarity heatmap",
        "section": "4. Results | 4.1 Embedding Structural Analysis",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_similarity_matrix_v1gu.json",
        "format": "heatmap (PNG or PDF)",
        "status": "NEEDS_LOCAL_OUTPUT",
        "notes": (
            "v1gu produced 12x12 cosine similarity matrix (real embeddings). "
            "Requires matplotlib heatmap rendering."
        ),
    },
    {
        "figure_id": "f004_neighbor_network",
        "title": "Top-k neighbor network (graph visualization)",
        "section": "4. Results | 4.1 Embedding Structural Analysis",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_neighbors_v1gu.csv",
        "format": "network graph (Gephi, Graphviz, or D3.js export)",
        "status": "NEEDS_LOCAL_OUTPUT",
        "notes": (
            "v1gu produced 60 neighbor pairs (12 patches x top-5 neighbors, real embeddings). "
            "Requires network graph rendering."
        ),
    },
    {
        "figure_id": "f005_regional_centroids",
        "title": "Regional embedding centroids and outliers (scatter)",
        "section": "4. Results | 4.2 Regional Structure",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
        "format": "scatter plot with annotations",
        "status": "NEEDS_LOCAL_OUTPUT",
        "notes": (
            "v1gu computed real centroids and medoids: "
            "Curitiba medoid=CUR_00357, Petropolis medoid=PET_00104, Recife medoid=REC_00205. "
            "Requires matplotlib scatter rendering."
        ),
    },
    {
        "figure_id": "f006_gis_coverage_matrix",
        "title": "GIS evidence coverage heatmap (patch x indicator)",
        "section": "4. Results | 4.3 External Evidence Availability",
        "source_files": "local_runs/dino_embeddings/v1gv/evidence_coverage_matrix_v1gv.csv",
        "format": "heatmap with status codes",
        "status": "NEEDS_LOCAL_OUTPUT",
        "notes": (
            "Data available from v1gv (128 patches x 3-5 indicators, real statuses). "
            "Requires rendering the CSV as a heatmap (Python/matplotlib or R)."
        ),
    },
    {
        "figure_id": "f007_sentinel_rgb_montage",
        "title": "RGB montage of selected patches (review candidates)",
        "section": "4. Results | 4.4 Visual Inspection Summary",
        "source_files": "data/sentinel/ (private workspace) + v1gw candidates list",
        "format": "grid of Sentinel RGB patches",
        "status": "NEEDS_LOCAL_OUTPUT",
        "notes": (
            "Candidate list available from v1gw. "
            "Requires local Sentinel TIF access and rasterio/matplotlib rendering."
        ),
    },
    {
        "figure_id": "f008_embedding_perturbation",
        "title": "Embedding stability under Sentinel acquisition variation",
        "section": "5. Validation | 5.1 Robustness Analysis",
        "source_files": "local_runs/dino_embeddings/v1gd/ (if executed)",
        "format": "scatter or error bars",
        "status": "BLOCKED",
        "notes": (
            "BLOCKED: perturbation analysis (v1gd) requires executed embeddings. "
            "Include only if v1gd produces outputs after v1fx unblocking."
        ),
    },
]

TCC_TABLES: list[dict[str, str]] = [
    {
        "table_id": "t001_patch_manifest",
        "title": "Sentinel patch inventory: canonical_patch_id, region, n_tifs",
        "section": "3. Methodology | 3.1 Data and Patches",
        "source_files": "manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv",
        "format": "CSV/DataFrame -> LaTeX table",
        "status": "READY",
        "notes": (
            "128 patches (Curitiba 43, Petropolis 48, Recife 37). "
            "Column subset: canonical_patch_id, region, eligibility_status."
        ),
    },
    {
        "table_id": "t002_embedding_corpus_stats",
        "title": "Embedding corpus statistics by region (count, dims, norm)",
        "section": "4. Results | 4.1 Embedding Structural Analysis",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
        "format": "summary table",
        "status": "READY",
        "notes": (
            "READY: v1gu produced 12 real embeddings (768-dim, DINOv2-with-registers-base, CPU). "
            "4 per region: Curitiba/Petropolis/Recife. Source: v1ge corpus from PROJETO TIFs."
        ),
    },
    {
        "table_id": "t003_similarity_statistics",
        "title": "Intra-region vs inter-region neighbor rate by region",
        "section": "4. Results | 4.2 Regional Structure",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
        "format": "summary table with percentages",
        "status": "READY",
        "notes": (
            "READY: v1gu computed intra_region_rate=0.367, inter_region_rate=0.633 "
            "(12 patches, top-5 neighbors). Direct from embedding_regional_summary_v1gu.json."
        ),
    },
    {
        "table_id": "t004_medoids_outliers",
        "title": "Regional medoids and outlier counts per region",
        "section": "4. Results | 4.2 Regional Structure",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
        "format": "summary table",
        "status": "READY",
        "notes": (
            "READY: v1gu computed medoids: CUR_00357 (Curitiba), PET_00104 (Petropolis), "
            "REC_00205 (Recife). Outliers: CUR_00350, PET_00016, REC_00019."
        ),
    },
    {
        "table_id": "t005_gis_indicators_by_region",
        "title": "GIS external evidence indicators and availability by region",
        "section": "4. Results | 4.3 External Evidence Availability",
        "source_files": "local_runs/dino_embeddings/v1gv/evidence_regional_summary_v1gv.csv",
        "format": "indicator x region cross-tabulation",
        "status": "READY",
        "notes": (
            "READY: v1gv populated with 128 patches x 3-5 real GIS indicators. "
            "regional_summary CSV has PARTIAL/NOT_ACQUIRED/MISSING counts per region."
        ),
    },
    {
        "table_id": "t006_review_candidates_summary",
        "title": "Human review candidate distribution by category and region",
        "section": "5. Methodology | 5.1 Review Stage Formalization",
        "source_files": "local_runs/dino_embeddings/v1gw/review_candidates_metadata_v1gw.json",
        "format": "category x region cross-tabulation",
        "status": "READY",
        "notes": (
            "READY: v1gw produced 47 real candidates (EMBEDDING_BASED mode). "
            "Categories: medoid_regional (3), outlier_structural (3), bridge_inter_regional, "
            "geometry_incomplete, geometry_complete, coverage_external_low. "
            "Source: v1gu real medoids/outliers from 12 DINOv2 embeddings."
        ),
    },
    {
        "table_id": "t007_embedding_neighbors_top_k",
        "title": "Example top-3 neighbors for representative patches",
        "section": "4. Results | 4.1 Embedding Structural Analysis",
        "source_files": "local_runs/dino_embeddings/v1gu/embedding_neighbors_v1gu.csv",
        "format": "tabular list with similarity values",
        "status": "NEEDS_LOCAL_OUTPUT",
        "notes": (
            "NEEDS_LOCAL_OUTPUT: v1gu produced 60 neighbor pairs (12 patches, top-5 neighbors). "
            "Source: local_runs/dino_embeddings/v1gu/embedding_neighbors_v1gu.csv. "
            "Requires tabular rendering/LaTeX."
        ),
    },
    {
        "table_id": "t008_validation_methods",
        "title": "Validation approaches and data sources (no ground truth)",
        "section": "5. Validation",
        "source_files": "docs/estado_metodologico_revp.md",
        "format": "descriptive table",
        "status": "READY",
        "notes": (
            "READY: estado_metodologico_revp.md documents all structural validation methods, "
            "active blockers, and explicitly states absence of ground truth."
        ),
    },
    {
        "table_id": "t009_external_evidence_registry",
        "title": "External evidence sources by region (type, tier, local status)",
        "section": "3. Methodology | 3.2 External GIS Evidence",
        "source_files": "datasets/external_evidence_registry.csv",
        "format": "summary table",
        "status": "READY",
        "notes": (
            "READY: 8 evidence records covering terrain, land use, drainage, "
            "administrative by region. Fields: source_name, region, evidence_type, "
            "local_status, evidence_tier."
        ),
    },
]

METHODOLOGICAL_GUARDRAILS: dict[str, Any] = {
    "review_only": True,
    "readiness_claims_honest": True,
    "no_overcommit": True,
    "blocked_artifacts_documented": True,
    "data_dependencies_clear": True,
}

STATUS_VALUES = ["READY", "NEEDS_LOCAL_OUTPUT", "NEEDS_MANUAL_REVIEW", "BLOCKED"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gx TCC figures and tables export plan."
    )
    parser.add_argument(
        "--mode", default="export-plan-run",
        choices=["export-plan-run"]
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def prepare_output_dir(path: Path, force: bool, resume: bool) -> None:
    if path.exists() and not force and not resume:
        raise FileExistsError(
            f"Output directory already exists: {path}. Use --force or --resume."
        )
    if path.exists() and force and not resume:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_figures_plan(output_dir: Path) -> None:
    """Export figures export plan as CSV and JSON."""
    rows = [
        {
            "figure_id": fig["figure_id"],
            "title": fig["title"],
            "section": fig["section"],
            "source_files": fig["source_files"],
            "format": fig["format"],
            "status": fig["status"],
            "notes": fig["notes"],
        }
        for fig in TCC_FIGURES
    ]

    fields = ["figure_id", "title", "section", "source_files", "format", "status", "notes"]
    write_csv(output_dir / "tcc_figures_export_plan_v1gx.csv", rows, fields)

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "figures": TCC_FIGURES,
    }
    write_json(output_dir / "tcc_figures_export_plan_v1gx.json", data)


def export_tables_plan(output_dir: Path) -> None:
    """Export tables export plan as CSV and JSON."""
    rows = [
        {
            "table_id": tbl["table_id"],
            "title": tbl["title"],
            "section": tbl["section"],
            "source_files": tbl["source_files"],
            "format": tbl["format"],
            "status": tbl["status"],
            "notes": tbl["notes"],
        }
        for tbl in TCC_TABLES
    ]

    fields = ["table_id", "title", "section", "source_files", "format", "status", "notes"]
    write_csv(output_dir / "tcc_tables_export_plan_v1gx.csv", rows, fields)

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "tables": TCC_TABLES,
    }
    write_json(output_dir / "tcc_tables_export_plan_v1gx.json", data)


def export_summary(output_dir: Path) -> None:
    """Export readiness summary."""
    fig_ready = sum(1 for f in TCC_FIGURES if f["status"] == "READY")
    fig_local = sum(1 for f in TCC_FIGURES if f["status"] == "NEEDS_LOCAL_OUTPUT")
    fig_review = sum(1 for f in TCC_FIGURES if f["status"] == "NEEDS_MANUAL_REVIEW")
    fig_blocked = sum(1 for f in TCC_FIGURES if f["status"] == "BLOCKED")

    tbl_ready = sum(1 for t in TCC_TABLES if t["status"] == "READY")
    tbl_local = sum(1 for t in TCC_TABLES if t["status"] == "NEEDS_LOCAL_OUTPUT")
    tbl_review = sum(1 for t in TCC_TABLES if t["status"] == "NEEDS_MANUAL_REVIEW")
    tbl_blocked = sum(1 for t in TCC_TABLES if t["status"] == "BLOCKED")

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "figures_summary": {
            "total": len(TCC_FIGURES),
            "ready": fig_ready,
            "needs_local_output": fig_local,
            "needs_manual_review": fig_review,
            "blocked": fig_blocked,
        },
        "tables_summary": {
            "total": len(TCC_TABLES),
            "ready": tbl_ready,
            "needs_local_output": tbl_local,
            "needs_manual_review": tbl_review,
            "blocked": tbl_blocked,
        },
        "methodological_guardrails": METHODOLOGICAL_GUARDRAILS,
        "instructions": {
            "ready": "Can be extracted immediately from source files",
            "needs_local_output": "Requires local computation/rendering; data dependency documented",
            "needs_manual_review": "Requires human judgment on inclusion/exclusion",
            "blocked": "Data unavailable or methodological constraint prevents inclusion",
        },
    }
    write_json(output_dir / "tcc_export_readiness_summary_v1gx.json", data)


def run_export_plan(args: argparse.Namespace) -> int:
    """Main export plan generation pipeline."""
    print(f"[{PHASE}] Starting TCC figures and tables export plan generation...")

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force, args.resume)
    print(f"[{PHASE}] Output directory: {rel(output_dir)}")

    print(f"[{PHASE}] Exporting figures export plan ({len(TCC_FIGURES)} artifacts)...")
    export_figures_plan(output_dir)

    print(f"[{PHASE}] Exporting tables export plan ({len(TCC_TABLES)} artifacts)...")
    export_tables_plan(output_dir)

    print(f"[{PHASE}] Generating readiness summary...")
    export_summary(output_dir)

    print(f"[{PHASE}] TCC export plan generation complete")
    print(f"[{PHASE}] Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_export_plan(args))
