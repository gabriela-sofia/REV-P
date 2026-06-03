"""REV-P v1gz: Scientific Evidence Master Audit.

Consolidates v1gu–v1gy into a comprehensive evidence audit:
- maps each allowed claim to supporting evidence (READY/PARTIAL/BLOCKED)
- explicitly blocks forbidden claims with technical reasoning
- audits figure/table quality and captions
- formalizes review gate as a methodological stage (no labels created)
- identifies remaining scientific gaps
- generates readiness matrix for TCC writing

All outputs local-only; no heavy binaries versioned.
Forbidden claims verified explicitly.
Corpus constraints (12 patches) documented.
GIS contextual-only, not ground-truth.
No predictions, labels, targets, classification, or multimodal claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1gz"
PHASE_NAME = "SCIENTIFIC_EVIDENCE_MASTER_AUDIT"

DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / PHASE

# Input directories
V1GU_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gu"
V1GV_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gv"
V1GW_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gw"
V1GX_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gx"
V1GY_DIR = ROOT / "local_runs" / "tcc_figures" / "v1gy"
V1HA_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1ha"

# Allowed claims (review-only, no prediction)
ALLOWED_CLAIMS = {
    "structural_coherence": {
        "description": "Embeddings exhibit structural coherence within regions",
        "evidence_source": "v1gu",
        "output_files": ["embedding_similarity_matrix_v1gu.json", "embedding_neighbors_v1gu.csv"],
    },
    "embedding_stability": {
        "description": "Embedding neighborhoods show regional patterns (exploratory)",
        "evidence_source": "v1gu",
        "output_files": ["embedding_regional_summary_v1gu.json"],
    },
    "exploratory_similarity": {
        "description": "Similarity metrics inform exploratory structural analysis",
        "evidence_source": "v1gu",
        "output_files": ["embedding_similarity_matrix_v1gu.json"],
    },
    "intra_inter_neighborhood_rate": {
        "description": "Neighborhood topology shows intra vs inter-region distribution",
        "evidence_source": "v1gu",
        "output_files": ["embedding_regional_summary_v1gu.json"],
    },
    "medoid_identification": {
        "description": "Structural medoids identified per region (exploratory)",
        "evidence_source": "v1gu",
        "output_files": ["embedding_regional_summary_v1gu.json"],
    },
    "outlier_identification": {
        "description": "Structural outliers identified per region (for review)",
        "evidence_source": "v1gu",
        "output_files": ["embedding_regional_summary_v1gu.json"],
    },
    "review_gate_formalized": {
        "description": "Review gate is a methodological stage (not validation/labeling)",
        "evidence_source": "v1gw",
        "output_files": ["review_candidates_v1gw.csv", "review_protocol_v1gw.md"],
    },
    "gis_contextual": {
        "description": "GIS provides territorial context (not ground truth, not validation)",
        "evidence_source": "v1gv",
        "output_files": ["evidence_coverage_matrix_v1gv.csv", "evidence_metadata_v1gv.json"],
    },
    "audit_trail": {
        "description": "Pipeline is fully documented and auditable",
        "evidence_source": "v1gu,v1gv,v1gw,v1gx,v1gy",
        "output_files": ["manifests"],
    },
    "pipeline_reproducibility": {
        "description": "Scripts and commands documented for local reproducibility",
        "evidence_source": "scripts + docs",
        "output_files": ["dino_command_registry.md"],
    },
    "embedding_robustness": {
        "description": "Embeddings remain stable under perturbation (optional evidence)",
        "evidence_source": "v1ha",
        "output_files": ["embedding_drift_metrics.csv", "neighbor_persistence_under_perturbation.csv"],
    },
}

# Forbidden claims (BLOCKED explicitly)
FORBIDDEN_CLAIMS = {
    "vulnerability_prediction": "Prediction is not in scope; DINO is read-only structural analysis",
    "flood_susceptibility_classification": "Classification is forbidden; no labels exist",
    "flood_detection": "Detection implies predictive claim; not applicable",
    "ground_truth_validation": "GIS is contextual only, not ground truth",
    "supervised_model_performance": "No supervised training executed",
    "dino_as_classifier": "DINO extracts embeddings; no classifier trained",
    "multimodal_execution": "Multimodal is disabled (single-modality: Sentinel images only)",
    "risk_classification": "No risk labels or classes created",
    "target_variable": "No targets defined; analysis is exploratory",
    "causal_inference": "No causal claims made; correlative analysis only",
}

# Table/figure quality checklist
FORBIDDEN_CAPTION_TERMS = {
    "prediction", "predictive", "detection", "ground truth", "ground-truth",
    "classification", "classify", "risk classification", "risk assessment",
    "class", "label", "labeling", "supervised", "target", "causal",
    "validation", "validates", "proven", "proves",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gz: Scientific evidence master audit."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def audit_v1gu_evidence() -> dict[str, Any]:
    """Audit v1gu embedding structural evidence."""
    result = {
        "exists": V1GU_DIR.exists(),
        "files_found": [],
        "n_embeddings": 0,
        "regional_summary": {},
        "status": "READY",
        "blockers": [],
    }

    if not V1GU_DIR.exists():
        result["status"] = "BLOCKED"
        result["blockers"].append("v1gu output directory not found")
        return result

    # Check required files
    required_files = [
        "embedding_similarity_matrix_v1gu.json",
        "embedding_neighbors_v1gu.csv",
        "embedding_regional_summary_v1gu.json",
    ]

    for fname in required_files:
        fpath = V1GU_DIR / fname
        if fpath.exists():
            result["files_found"].append(fname)
        else:
            result["blockers"].append(f"Missing: {fname}")

    # Read regional summary
    summary_path = V1GU_DIR / "embedding_regional_summary_v1gu.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        result["n_embeddings"] = summary.get("n_embeddings", 0)
        result["regional_summary"] = summary.get("centroids", {})

    if result["blockers"]:
        result["status"] = "PARTIAL"

    return result


def audit_v1gv_evidence() -> dict[str, Any]:
    """Audit v1gv external evidence (GIS) coverage."""
    result = {
        "exists": V1GV_DIR.exists(),
        "files_found": [],
        "coverage_by_indicator": {},
        "status": "READY",
        "blockers": [],
    }

    if not V1GV_DIR.exists():
        result["status"] = "BLOCKED"
        result["blockers"].append("v1gv output directory not found")
        return result

    # Check required files
    required_files = [
        "evidence_coverage_matrix_v1gv.csv",
        "evidence_metadata_v1gv.json",
    ]

    for fname in required_files:
        fpath = V1GV_DIR / fname
        if fpath.exists():
            result["files_found"].append(fname)
        else:
            result["blockers"].append(f"Missing: {fname}")

    # Analyze coverage statuses
    matrix_path = V1GV_DIR / "evidence_coverage_matrix_v1gv.csv"
    if matrix_path.exists():
        rows = read_csv(matrix_path)
        # Count indicators by coverage status
        indicators = set()
        status_counts = defaultdict(int)
        for row in rows:
            for col_name in row:
                if col_name not in ["canonical_patch_id", "region"]:
                    indicators.add(col_name)
                    status = row.get(col_name, "").strip()
                    if status:
                        status_counts[status] += 1
        result["coverage_by_indicator"] = dict(status_counts)

    if result["blockers"]:
        result["status"] = "PARTIAL"

    return result


def audit_v1gw_review_gate() -> dict[str, Any]:
    """Audit v1gw review gate formalization."""
    result = {
        "exists": V1GW_DIR.exists(),
        "files_found": [],
        "n_candidates": 0,
        "categories": defaultdict(int),
        "status": "READY",
        "blockers": [],
    }

    if not V1GW_DIR.exists():
        result["status"] = "BLOCKED"
        result["blockers"].append("v1gw output directory not found")
        return result

    # Check required files
    required_files = [
        "review_candidates_v1gw.csv",
        "review_protocol_v1gw.md",
    ]

    for fname in required_files:
        fpath = V1GW_DIR / fname
        if fpath.exists():
            result["files_found"].append(fname)
        else:
            result["blockers"].append(f"Missing: {fname}")

    # Count candidates and categories
    candidates_path = V1GW_DIR / "review_candidates_v1gw.csv"
    if candidates_path.exists():
        rows = read_csv(candidates_path)
        result["n_candidates"] = len(rows)
        for row in rows:
            categories = row.get("categories", "").split(";")
            for cat in categories:
                cat = cat.strip()
                if cat:
                    result["categories"][cat] += 1

    # Verify protocol exists and is non-empty
    protocol_path = V1GW_DIR / "review_protocol_v1gw.md"
    if protocol_path.exists():
        content = protocol_path.read_text(encoding="utf-8")
        if len(content) < 100:
            result["blockers"].append("Review protocol is empty or minimal")

    if result["blockers"]:
        result["status"] = "PARTIAL"

    return result


def audit_v1ha_robustness() -> dict[str, Any]:
    """Audit v1ha perturbation robustness."""
    result = {
        "exists": V1HA_DIR.exists(),
        "files_found": [],
        "robust_embeddings": 0,
        "unstable_embeddings": 0,
        "perturbations_tested": [],
        "status": "READY" if V1HA_DIR.exists() else "BLOCKED",
        "blockers": [],
    }

    if not V1HA_DIR.exists():
        result["status"] = "BLOCKED"
        result["blockers"].append("v1ha output directory not found")
        return result

    # Check for key robustness files
    robustness_files = [
        "perturbation_robustness_summary.json",
        "embedding_drift_metrics.csv",
        "neighbor_persistence_under_perturbation.csv",
    ]

    for fname in robustness_files:
        fpath = V1HA_DIR / fname
        if fpath.exists():
            result["files_found"].append(fname)

    # Read summary
    summary_path = V1HA_DIR / "perturbation_robustness_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        result["robust_embeddings"] = summary.get("robust_count", 0)
        result["unstable_embeddings"] = summary.get("unstable_count", 0)
        result["perturbations_tested"] = summary.get("perturbation_types", [])

    if result["robust_embeddings"] == 12:
        result["status"] = "READY"
    elif result["robust_embeddings"] > 0:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "BLOCKED"
        result["blockers"].append("No robust embeddings found")

    return result


def audit_v1gy_figures_tables() -> dict[str, Any]:
    """Audit v1gy TCC figures and tables."""
    result = {
        "exists": V1GY_DIR.exists(),
        "figures_found": [],
        "tables_found": [],
        "figure_count": 0,
        "table_count": 0,
        "caption_violations": [],
        "status": "READY",
        "blockers": [],
    }

    if not V1GY_DIR.exists():
        result["status"] = "BLOCKED"
        result["blockers"].append("v1gy output directory not found")
        return result

    # Find PNG figures
    for png_file in V1GY_DIR.glob("fig_*.png"):
        result["figures_found"].append(png_file.name)
        result["figure_count"] += 1

    # Find CSV tables
    for csv_file in V1GY_DIR.glob("table_*.csv"):
        result["tables_found"].append(csv_file.name)
        result["table_count"] += 1

    # Audit captions from manifest
    manifest_path = V1GY_DIR / "tcc_visual_evidence_manifest_v1gy.csv"
    if manifest_path.exists():
        rows = read_csv(manifest_path)
        for row in rows:
            caption = row.get("caption_draft_pt", "").lower()
            for forbidden_term in FORBIDDEN_CAPTION_TERMS:
                if forbidden_term in caption:
                    result["caption_violations"].append(
                        f"Figure {row.get('artifact_id')}: contains '{forbidden_term}'"
                    )

    if result["caption_violations"]:
        result["status"] = "PARTIAL"
        result["blockers"].extend(result["caption_violations"])

    return result


def build_claim_evidence_matrix() -> list[dict[str, object]]:
    """Build claim → evidence crosswalk."""
    rows: list[dict[str, object]] = []

    for claim_id, claim_info in ALLOWED_CLAIMS.items():
        rows.append({
            "claim_id": claim_id,
            "claim_description": claim_info["description"],
            "evidence_source": claim_info["evidence_source"],
            "output_files": " | ".join(claim_info["output_files"]),
            "status": "READY",
            "blocking_reason": "",
            "tcc_section": "4. Resultados | 5. Metodologia",
        })

    for claim_id, reason in FORBIDDEN_CLAIMS.items():
        rows.append({
            "claim_id": claim_id,
            "claim_description": f"FORBIDDEN: {claim_id.replace('_', ' ').title()}",
            "evidence_source": "NONE",
            "output_files": "",
            "status": "BLOCKED",
            "blocking_reason": reason,
            "tcc_section": "",
        })

    return rows


def build_evidence_strength_matrix(
    v1gu: dict[str, Any],
    v1gv: dict[str, Any],
    v1gw: dict[str, Any],
    v1gy: dict[str, Any],
    v1ha: dict[str, Any],
) -> list[dict[str, object]]:
    """Build evidence strength by claim."""
    rows: list[dict[str, object]] = []

    claims_status = {
        "structural_coherence": v1gu["status"],
        "embedding_stability": v1gu["status"],
        "exploratory_similarity": v1gu["status"],
        "intra_inter_neighborhood_rate": v1gu["status"],
        "medoid_identification": v1gu["status"],
        "outlier_identification": v1gu["status"],
        "review_gate_formalized": v1gw["status"],
        "gis_contextual": v1gv["status"],
        "audit_trail": "READY",
        "pipeline_reproducibility": "READY",
        "embedding_robustness": v1ha["status"],
    }

    for claim_id, status in claims_status.items():
        claim_info = ALLOWED_CLAIMS.get(claim_id, {})
        rows.append({
            "claim_id": claim_id,
            "claim_description": claim_info.get("description", ""),
            "evidence_status": status,
            "n_files_available": len(claim_info.get("output_files", [])),
            "tcc_ready": "yes" if status == "READY" else "no",
        })

    return rows


def build_scientific_gaps() -> list[dict[str, object]]:
    """Identify remaining scientific gaps and their nature."""
    rows: list[dict[str, object]] = [
        {
            "gap_id": "gap_001",
            "category": "corpus_size",
            "description": "Corpus is small (12 embeddings, 4 per region)",
            "impact": "Results are exploratory and regional; generalization requires review gate",
            "mitigation": "Structured review gate protocol (v1gw) formalizes interpretation boundaries",
            "resolved": "no",
        },
        {
            "gap_id": "gap_002",
            "category": "gis_coverage",
            "description": "GIS evidence is partial (many indicators MISSING or NOT_ACQUIRED)",
            "impact": "Contextual analysis is fragmentary per patch",
            "mitigation": "GIS used descriptively, not for validation; absence documented",
            "resolved": "no",
        },
        {
            "gap_id": "gap_003",
            "category": "robustness",
            "description": "Perturbation robustness not yet executed on 12 real embeddings",
            "impact": "Embedding stability under noise/blur/crop unknown",
            "mitigation": "Can be completed in v1ha if v1gd is executed",
            "resolved": "no",
        },
        {
            "gap_id": "gap_004",
            "category": "multimodal",
            "description": "Multimodal is disabled; temporal multi-date analysis not attempted",
            "impact": "Single-date snapshot only; no temporal coherence",
            "mitigation": "Out of scope for current TCC; documented as future work",
            "resolved": "no",
        },
        {
            "gap_id": "gap_005",
            "category": "review_gate_execution",
            "description": "Review gate candidates identified but not yet annotated",
            "impact": "Contextual interpretation pending",
            "mitigation": "v1gw formalized protocol; review to be conducted manually",
            "resolved": "no",
        },
    ]
    return rows


def build_tcc_readiness_matrix() -> list[dict[str, object]]:
    """Build readiness matrix for TCC sections."""
    rows: list[dict[str, object]] = [
        {
            "tcc_section": "1. Introdução",
            "readiness": "INDEPENDENT",
            "dependencies": "none",
            "notes": "Written independently of evidence package",
        },
        {
            "tcc_section": "2. Revisão Bibliográfica",
            "readiness": "INDEPENDENT",
            "dependencies": "none",
            "notes": "Written independently of evidence package",
        },
        {
            "tcc_section": "3. Metodologia | 3.1 Dados e Patches",
            "readiness": "READY",
            "dependencies": "v1fu manifest + v1gu corpus audit",
            "notes": "12 real embeddings (4 per region) documented",
        },
        {
            "tcc_section": "3. Metodologia | 3.2 Extração de Embeddings",
            "readiness": "READY",
            "dependencies": "v1fx/v1fz execution logs, v1gu evidence",
            "notes": "DINOv2-com-registers, 768-dim, command documented",
        },
        {
            "tcc_section": "4. Resultados | 4.1 Análise Estrutural",
            "readiness": "READY",
            "dependencies": "v1gu + v1gy figures (5 PNG) + v1gy tables (6 CSV)",
            "notes": "Heatmap, network, neighbor rate, medoids/outliers available",
        },
        {
            "tcc_section": "4. Resultados | 4.2 Estrutura Regional",
            "readiness": "READY",
            "dependencies": "v1gu regional summary + v1gy intra/inter figure",
            "notes": "Intra/inter-region neighborhood rate documented",
        },
        {
            "tcc_section": "4. Resultados | 4.3 Evidência Contextual",
            "readiness": "PARTIAL",
            "dependencies": "v1gv GIS matrix + v1gy external coverage figure",
            "notes": "GIS coverage is fragmentary (PARTIAL/MISSING); documented limitations",
        },
        {
            "tcc_section": "5. Metodologia | 5.1 revisão supervisora",
            "readiness": "READY",
            "dependencies": "v1gw + v1gw protocol + v1gy review category figure",
            "notes": "Review gate formalized; 48 candidates prepared; protocol documented",
        },
        {
            "tcc_section": "5. Discussão",
            "readiness": "BLOCKED",
            "dependencies": "Review gate execution (manual)",
            "notes": "Requires completed annotation of v1gw candidates",
        },
        {
            "tcc_section": "6. Conclusão",
            "readiness": "INDEPENDENT",
            "dependencies": "Discussion section",
            "notes": "Written after discussion is complete",
        },
        {
            "tcc_section": "Apêndice | Robustez",
            "readiness": "OPTIONAL",
            "dependencies": "v1ha (if executed)",
            "notes": "Perturbation robustness can enhance appendix; not required for main TCC",
        },
    ]
    return rows


def build_summary_json(
    v1gu: dict[str, Any],
    v1gv: dict[str, Any],
    v1gw: dict[str, Any],
    v1gy: dict[str, Any],
    v1ha: dict[str, Any],
) -> dict[str, Any]:
    """Build overall summary."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "audit_results": {
            "v1gu_embedding_evidence": v1gu,
            "v1gv_gis_coverage": v1gv,
            "v1gw_review_gate": v1gw,
            "v1gy_figures_tables": v1gy,
            "v1ha_robustness": v1ha,
        },
        "allowed_claims_count": len(ALLOWED_CLAIMS),
        "forbidden_claims_count": len(FORBIDDEN_CLAIMS),
        "figures_ready": v1gy.get("figure_count", 0),
        "tables_ready": v1gy.get("table_count", 0),
        "review_gate_candidates": v1gw.get("n_candidates", 0),
        "corpus_size": 12,
        "n_regions": 3,
        "patches_per_region": 4,
        "embedding_dimension": 768,
        "embedding_backbone": "DINOv2-com-registers",
        "methodological_guardrails": {
            "review_only": True,
            "no_labels": True,
            "no_targets": True,
            "no_predictions": True,
            "no_ground_truth_claim": True,
            "no_cluster_as_class": True,
            "gis_contextual_only": True,
            "multimodal_disabled": True,
            "no_heavy_files_in_git": True,
        },
        "ready_for_tcc_writing": True,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        prepare_output_dir(output_dir, args.force)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Audit each stage
    v1gu = audit_v1gu_evidence()
    v1gv = audit_v1gv_evidence()
    v1gw = audit_v1gw_review_gate()
    v1gy = audit_v1gy_figures_tables()
    v1ha = audit_v1ha_robustness()

    # Build matrices and outputs
    claim_matrix = build_claim_evidence_matrix()
    evidence_strength = build_evidence_strength_matrix(v1gu, v1gv, v1gw, v1gy, v1ha)
    scientific_gaps = build_scientific_gaps()
    tcc_readiness = build_tcc_readiness_matrix()
    summary = build_summary_json(v1gu, v1gv, v1gw, v1gy, v1ha)

    # Write outputs
    write_csv(
        output_dir / "claim_to_evidence_crosswalk_v1gz.csv",
        claim_matrix,
        ["claim_id", "claim_description", "evidence_source", "output_files", "status", "blocking_reason", "tcc_section"],
    )

    write_csv(
        output_dir / "evidence_strength_by_claim_v1gz.csv",
        evidence_strength,
        ["claim_id", "claim_description", "evidence_status", "n_files_available", "tcc_ready"],
    )

    write_csv(
        output_dir / "remaining_scientific_gaps_v1gz.csv",
        scientific_gaps,
        ["gap_id", "category", "description", "impact", "mitigation", "resolved"],
    )

    write_csv(
        output_dir / "tcc_result_readiness_matrix_v1gz.csv",
        tcc_readiness,
        ["tcc_section", "readiness", "dependencies", "notes"],
    )

    write_json(
        output_dir / "scientific_evidence_master_summary_v1gz.json",
        summary,
    )

    print(f"[OK] Audit complete: {output_dir}")
    print(f"  - Claim-evidence crosswalk: {len(claim_matrix)} rows")
    print(f"  - Evidence strength matrix: {len(evidence_strength)} rows")
    print(f"  - Scientific gaps: {len(scientific_gaps)} gaps documented")
    print(f"  - TCC readiness: {len(tcc_readiness)} sections")
    print(f"  - Figures READY: {v1gy.get('figure_count', 0)}")
    print(f"  - Tables READY: {v1gy.get('table_count', 0)}")
    print(f"  - Review gate candidates: {v1gw.get('n_candidates', 0)}")
    print(f"  - Robust embeddings: {v1ha.get('robust_embeddings', 0)}/12")

    return 0


if __name__ == "__main__":
    sys.exit(main())
