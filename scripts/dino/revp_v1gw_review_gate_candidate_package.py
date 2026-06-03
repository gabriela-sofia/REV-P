"""REV-P v1gw: Review gate candidate package.

Formalizes review gate as a methodological stage. Selects candidate
patches from structural evidence (v1gu) and GIS coverage (v1gv).

When v1gu embeddings are blocked, falls back to manifest-derived candidates
using geometry status and GIS coverage as selection criteria — with
explicit documentation of why embedding-based selection is unavailable.

Field mapping:
  v1fu manifest: canonical_patch_id, region
  v1gu blocker:  embedding_patch_status_v1gu.csv -> canonical_patch_id, embedding_status
  v1gv matrix:   evidence_coverage_matrix_v1gv.csv -> canonical_patch_id + indicators

Allowed claims: structural evidence informs candidate selection
Forbidden: candidates are pre-classified; review assigns labels
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
PHASE = "v1gw"
PHASE_NAME = "REVIEW_GATE_CANDIDATE_PACKAGE"

DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gw"
DEFAULT_INPUT_MANIFEST = (
    ROOT / "manifests" / "dino_inputs"
    / "revp_v1fu_dino_sentinel_input_manifest"
    / "dino_sentinel_input_manifest_v1fu.csv"
)
DEFAULT_V1GU_BLOCKER = (
    ROOT / "local_runs" / "dino_embeddings" / "v1gu"
    / "embedding_structural_evidence_blocker_v1gu.json"
)
DEFAULT_V1GU_REGIONAL = (
    ROOT / "local_runs" / "dino_embeddings" / "v1gu"
    / "embedding_regional_summary_v1gu.json"
)
DEFAULT_V1GV_MATRIX = (
    ROOT / "local_runs" / "dino_embeddings" / "v1gv"
    / "evidence_coverage_matrix_v1gv.csv"
)
DEFAULT_PATCH_EXTENT = (
    ROOT / "local_runs" / "dino_embeddings" / "v1gt"
    / "land_use_coverage_patch_extent_v1gt.csv"
)

FIELD_PATCH_ID = "canonical_patch_id"
FIELD_REGION = "region"

METHODOLOGICAL_GUARDRAILS: dict[str, Any] = {
    "review_only": True,
    "review_gate_is_methodological_stage": True,
    "automatic_classification_forbidden": True,
    "candidates_are_not_classified": True,
    "review_is_not_validation": True,
    "review_does_not_create_labels": True,
}

CANDIDATE_CATEGORIES = [
    "medoid_regional",
    "outlier_structural",
    "bridge_inter_regional",
    "coherence_embedding_gis_high",
    "conflict_embedding_gis",
    "coverage_external_low",
    "geometry_incomplete",
    "geometry_complete",
]

REVIEW_PROTOCOL = """# Review Gate Protocol (v1gw)

## Purpose
Review gate formalizes structural evidence interpretation by visual inspection
and domain expertise. Reviewers assess candidate patches to build interpretive
understanding of the embedding space and GIS coverage — NOT to assign labels,
classify vulnerability, or validate model outputs.

## This review does NOT:
- Validate DINO embeddings against ground truth
- Create vulnerability labels or flood susceptibility scores
- Assign risk predictions or hazard classes
- Measure model performance

## Candidate selection basis

Candidates are selected by one or more criteria:

| Category | Basis |
|---|---|
| medoid_regional | Patch closest to regional embedding centroid (from v1gu) |
| outlier_structural | Patch distant from regional centroid (from v1gu) |
| bridge_inter_regional | Patch similar to patches from other regions (from v1gu) |
| coherence_embedding_gis_high | Consistent signals from embedding + GIS indicators |
| conflict_embedding_gis | Divergent signals between embedding and GIS |
| coverage_external_low | Patch with few AVAILABLE/PARTIAL indicators |
| geometry_incomplete | Patch with CENTROID_ONLY geometry (no bounding box) |
| geometry_complete | Patch with complete geometry (usable as reference) |

When embedding evidence is unavailable (v1gu BLOCKED), candidates are derived
from manifest-level attributes: geometry status and GIS indicator coverage.

## Per-patch review checklist

For each candidate:

1. Visual inspection (Sentinel RGB / NDVI)
   - [ ] Land use type observable (urban, vegetation, water, bare soil)
   - [ ] Presence of water bodies or drainage features
   - [ ] Infrastructure characteristics (roads, buildings)
   - [ ] Data quality (cloud cover, shadows, acquisition artifacts)

2. Geographic context
   - [ ] Proximity to rivers or water bodies (qualitative)
   - [ ] Topographic context (slope, elevation from DEM where available)
   - [ ] Administrative/regional context

3. Structural signals
   - [ ] DINO nearest neighbors: visually similar? (if v1gu available)
   - [ ] GIS indicator consistency: does the coverage make sense?
   - [ ] Geometry completeness: does the patch have spatial grounding?

4. Observations
   - [ ] Notable visual patterns or anomalies
   - [ ] Data quality issues (artifacts, misalignment)
   - [ ] Recommendations for follow-up

## Output

Reviewer annotations go in: `review_candidates_v1gw_annotated_TEMPLATE.csv`
(copy and fill; original template is read-only for reproducibility)

## Guidance

- Be specific; reference specific GIS indicators or neighbors
- "Unclear" is better than guessing
- Do NOT assign vulnerability or susceptibility
- Focus on structural and contextual evidence only
"""

# Region normalization
REGION_ALIASES: dict[str, str] = {
    "Petropolis": "Petropolis",
    "Petrópolis": "Petropolis",
    "Curitiba": "Curitiba",
    "Recife": "Recife",
}


def normalize_region(raw: str) -> str:
    return REGION_ALIASES.get(raw.strip(), raw.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gw review gate candidate package."
    )
    parser.add_argument("--mode", default="candidate-generation-run",
                        choices=["candidate-generation-run"])
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--v1gu-regional", default=str(DEFAULT_V1GU_REGIONAL))
    parser.add_argument("--v1gu-blocker", default=str(DEFAULT_V1GU_BLOCKER))
    parser.add_argument("--v1gv-matrix", default=str(DEFAULT_V1GV_MATRIX))
    parser.add_argument("--patch-extent", default=str(DEFAULT_PATCH_EXTENT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def load_geometry_status(patch_extent_path: Path) -> dict[str, str]:
    """Load geometry_status per canonical_patch_id from v1gt patch extent."""
    if not patch_extent_path.exists():
        return {}
    rows = read_csv(patch_extent_path)
    return {r["canonical_patch_id"]: r.get("geometry_status", "") for r in rows
            if r.get("canonical_patch_id")}


def load_coverage_matrix(v1gv_path: Path) -> dict[str, dict[str, str]]:
    """Load coverage matrix: canonical_patch_id -> {indicator: status}."""
    if not v1gv_path.exists():
        return {}
    rows = read_csv(v1gv_path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = row.get("canonical_patch_id", "").strip()
        if pid:
            result[pid] = {k: v for k, v in row.items() if k not in
                           {"canonical_patch_id", "region"}}
    return result


def load_medoids_from_v1gu(v1gu_path: Path) -> dict[str, dict[str, Any]]:
    """Load medoids and outliers from v1gu regional summary (when available)."""
    if not v1gu_path.exists():
        return {}
    data = read_json(v1gu_path)
    return data.get("medoids_and_outliers", {})


def count_available_indicators(indicators: dict[str, str]) -> tuple[int, int]:
    """
    Return (n_available, n_total) for a patch's indicator dict.
    Only counts non-empty indicator values (other-region columns are blank).
    """
    available_statuses = {"AVAILABLE", "PARTIAL", "LOCAL_ONLY"}
    meta_keys = {"region", "region_normalized", "n_indicators", "canonical_patch_id"}
    valid = {k: v for k, v in indicators.items()
             if k not in meta_keys and v.strip() != ""}
    n_total = len(valid)
    n_avail = sum(1 for v in valid.values() if v in available_statuses)
    return n_avail, n_total


def select_review_candidates_with_embeddings(
    manifest: list[dict[str, str]],
    medoids: dict[str, dict[str, Any]],
    coverage: dict[str, dict[str, str]],
    geometry_status: dict[str, str],
) -> list[dict[str, Any]]:
    """Select candidates using embedding evidence (v1gu available)."""
    candidates: dict[str, dict[str, Any]] = {}

    for region, med_info in medoids.items():
        pid = med_info.get("medoid")
        if pid and pid not in candidates:
            candidates[pid] = {
                "canonical_patch_id": pid,
                "region": region,
                "categories": ["medoid_regional"],
                "selection_basis": f"Regional medoid for {region} (v1gu)",
                "embedding_evidence": "AVAILABLE",
            }
        for outlier_pid in med_info.get("outliers", []):
            if outlier_pid not in candidates:
                candidates[outlier_pid] = {
                    "canonical_patch_id": outlier_pid,
                    "region": region,
                    "categories": [],
                    "selection_basis": f"Structural outlier in {region} (v1gu)",
                    "embedding_evidence": "AVAILABLE",
                }
            if "outlier_structural" not in candidates[outlier_pid]["categories"]:
                candidates[outlier_pid]["categories"].append("outlier_structural")

    # Add coverage-based candidates
    for pid, inds in coverage.items():
        n_avail, n_total = count_available_indicators(inds)
        if n_total > 0 and n_avail / n_total < 0.4:
            if pid not in candidates:
                region = next(
                    (r[FIELD_REGION] for r in manifest if r.get(FIELD_PATCH_ID) == pid), ""
                )
                candidates[pid] = {
                    "canonical_patch_id": pid,
                    "region": region,
                    "categories": [],
                    "selection_basis": f"Low GIS coverage ({n_avail}/{n_total})",
                    "embedding_evidence": "AVAILABLE",
                }
            if "coverage_external_low" not in candidates[pid]["categories"]:
                candidates[pid]["categories"].append("coverage_external_low")

    return sorted(candidates.values(), key=lambda x: x["canonical_patch_id"])


def select_review_candidates_fallback(
    manifest: list[dict[str, str]],
    coverage: dict[str, dict[str, str]],
    geometry_status: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Fallback candidate selection when v1gu embeddings are blocked.
    Uses manifest-level attributes: geometry completeness and GIS coverage.
    """
    candidates: dict[str, dict[str, Any]] = {}

    # Candidates by geometry completeness
    for pid, gstatus in geometry_status.items():
        region = next(
            (r[FIELD_REGION] for r in manifest if r.get(FIELD_PATCH_ID) == pid), ""
        )
        if not region:
            continue
        if pid not in candidates:
            candidates[pid] = {
                "canonical_patch_id": pid,
                "region": region,
                "categories": [],
                "selection_basis": "",
                "embedding_evidence": "BLOCKED_NO_NPZ",
            }
        cat = "geometry_complete" if gstatus != "CENTROID_ONLY" else "geometry_incomplete"
        if cat not in candidates[pid]["categories"]:
            candidates[pid]["categories"].append(cat)
        candidates[pid]["selection_basis"] = f"geometry_status={gstatus}"

    # Candidates by GIS coverage (low coverage = needs review)
    for pid, inds in coverage.items():
        n_avail, n_total = count_available_indicators(inds)
        if n_total == 0:
            continue
        region = next(
            (r[FIELD_REGION] for r in manifest if r.get(FIELD_PATCH_ID) == pid), ""
        )
        if pid not in candidates:
            candidates[pid] = {
                "canonical_patch_id": pid,
                "region": region,
                "categories": [],
                "selection_basis": "",
                "embedding_evidence": "BLOCKED_NO_NPZ",
            }

        rate = n_avail / n_total
        if rate < 0.4:
            if "coverage_external_low" not in candidates[pid]["categories"]:
                candidates[pid]["categories"].append("coverage_external_low")
            candidates[pid]["selection_basis"] += f" | low_coverage={n_avail}/{n_total}"
        elif rate >= 0.6:
            # High coverage patches are good review anchors
            if "coherence_embedding_gis_high" not in candidates[pid]["categories"]:
                candidates[pid]["categories"].append("coherence_embedding_gis_high")
            candidates[pid]["selection_basis"] += f" | high_gis_coverage={n_avail}/{n_total}"

    # Ensure all regions have at least one candidate per region
    regions_represented: set[str] = {normalize_region(c["region"])
                                     for c in candidates.values() if c["categories"]}
    for row in manifest:
        pid = row.get(FIELD_PATCH_ID, "").strip()
        region = row.get(FIELD_REGION, "").strip()
        region_norm = normalize_region(region)
        if region_norm not in regions_represented and pid:
            candidates[pid] = {
                "canonical_patch_id": pid,
                "region": region,
                "categories": ["geometry_incomplete"],
                "selection_basis": "region_representative_fallback",
                "embedding_evidence": "BLOCKED_NO_NPZ",
            }
            regions_represented.add(region_norm)

    # Filter to only candidates with at least one category
    result = [c for c in candidates.values() if c["categories"]]
    result.sort(key=lambda x: x["canonical_patch_id"])
    return result


def export_candidates_csv(
    candidates: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    rows = [
        {
            "canonical_patch_id": c["canonical_patch_id"],
            "region": c["region"],
            "categories": "; ".join(c["categories"]),
            "n_categories": len(c["categories"]),
            "selection_basis": c.get("selection_basis", ""),
            "embedding_evidence": c.get("embedding_evidence", ""),
            "review_status": "NOT_REVIEWED",
            "visual_notes": "",
            "data_quality_notes": "",
            "gis_coherence_notes": "",
            "followup_recommended": "",
            "followup_notes": "",
        }
        for c in candidates
    ]
    fields = [
        "canonical_patch_id", "region", "categories", "n_categories",
        "selection_basis", "embedding_evidence", "review_status",
        "visual_notes", "data_quality_notes", "gis_coherence_notes",
        "followup_recommended", "followup_notes",
    ]
    write_csv(output_dir / "review_candidates_v1gw.csv", rows, fields)


def export_template_csv(
    candidates: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    rows = [
        {
            "canonical_patch_id": c["canonical_patch_id"],
            "categories": "; ".join(c["categories"]),
            "region": c["region"],
            "selection_basis": c.get("selection_basis", ""),
            "review_status": "",
            "visual_notes": "",
            "data_quality_notes": "",
            "gis_coherence_notes": "",
            "followup_recommended": "",
            "followup_notes": "",
        }
        for c in candidates
    ]
    fields = [
        "canonical_patch_id", "categories", "region", "selection_basis",
        "review_status", "visual_notes", "data_quality_notes",
        "gis_coherence_notes", "followup_recommended", "followup_notes",
    ]
    write_csv(
        output_dir / "review_candidates_v1gw_annotated_TEMPLATE.csv",
        rows, fields,
    )


def export_metadata(
    candidates: list[dict[str, Any]],
    embedding_mode: str,
    output_dir: Path,
) -> None:
    category_counts: dict[str, int] = {}
    for c in candidates:
        for cat in c["categories"]:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    region_counts: dict[str, int] = {}
    for c in candidates:
        region = normalize_region(c["region"])
        region_counts[region] = region_counts.get(region, 0) + 1

    data: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "embedding_mode": embedding_mode,
        "n_candidates": len(candidates),
        "category_counts": category_counts,
        "region_counts": region_counts,
        "allowed_categories": CANDIDATE_CATEGORIES,
        "methodological_guardrails": METHODOLOGICAL_GUARDRAILS,
    }
    write_json(output_dir / "review_candidates_metadata_v1gw.json", data)


def run_candidate_generation(args: argparse.Namespace) -> int:
    print(f"[{PHASE}] Starting review gate candidate package generation...")

    manifest_path = Path(args.input_manifest)
    if not manifest_path.exists():
        print(f"[!] ERROR: manifest not found: {manifest_path}")
        return 1

    manifest = [r for r in read_csv(manifest_path) if r.get(FIELD_PATCH_ID, "").strip()]
    print(f"[{PHASE}] Manifest: {len(manifest)} patches — {rel(manifest_path)}")

    coverage = load_coverage_matrix(Path(args.v1gv_matrix))
    print(f"[{PHASE}] Coverage matrix: {len(coverage)} patches")

    geometry_status = load_geometry_status(Path(args.patch_extent))
    print(f"[{PHASE}] Geometry status: {len(geometry_status)} patches")

    # Decide selection mode based on v1gu availability
    v1gu_regional_path = Path(args.v1gu_regional)
    v1gu_blocker_path = Path(args.v1gu_blocker)

    if v1gu_regional_path.exists():
        medoids = load_medoids_from_v1gu(v1gu_regional_path)
        n_emb = read_json(v1gu_regional_path).get("n_embeddings", 0)
        if n_emb >= 2:
            embedding_mode = "EMBEDDING_BASED"
            print(f"[{PHASE}] Using embedding-based selection ({n_emb} embeddings)")
            candidates = select_review_candidates_with_embeddings(
                manifest, medoids, coverage, geometry_status,
            )
        else:
            embedding_mode = "MANIFEST_FALLBACK"
            print(f"[{PHASE}] v1gu present but {n_emb} embeddings — using manifest fallback")
            candidates = select_review_candidates_fallback(manifest, coverage, geometry_status)
    else:
        embedding_mode = "MANIFEST_FALLBACK"
        print(f"[{PHASE}] v1gu regional summary not found — using manifest fallback")
        if v1gu_blocker_path.exists():
            blocker = read_json(v1gu_blocker_path)
            print(f"[{PHASE}] v1gu blocker: {blocker.get('blocker_code', '?')}")
        candidates = select_review_candidates_fallback(manifest, coverage, geometry_status)

    print(f"[{PHASE}] Candidates identified: {len(candidates)}")

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force, args.resume)
    print(f"[{PHASE}] Output: {rel(output_dir)}")

    export_candidates_csv(candidates, output_dir)
    export_template_csv(candidates, output_dir)
    export_metadata(candidates, embedding_mode, output_dir)
    write_markdown(output_dir / "review_protocol_v1gw.md", REVIEW_PROTOCOL)

    print(f"[{PHASE}] Review candidates: {len(candidates)} (mode={embedding_mode})")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_candidate_generation(args))
