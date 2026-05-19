"""REV-P v1gv: External evidence coverage matrix.

Consolidates GIS and external evidence coverage per patch and region.
Reads the canonical patch manifest (canonical_patch_id field) and the
external evidence registry / v1gt overlap audit to populate real statuses.

Field mapping:
  v1fu manifest: canonical_patch_id, region
  v1gt overlap audit: patch_id (= canonical_patch_id), coverage_status

Allowed claims: GIS contextualizes territory; GIS is interpretable baseline
Forbidden: GIS is ground truth; GIS validates DINO; GIS proves vulnerability

Coverage status vocabulary (per indicator per patch):
  AVAILABLE       – data present, spatially validated for this patch
  PARTIAL         – data present regionally; patch-level validation pending
  BBOX_ONLY       – dataset bbox overlaps; no centroid/footprint confirmation
  BLOCKED         – data exists but blocked (CRS, format, dependency)
  NOT_ACQUIRED    – source identified; data not yet downloaded
  LOCAL_ONLY      – data local/private workspace; cannot be versioned
  MISSING         – source not available or not identified
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
PHASE = "v1gv"
PHASE_NAME = "EXTERNAL_EVIDENCE_COVERAGE_MATRIX"

DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gv"
DEFAULT_INPUT_MANIFEST = (
    ROOT / "manifests" / "dino_inputs"
    / "revp_v1fu_dino_sentinel_input_manifest"
    / "dino_sentinel_input_manifest_v1fu.csv"
)
DEFAULT_EVIDENCE_REGISTRY = ROOT / "datasets" / "external_evidence_registry.csv"
DEFAULT_V1GT_OVERLAP = (
    ROOT / "local_runs" / "dino_embeddings" / "v1gt"
    / "land_use_coverage_overlap_audit_v1gt.csv"
)

# Authoritative field name in v1fu manifest
FIELD_PATCH_ID = "canonical_patch_id"
FIELD_REGION = "region"

METHODOLOGICAL_GUARDRAILS: dict[str, Any] = {
    "review_only": True,
    "gis_is_ground_truth": False,
    "gis_validates_dino": False,
    "gis_contextualizes_territory": True,
    "gis_is_interpretable_baseline": True,
    "predictive_claims_from_gis": False,
}

COVERAGE_STATUS_VALUES = [
    "AVAILABLE",
    "PARTIAL",
    "BBOX_ONLY",
    "BLOCKED",
    "NOT_ACQUIRED",
    "LOCAL_ONLY",
    "MISSING",
]

# Indicator definitions with per-region status grounded in external_evidence_registry
# and v1gt audit outcomes. Sources documented; GIS is contextual, not ground truth.
REGIONAL_INDICATOR_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "Curitiba": [
        {
            "indicator_id": "terrain_geocuritiba",
            "name": "Terrain/DEM — GeoCuritiba (MDT/MDS)",
            "source": "Prefeitura de Curitiba / GeoCuritiba ArcGIS REST",
            "evidence_id": "curitiba_geocuritiba",
            "status": "PARTIAL",
            "notes": (
                "169 ArcGIS metadata records present; extent coords available. "
                "CRS MDS lacks numeric WKID. Patch-bound not validated. "
                "CUR_08–CUR_14 are placeholder patches without canonical geometry."
            ),
        },
        {
            "indicator_id": "land_use",
            "name": "Land Use / Land Cover",
            "source": "MapBiomas (candidate) — not yet acquired",
            "evidence_id": "mapbiomas_candidate",
            "status": "NOT_ACQUIRED",
            "notes": (
                "No FBDS layer for Curitiba region. MapBiomas (national LULC) "
                "identified as candidate but not downloaded."
            ),
        },
        {
            "indicator_id": "drainage",
            "name": "Drainage / hydrography",
            "source": "Not identified for Curitiba",
            "evidence_id": "",
            "status": "MISSING",
            "notes": "No drainage layer acquired or identified for Curitiba patches.",
        },
        {
            "indicator_id": "population_density",
            "name": "Population density (IBGE Census)",
            "source": "IBGE Censo 2022",
            "evidence_id": "",
            "status": "NOT_ACQUIRED",
            "notes": "IBGE grid not yet downloaded; source identified.",
        },
        {
            "indicator_id": "administrative_defesa_civil",
            "name": "Defesa Civil / risk zones",
            "source": "GeoIDC Curitiba",
            "evidence_id": "curitiba_geodc",
            "status": "MISSING",
            "notes": "Source not acquired; no evidence of access to this layer.",
        },
    ],
    "Petropolis": [
        {
            "indicator_id": "terrain_sgb_rigeo",
            "name": "Terrain — SGB/RIGeo (SHP + ESRI GRID)",
            "source": "SGB/CPRM — Serviço Geológico do Brasil",
            "evidence_id": "petropolis_sgb_rigeo",
            "status": "PARTIAL",
            "notes": (
                "Binary SHP headers inspected; dataset envelope bounds present. "
                "Individual patch footprint not validated. "
                "14 sidecars confirmed / 3 with warnings / 10 without sidecar."
            ),
        },
        {
            "indicator_id": "land_use_fbds",
            "name": "Land Use — FBDS (GeoJSON converted)",
            "source": "FBDS — Fundação Brasileira para o Desenvolvimento Sustentável",
            "evidence_id": "petropolis_fbds",
            "status": "PARTIAL",
            "notes": (
                "WGS84 bbox lon[-43.378,-42.978] lat[-22.575,-22.202]; 6861 features. "
                "Patches south of lat -22.575 not covered. CLASSE_USO is not ground truth."
            ),
        },
        {
            "indicator_id": "geological_cprm",
            "name": "Geological context — SGB/CPRM cartas",
            "source": "SGB/CPRM",
            "evidence_id": "petropolis_sgb_cprm_cartas",
            "status": "PARTIAL",
            "notes": (
                "Regional geological maps indexed. Scale 1:250k — insufficient "
                "for individual patch analysis. Background context only."
            ),
        },
        {
            "indicator_id": "drainage",
            "name": "Drainage / hydrography",
            "source": "Not identified for Petrópolis",
            "evidence_id": "",
            "status": "MISSING",
            "notes": "No drainage layer acquired or identified for Petrópolis patches.",
        },
        {
            "indicator_id": "population_density",
            "name": "Population density (IBGE Census)",
            "source": "IBGE Censo 2022",
            "evidence_id": "",
            "status": "NOT_ACQUIRED",
            "notes": "IBGE grid not yet downloaded; source identified.",
        },
    ],
    "Recife": [
        {
            "indicator_id": "terrain_pe3d",
            "name": "Terrain/DEM — PE3D (MDT + SC-25)",
            "source": "Programa PE3D — Pernambuco 3D",
            "evidence_id": "recife_pe3d_mde",
            "status": "PARTIAL",
            "notes": (
                "66 raster headers confirmed EPSG:31985 + dataset bounds. "
                "Stack: 48 MDT tiles + 7 SC-25. No pixel validation; "
                "no patch-level alignment confirmed."
            ),
        },
        {
            "indicator_id": "drainage_esig",
            "name": "Drainage — ESIG/EMLURB macrodrenagem",
            "source": "ESIG / EMLURB Recife",
            "evidence_id": "recife_esig_drainage",
            "status": "PARTIAL",
            "notes": (
                "Geometry layers present; CRS and projection manually verified. "
                "Scale and resolution not audited per patch. "
                "Covers municipality of Recife."
            ),
        },
        {
            "indicator_id": "land_use",
            "name": "Land Use / Land Cover",
            "source": "MapBiomas (candidate) — not yet acquired",
            "evidence_id": "mapbiomas_candidate",
            "status": "NOT_ACQUIRED",
            "notes": "No FBDS layer for Recife. MapBiomas candidate not downloaded.",
        },
        {
            "indicator_id": "population_density",
            "name": "Population density (IBGE Census)",
            "source": "IBGE Censo 2022",
            "evidence_id": "",
            "status": "NOT_ACQUIRED",
            "notes": "IBGE grid not yet downloaded; source identified.",
        },
        {
            "indicator_id": "coastal_context",
            "name": "Coastal proximity context",
            "source": "Derivable from canonical patch coordinates (qualitative)",
            "evidence_id": "",
            "status": "PARTIAL",
            "notes": (
                "Recife is coastal city; proximity to ocean is geographically "
                "evident. Exact metric requires patch centroid computation "
                "from geometry (geometry_status=CENTROID_ONLY for most patches)."
            ),
        },
    ],
}

# Region name normalization (manifest may use diacritics variant)
REGION_ALIASES: dict[str, str] = {
    "Petropolis": "Petropolis",
    "Petrópolis": "Petropolis",
    "petrópolis": "Petropolis",
    "petropolis": "Petropolis",
    "Curitiba": "Curitiba",
    "curitiba": "Curitiba",
    "Recife": "Recife",
    "recife": "Recife",
}


def normalize_region(raw: str) -> str:
    return REGION_ALIASES.get(raw.strip(), raw.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gv external evidence coverage matrix."
    )
    parser.add_argument(
        "--mode", default="coverage-matrix-run",
        choices=["coverage-matrix-run"],
    )
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--evidence-registry", default=str(DEFAULT_EVIDENCE_REGISTRY))
    parser.add_argument("--v1gt-overlap", default=str(DEFAULT_V1GT_OVERLAP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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



# Only source_ids from v1gt overlap audit that represent land-use layers.
# RJ_3303906_USO is the FBDS layer for Petrópolis (sole confirmed land-use source).
# SGB/CPRM hydro and terrain sources are excluded from land_use indicator.
LAND_USE_SOURCE_IDS = {"RJ_3303906_USO"}


def load_v1gt_land_use_status(v1gt_overlap_path: Path) -> dict[str, str]:
    """
    Build per-patch land-use coverage from v1gt overlap audit.
    Only considers source_ids that represent actual land-use layers.
    SGB/CPRM hydro and terrain overlaps are excluded — they are not land-use.

    Status mapping:
      POTENTIALLY_COVERED_BY_CENTROID (from land-use source) -> PARTIAL
      No land-use source match -> NOT_ACQUIRED
    """
    if not v1gt_overlap_path.exists():
        return {}
    rows = read_csv(v1gt_overlap_path)

    # Filter to land-use source_ids only
    by_patch: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        pid = r.get("patch_id", "").strip()
        source_id = r.get("source_id", "").strip()
        status = r.get("coverage_status", "").strip()
        if pid and source_id in LAND_USE_SOURCE_IDS and status:
            by_patch[pid].append(status)

    consolidated: dict[str, str] = {}
    for pid, statuses in by_patch.items():
        if "POTENTIALLY_COVERED_BY_CENTROID" in statuses or "COVERED" in statuses:
            consolidated[pid] = "PARTIAL"
        else:
            consolidated[pid] = "NOT_ACQUIRED"
    return consolidated


def build_coverage_matrix(
    patches: list[dict[str, str]],
    v1gt_land_use: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Build per-patch coverage matrix rows using real indicator status.
    Uses canonical_patch_id as the authoritative patch identifier.
    Integrates v1gt land-use coverage per patch when available.
    """
    rows: list[dict[str, Any]] = []

    for patch_row in patches:
        patch_id = patch_row.get(FIELD_PATCH_ID, "").strip()
        raw_region = patch_row.get(FIELD_REGION, "").strip()
        region_key = normalize_region(raw_region)

        if not patch_id:
            continue

        template_indicators = REGIONAL_INDICATOR_TEMPLATES.get(region_key, [])

        row: dict[str, Any] = {
            "canonical_patch_id": patch_id,
            "region": raw_region,
            "region_normalized": region_key,
            "n_indicators": len(template_indicators),
        }

        for ind in template_indicators:
            ind_id = ind["indicator_id"]
            ind_status = ind["status"]

            # Override land_use status with v1gt per-patch result when available
            if "land_use" in ind_id and patch_id in v1gt_land_use:
                ind_status = v1gt_land_use[patch_id]

            row[ind_id] = ind_status

        rows.append(row)

    return rows


def aggregate_by_region(
    matrix_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate coverage counts per region × indicator."""
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix_rows:
        by_region[row["region_normalized"]].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for region, rows in by_region.items():
        n_patches = len(rows)
        all_ind_ids: set[str] = set()
        for r in rows:
            all_ind_ids.update(k for k in r if k not in
                               {"canonical_patch_id", "region", "region_normalized", "n_indicators"})

        ind_counts: dict[str, dict[str, int]] = {}
        for ind_id in sorted(all_ind_ids):
            counts = {s: 0 for s in COVERAGE_STATUS_VALUES}
            for r in rows:
                status = r.get(ind_id, "MISSING")
                if status in counts:
                    counts[status] += 1
            ind_counts[ind_id] = counts

        summary[region] = {
            "n_patches": n_patches,
            "indicator_coverage": ind_counts,
        }
    return summary


def export_coverage_matrix_csv(
    matrix_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    # Collect all indicator columns
    all_ind_ids: list[str] = []
    seen: set[str] = set()
    meta_fields = {"canonical_patch_id", "region", "region_normalized", "n_indicators"}
    for row in matrix_rows:
        for k in row:
            if k not in meta_fields and k not in seen:
                all_ind_ids.append(k)
                seen.add(k)

    fields = ["canonical_patch_id", "region"] + all_ind_ids
    write_csv(output_dir / "evidence_coverage_matrix_v1gv.csv", matrix_rows, fields)


def export_regional_summary_csv(
    regional_stats: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for region in sorted(regional_stats):
        stats = regional_stats[region]
        for ind_id, counts in stats["indicator_coverage"].items():
            row: dict[str, Any] = {
                "region": region,
                "indicator_id": ind_id,
                "n_patches": stats["n_patches"],
            }
            row.update(counts)
            rows.append(row)

    fields = ["region", "indicator_id", "n_patches"] + COVERAGE_STATUS_VALUES
    write_csv(output_dir / "evidence_regional_summary_v1gv.csv", rows, fields)


def export_indicator_definitions(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for region_key, indicators in REGIONAL_INDICATOR_TEMPLATES.items():
        for ind in indicators:
            rows.append({
                "region": region_key,
                "indicator_id": ind["indicator_id"],
                "name": ind["name"],
                "source": ind["source"],
                "evidence_id": ind.get("evidence_id", ""),
                "default_status": ind["status"],
                "notes": ind.get("notes", ""),
            })

    fields = ["region", "indicator_id", "name", "source", "evidence_id",
              "default_status", "notes"]
    write_csv(output_dir / "indicator_definitions_v1gv.csv", rows, fields)


def export_metadata(
    n_patches: int,
    n_missing_patch_id: int,
    v1gt_loaded: int,
    output_dir: Path,
) -> None:
    data: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "n_patches_in_manifest": n_patches,
        "n_patches_without_patch_id": n_missing_patch_id,
        "v1gt_land_use_statuses_loaded": v1gt_loaded,
        "coverage_status_values": COVERAGE_STATUS_VALUES,
        "methodological_guardrails": METHODOLOGICAL_GUARDRAILS,
        "notes": {
            "gis_contextualizes": (
                "GIS data provides territorial context for patch analysis. "
                "GIS is NOT ground truth, NOT validation target, NOT proof of vulnerability."
            ),
            "status_meaning": {
                "AVAILABLE": "Spatially validated and accessible for this patch",
                "PARTIAL": "Present regionally; patch-level spatial validation pending",
                "BBOX_ONLY": "Dataset bbox overlaps; no centroid/footprint confirmation",
                "BLOCKED": "Data exists; access blocked by CRS/format/dependency issue",
                "NOT_ACQUIRED": "Source identified; data not yet downloaded",
                "LOCAL_ONLY": "Present in private local workspace; not versionable",
                "MISSING": "Source not identified or not available",
            },
        },
    }
    write_json(output_dir / "evidence_metadata_v1gv.json", data)


def run_coverage_matrix(args: argparse.Namespace) -> int:
    print(f"[{PHASE}] Starting external evidence coverage matrix...")

    manifest_path = Path(args.input_manifest)
    if not manifest_path.exists():
        print(f"[!] ERROR: manifest not found: {manifest_path}")
        return 1

    patches_raw = read_csv(manifest_path)
    patches = [r for r in patches_raw if r.get(FIELD_PATCH_ID, "").strip()]
    n_missing_id = len(patches_raw) - len(patches)
    print(f"[{PHASE}] Manifest: {len(patches)} patches with {FIELD_PATCH_ID} "
          f"({n_missing_id} skipped without id) — {rel(manifest_path)}")

    v1gt_path = Path(args.v1gt_overlap)
    v1gt_land_use = load_v1gt_land_use_status(v1gt_path)
    print(f"[{PHASE}] v1gt land-use statuses loaded: {len(v1gt_land_use)} patches")

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force, args.resume)
    print(f"[{PHASE}] Output: {rel(output_dir)}")

    print(f"[{PHASE}] Building coverage matrix...")
    matrix_rows = build_coverage_matrix(patches, v1gt_land_use)
    print(f"[{PHASE}] Matrix rows: {len(matrix_rows)}")

    print(f"[{PHASE}] Aggregating by region...")
    regional_stats = aggregate_by_region(matrix_rows)
    print(f"[{PHASE}] Regions: {sorted(regional_stats.keys())}")

    export_coverage_matrix_csv(matrix_rows, output_dir)
    export_regional_summary_csv(regional_stats, output_dir)
    export_indicator_definitions(output_dir)
    export_metadata(len(patches), n_missing_id, len(v1gt_land_use), output_dir)

    print(f"[{PHASE}] Coverage matrix: {len(matrix_rows)} rows")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_coverage_matrix(args))
