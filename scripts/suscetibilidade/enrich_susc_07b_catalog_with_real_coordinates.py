"""
SUSC-07B Catalog Enrichment with Real Coordinates (review-only)

Joins the SUSC-07A documentary evidence catalog with the SUSC-07B extracted real
coordinates AT REGION LEVEL. Because the extracted geometries are traceable but
not tied to a specific documentary evidence_id, enrichment records that
region-level real geometry is now available WITHOUT falsely linking each
documentary evidence to a specific coordinate. Nothing becomes ground truth.

Reads:
  - datasets/suscetibilidade/susc_07_event_evidence_catalog_v1.csv
  - datasets/suscetibilidade/susc_07b_real_event_coordinates_v1.csv

Writes:
  - datasets/suscetibilidade/susc_07b_event_evidence_geometry_enriched_v1.csv
  - manifests/suscetibilidade/susc_07b_event_geometry_enrichment_manifest_v1.json
  - outputs_public/suscetibilidade/SUSC_07B_event_geometry_enrichment_audit.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_07_event_evidence_catalog_v1.csv"
COORDS = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"

OUT_ENRICHED = (
    ROOT / "datasets" / "suscetibilidade" / "susc_07b_event_evidence_geometry_enriched_v1.csv"
)
OUT_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_07b_event_geometry_enrichment_manifest_v1.json"
)
OUT_AUDIT = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_event_geometry_enrichment_audit.csv"
)

ENRICHED_FIELDS = [
    "evidence_id", "region", "event_type", "date_or_period", "source_path",
    "original_geometry_status", "enriched_has_geometry", "enriched_geometry_type",
    "enriched_geometry_source", "enriched_lat", "enriched_lon", "enriched_bbox",
    "enriched_spatial_precision", "enrichment_method", "enrichment_confidence",
    "can_link_to_patch_after_enrichment", "can_be_ground_truth",
    "allowed_for_training", "review_only", "enrichment_notes",
]


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def union_bbox(bboxes):
    pts = []
    for b in bboxes:
        try:
            xs = [float(x) for x in b.split(";")]
            if len(xs) == 4:
                pts.append(xs)
        except ValueError:
            continue
    if not pts:
        return ""
    return ";".join(str(round(v, 6)) for v in [
        min(p[0] for p in pts), min(p[1] for p in pts),
        max(p[2] for p in pts), max(p[3] for p in pts)])


def main() -> int:
    print("=" * 60)
    print("SUSC-07B Catalog Enrichment with Real Coordinates (review-only)")
    print("=" * 60)

    for p in (CATALOG, COORDS):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    catalog = load_csv(CATALOG)
    coords = load_csv(COORDS)

    # region -> aggregated real geometry availability
    region_sources: dict[str, list[str]] = defaultdict(list)
    region_types: dict[str, set[str]] = defaultdict(set)
    region_bboxes: dict[str, list[str]] = defaultdict(list)
    for c in coords:
        reg = c["region"]
        region_sources[reg].append(c["source_name"])
        region_types[reg].add(c["geometry_type"])
        if c["bbox"]:
            region_bboxes[reg].append(c["bbox"])

    enriched_rows = []
    audit_rows = []
    n_enriched = 0
    for ev in catalog:
        region = ev["region"]
        srcs = region_sources.get(region, [])
        has_region_geom = bool(srcs)
        if has_region_geom:
            n_enriched += 1
            gtypes = ";".join(sorted(region_types.get(region, set())))
            sources = ";".join(sorted(set(srcs))[:6])
            bbox = union_bbox(region_bboxes.get(region, []))
            precision = "region_traceable_geometry_available"
            link_after = "false"  # documentary evidence still not tied to its own coordinate
            conf = "low"
            note = ("Geometria REAL rastreavel disponivel em nivel de regiao (fontes locais); "
                    "NAO vinculada a este evidence_id especifico -> requires manual review; "
                    "review-only, nao e ground truth.")
        else:
            gtypes = ""
            sources = ""
            bbox = ""
            precision = ev.get("spatial_precision", "unknown")
            link_after = "false"
            conf = "insufficient"
            note = "Sem geometria real rastreavel para a regiao nesta passada."
        enriched_rows.append({
            "evidence_id": ev["evidence_id"],
            "region": region,
            "event_type": ev.get("event_type", ""),
            "date_or_period": ev.get("date_or_period", "unknown"),
            "source_path": ev.get("source_path", ""),
            "original_geometry_status": ev.get("geometry_status", ""),
            "enriched_has_geometry": str(has_region_geom).lower(),
            "enriched_geometry_type": gtypes,
            "enriched_geometry_source": sources,
            "enriched_lat": "",
            "enriched_lon": "",
            "enriched_bbox": bbox,
            "enriched_spatial_precision": precision,
            "enrichment_method": "region_level_join_with_traceable_geometry" if has_region_geom else "no_geometry_join",
            "enrichment_confidence": conf,
            "can_link_to_patch_after_enrichment": link_after,
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "enrichment_notes": note,
        })
        audit_rows.append({
            "evidence_id": ev["evidence_id"], "region": region,
            "region_geometry_available": str(has_region_geom).lower(),
            "geometry_types": gtypes, "n_region_sources": len(srcs),
        })

    OUT_ENRICHED.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ENRICHED.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ENRICHED_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(enriched_rows)

    with OUT_AUDIT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["evidence_id", "region", "region_geometry_available",
                                          "geometry_types", "n_region_sources"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(audit_rows)

    manifest = {
        "artifact": "datasets/suscetibilidade/susc_07b_event_evidence_geometry_enriched_v1.csv",
        "catalog_source": "datasets/suscetibilidade/susc_07_event_evidence_catalog_v1.csv",
        "coordinates_source": "datasets/suscetibilidade/susc_07b_real_event_coordinates_v1.csv",
        "n_evidence": len(catalog),
        "n_enriched_with_region_geometry": n_enriched,
        "regions_with_real_geometry": sorted([r for r, s in region_sources.items() if s]),
        "can_be_used_as_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "scientific_status": "region_level_geometry_enrichment_not_patch_ground_truth",
        "disclaimer": (
            "Enriquecimento em nivel de regiao com geometria real rastreavel. NAO vincula "
            "evidencia documental a coordenada propria; nao e ground truth por patch."
        ),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nevidence rows: {len(catalog)} | enriched with region geometry: {n_enriched}")
    print("regions with real geometry:", manifest["regions_with_real_geometry"])
    print("\nRegion-level enrichment. No patch ground truth. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
