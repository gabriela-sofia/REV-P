"""
SUSC-07A Event Evidence Catalog Builder (review-only)

Normalizes documentary/event evidence found in the repository into a single
auditable catalog. Reads the SUSC-07A scan and the real Protocol C event
registries (observed event registry + external candidate event index) WITHOUT
inventing anything. Every record gets a confidence level and an explicit
limitation; nothing becomes ground truth.

Reads (when present):
  - outputs_public/suscetibilidade/SUSC_07A_event_source_scan.csv
  - outputs_public/tables/revp_observed_event_registry_v2dz.csv
  - outputs_public/tables/revp_indice_eventos_externos_candidatos_mv1.csv

Writes:
  - datasets/suscetibilidade/susc_07_event_evidence_catalog_v1.csv
  - manifests/suscetibilidade/susc_07_event_evidence_catalog_manifest_v1.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCAN_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_event_source_scan.csv"
OBSERVED = ROOT / "outputs_public" / "tables" / "revp_observed_event_registry_v2dz.csv"
EXTERNAL = ROOT / "outputs_public" / "tables" / "revp_indice_eventos_externos_candidatos_mv1.csv"

OUT_CSV = ROOT / "datasets" / "suscetibilidade" / "susc_07_event_evidence_catalog_v1.csv"
OUT_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_07_event_evidence_catalog_manifest_v1.json"
)

CATALOG_FIELDS = [
    "evidence_id", "region", "event_type", "event_subtype", "date_or_period",
    "source_path", "source_type", "has_geometry", "geometry_status",
    "temporal_status", "spatial_precision", "confidence_level", "can_link_to_patch",
    "can_be_ground_truth", "allowed_for_training", "review_only",
    "evidence_status", "notes",
]

REGION_NORM = {
    "rec": "recife", "recife": "recife", "pe": "recife", "olinda/pe": "recife",
    "pet": "petropolis", "petropolis": "petropolis", "petrópolis": "petropolis",
    "cur": "curitiba", "curitiba": "curitiba",
}


def norm_region(raw: str) -> str:
    return REGION_NORM.get((raw or "").strip().lower(), "unknown")


def truthy(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "sim"}


def load_csv(path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def temporal_status(precision: str, date: str) -> str:
    p = (precision or "").upper()
    if "DAY" in p or "EXACT" in p:
        return "day"
    if "RANGE" in p or "WINDOW" in p or "PERIOD" in p:
        return "period"
    if "MONTH" in p:
        return "month"
    if "YEAR" in p:
        return "year"
    return "day" if date else "unknown"


def derive(region, has_geom, temporal, date):
    """Return (spatial_precision, confidence_level, can_link, evidence_status)."""
    # No usable coordinates exist in these registries -> never patch-linkable.
    if region == "unknown":
        return "unknown", "insufficient", False, "insufficient_for_patch_event_link"
    if has_geom:
        # geometry flagged but unvalidated / no coordinate column -> approximate
        spatial = "municipality"
        conf = "medium" if temporal in {"day", "period"} else "low"
        return spatial, conf, False, "approximate_spatial_evidence"
    if temporal in {"day", "period", "month", "year"} and date:
        return "region", "low", False, "region_period_context"
    return "text_only", "insufficient", False, "documentary_context_only"


def main() -> int:
    print("=" * 60)
    print("SUSC-07A Event Evidence Catalog Builder (review-only)")
    print("=" * 60)

    records = []
    sources_read = []

    # ---- observed event registry (primary, real) ----
    if OBSERVED.exists():
        sources_read.append(str(OBSERVED.relative_to(ROOT)).replace("\\", "/"))
        for i, r in enumerate(load_csv(OBSERVED)):
            region = norm_region(r.get("region", ""))
            has_geom = truthy(r.get("observed_geometry_available"))
            date = (r.get("event_start_date") or "").strip()
            end = (r.get("event_end_date") or "").strip()
            temporal = temporal_status(r.get("event_date_precision", ""), date)
            spatial, conf, can_link, status = derive(region, has_geom, temporal, date)
            date_or_period = date if date == end or not end else f"{date}..{end}"
            records.append({
                "evidence_id": f"EVID_OBS_{i:04d}",
                "region": region,
                "event_type": "flood_or_disaster_record",
                "event_subtype": (r.get("source_family") or "unknown"),
                "date_or_period": date_or_period or "unknown",
                "source_path": "outputs_public/tables/revp_observed_event_registry_v2dz.csv",
                "source_type": "official" if "CPRM" in (r.get("source_id") or "") else "manual_registry",
                "has_geometry": str(has_geom).lower(),
                "geometry_status": "flagged_unvalidated_no_coordinate" if has_geom else "no_geometry",
                "temporal_status": temporal,
                "spatial_precision": spatial,
                "confidence_level": conf,
                "can_link_to_patch": str(can_link).lower(),
                "can_be_ground_truth": "false",
                "allowed_for_training": "false",
                "review_only": "true",
                "evidence_status": status,
                "notes": (r.get("registry_status") or "")[:80],
            })

    # ---- external candidate event index (secondary, real) ----
    if EXTERNAL.exists():
        sources_read.append(str(EXTERNAL.relative_to(ROOT)).replace("\\", "/"))
        for i, r in enumerate(load_csv(EXTERNAL)):
            region = norm_region(r.get("regiao") or r.get("cidade", ""))
            has_geom = truthy(r.get("geometria_disponivel"))
            date = (r.get("data_evento_inicio") or "").strip()
            end = (r.get("data_evento_fim") or "").strip()
            temporal = temporal_status(r.get("janela_temporal_status", ""), date)
            spatial, conf, can_link, status = derive(region, has_geom, temporal, date)
            date_or_period = date if date == end or not end else f"{date}..{end}"
            records.append({
                "evidence_id": f"EVID_EXT_{i:04d}",
                "region": region,
                "event_type": (r.get("categoria_evento") or "flood_or_disaster_record"),
                "event_subtype": (r.get("descricao_evento") or "")[:60] or "unknown",
                "date_or_period": date_or_period or "unknown",
                "source_path": "outputs_public/tables/revp_indice_eventos_externos_candidatos_mv1.csv",
                "source_type": "manual_registry",
                "has_geometry": str(has_geom).lower(),
                "geometry_status": (r.get("geometria_tipo") or "no_geometry"),
                "temporal_status": temporal,
                "spatial_precision": spatial,
                "confidence_level": conf,
                "can_link_to_patch": str(can_link).lower(),
                "can_be_ground_truth": "false",
                "allowed_for_training": "false",
                "review_only": "true",
                "evidence_status": status,
                "notes": (r.get("motivo_nao_label") or "")[:80],
            })

    if not records:
        print("STOP: no real event registry found to catalog.")
        return 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(records)

    by_region = Counter(r["region"] for r in records)
    with_date = sum(1 for r in records if r["date_or_period"] != "unknown")
    with_geom = sum(1 for r in records if r["has_geometry"] == "true")
    linkable = sum(1 for r in records if r["can_link_to_patch"] == "true")
    by_status = Counter(r["evidence_status"] for r in records)

    manifest = {
        "artifact": "datasets/suscetibilidade/susc_07_event_evidence_catalog_v1.csv",
        "sources_read": sources_read,
        "scan_input": str(SCAN_CSV.relative_to(ROOT)).replace("\\", "/") if SCAN_CSV.exists() else None,
        "n_evidence": len(records),
        "by_region": dict(by_region),
        "with_date": with_date,
        "with_geometry": with_geom,
        "linkable_to_patch": linkable,
        "by_evidence_status": dict(by_status),
        "allowed_for_training": False,
        "can_be_used_as_ground_truth": False,
        "review_only": True,
        "scientific_status": "documentary_event_evidence_not_patch_ground_truth",
        "disclaimer": (
            "Catalogo review-only de evidencia documental/evento. NAO e ground truth "
            "por patch; sem coordenada precisa, can_link_to_patch=false."
        ),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nevidence cataloged: {len(records)}")
    print("by region:", dict(by_region))
    print(f"with date: {with_date} | with geometry: {with_geom} | linkable to patch: {linkable}")
    print("by status:", dict(by_status))
    print("\nNo ground truth. review-only. No patch-level confirmation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
