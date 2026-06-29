"""SUSC-15A FASE 7 - build event footprint candidates or future plans."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
GEOCODED = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
REFS = DAT / "susc_15a_precision_reference_features_v1.csv"
FOOTPRINTS = DAT / "susc_15a_event_footprint_candidates_v1.csv"
REPORT = OUT / "SUSC_15A_event_footprint_candidates_report.md"

FIELDS = [
    "footprint_candidate_id", "event_id", "region", "event_date", "candidate_class",
    "source_id", "source_url_or_path", "pre_window", "post_window", "aoi",
    "sar_before_after", "permanent_water_mask", "hand_slope_mask",
    "output_expected_polygon", "execution_status", "allowed_for_training",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "review_only", "notes",
]


def _plan(row: dict, idx: int, refs_by_region: dict[str, list[dict]]) -> dict:
    region = row.get("region", "")
    event_date = row.get("event_date", "")
    polygons = [r for r in refs_by_region.get(region, []) if r.get("feature_type") in {"official_event_polygon", "flood_polygon"}]
    if row.get("precision_tier") == "T0_official_event_polygon":
        cls = "official_vector_flood_footprint"
        source_id = row.get("source_name", "")
        source = row.get("source_url_or_path", "")
        status = "available_review_only"
    elif polygons:
        cls = "technical_vector_flood_footprint"
        source_id = polygons[0].get("source_id", "")
        source = polygons[0].get("geojson_ref", "")
        status = "candidate_requires_manual_event_date_review"
    elif event_date and region:
        cls = "remote_sensing_candidate_footprint_plan"
        source_id = "future_sentinel_plan"
        source = ""
        status = "not_executed_requires_future_authenticated_or_manual_run"
    else:
        cls = "no_vector_footprint_found"
        source_id = ""
        source = ""
        status = "blocked_no_date_or_region"
    return {
        "footprint_candidate_id": f"S15AFP_{idx:06d}",
        "event_id": row.get("event_id", ""),
        "region": region,
        "event_date": event_date,
        "candidate_class": cls,
        "source_id": source_id,
        "source_url_or_path": source,
        "pre_window": f"{event_date}-30d" if event_date else "",
        "post_window": f"{event_date}+10d" if event_date else "",
        "aoi": f"region={region}; patch matrix bbox scope" if region else "",
        "sar_before_after": "Sentinel-1 before/after plan only; no raster downloaded",
        "permanent_water_mask": "required in future execution",
        "hand_slope_mask": "required in future execution",
        "output_expected_polygon": "review-only flood extent polygon candidate",
        "execution_status": status,
        "allowed_for_training": "false",
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "review_only": "true",
        "notes": "No heavy raster acquisition in SUSC-15A.",
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Event Footprint Candidates")
    print("=" * 60)
    refs_by_region: dict[str, list[dict]] = {}
    for ref in read_csv(REFS) if REFS.exists() else []:
        refs_by_region.setdefault(ref.get("region", ""), []).append(ref)
    rows = [_plan(row, idx, refs_by_region) for idx, row in enumerate(read_csv(GEOCODED) if GEOCODED.exists() else [])]
    write_csv(FOOTPRINTS, rows, FIELDS)
    counts = Counter(r["candidate_class"] for r in rows)
    write_markdown(REPORT, f"""# SUSC-15A - candidatos de footprint observacional

Status: review-only. Nenhum raster pesado foi baixado ou processado.

- Eventos avaliados: **{len(rows)}**
- Classes: **{dict(counts)}**

Quando nao ha vetor oficial/tecnico, o artefato registra plano reprodutivel
para execucao futura com Sentinel-1/Sentinel-2, mascara de agua permanente e
HAND/slope. A execucao futura permanece bloqueada quando exigir autenticacao,
GEE ou aquisicao raster pesada.
""")
    print(f"footprint candidates: {len(rows)} | {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
