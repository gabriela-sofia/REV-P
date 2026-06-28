"""
SUSC-14A FASE 6b - acquire public flood-footprint vector packages (controlled).

Reads the FASE 6a footprint source registry. Live acquisition only runs with
SUSC_13B_NETWORK=1 (offline => network_disabled). Accepts vector/documentary
formats (GeoJSON/SHP ZIP/KML/KMZ/GPKG/WKT/PDF). Never downloads heavy raster: a
raster-only / preview-only source is registered as
``footprint_unavailable_vector_required``.

Writes:
  - manifests/suscetibilidade/susc_14a_flood_footprint_manifest_v1.csv
  - datasets/suscetibilidade/flood_footprints_susc14a/  (raw, gitignored)
  - (discovery report is written by discover_susc_14a_flood_footprint_sources.py)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import ensure_dir, read_csv, rel, write_csv  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_14a_flood_footprint_source_registry_v1.csv"
DROP = ROOT / "datasets" / "suscetibilidade" / "flood_footprints_susc14a"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14a_flood_footprint_manifest_v1.csv"

FIELDS = [
    "footprint_id", "footprint_source_id", "region", "provider", "product",
    "event_reference", "event_period", "source_url", "acquisition_status",
    "http_status", "content_type", "file_path", "file_size_bytes", "sha256",
    "content_ext", "geometry_available", "blocked_reason", "requires_manual_review",
    "can_be_ground_truth", "allowed_for_training", "review_only", "notes",
]

RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf"}
VECTOR_EXTS = {".geojson", ".json", ".kml", ".kmz", ".gpkg", ".wkt", ".zip", ".shp"}


def _gov(row: dict) -> dict:
    row["requires_manual_review"] = "true"
    row["can_be_ground_truth"] = "false"
    row["allowed_for_training"] = "false"
    row["review_only"] = "true"
    return row


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Flood Footprint Acquisition")
    print("=" * 60)
    ensure_dir(DROP)
    if not REGISTRY.exists():
        print(f"STOP: missing footprint registry {rel(REGISTRY)}")
        return 1
    net = g14.network_enabled()
    rows: list[dict] = []
    for src in read_csv(REGISTRY):
        base = {
            "footprint_source_id": src.get("footprint_source_id", ""), "region": src.get("region", ""),
            "provider": src.get("provider", ""), "product": src.get("product", ""),
            "event_reference": src.get("event_reference", ""), "event_period": src.get("event_period", ""),
            "source_url": src.get("source_url", ""),
        }
        if src.get("download_candidate") != "true":
            rows.append(_gov({**base, "acquisition_status": "not_a_vector_candidate", "http_status": "",
                "content_type": "", "file_path": "", "file_size_bytes": "", "sha256": "",
                "content_ext": "", "geometry_available": "false", "blocked_reason": "raster_or_non_vector",
                "notes": "fonte sem expectativa de vetor publico"}))
            continue
        if not net:
            rows.append(_gov({**base, "acquisition_status": "network_disabled", "http_status": "",
                "content_type": "", "file_path": "", "file_size_bytes": "", "sha256": "",
                "content_ext": "", "geometry_available": "false", "blocked_reason": "",
                "notes": f"{g14.NETWORK_ENV} nao habilitado; aquisicao de footprint ignorada (offline-safe)"}))
            continue
        # network on: probe catalog root only (metadata). Direct EMS/DFO package
        # URLs are not invented; a real activation download needs manual selection
        # of the EMSR product, recorded here as a probe + manual-review pointer.
        res = g14.http_get(src.get("source_url", ""), accept="application/json")
        status = "probed_catalog" if res.get("status") == "ok" else str(res.get("status", "failed"))
        if src.get("raster_only_risk") == "true":
            status = "footprint_unavailable_vector_required"
        rows.append(_gov({**base, "acquisition_status": status,
            "http_status": str(res.get("http_code", "")), "content_type": str(res.get("content_type", "")),
            "file_path": "", "file_size_bytes": "", "sha256": "", "content_ext": "",
            "geometry_available": "false",
            "blocked_reason": "raster_only_preview" if status.startswith("footprint_unavailable") else "",
            "notes": "probe do catalogo; selecao do produto vetorial exige revisao manual; sem raster"}))

    for i, r in enumerate(rows):
        r["footprint_id"] = f"S14AFPDL_{i:03d}"
    rows = [{k: r.get(k, "") for k in FIELDS} for r in rows]
    write_csv(MANIFEST, rows, FIELDS)

    status = Counter(r["acquisition_status"] for r in rows)
    acquired = sum(1 for r in rows if r["geometry_available"] == "true")
    print(f"footprint manifest -> {rel(MANIFEST)} ({len(rows)} rows)")
    print(f"acquired vector footprints: {acquired} | status: {dict(status)}")
    print("review-only. No raster. No invented direct URL. No ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
