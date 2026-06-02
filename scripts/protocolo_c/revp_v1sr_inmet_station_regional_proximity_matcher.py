"""REV-P v1sr — INMET station regional proximity matcher.

Re-reads station metadata from INMET ZIPs with correct decimal parsing
(comma → dot for Brazilian locale). Computes geodetic distance from each
station to each target region centroid. Review-only.
"""
from __future__ import annotations
import argparse, zipfile
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p, raw_root,
    write_csv_with_header, write_schema, write_doc,
    guardrail_row, scan_guardrails,
    parse_float_safe, haversine_km, station_region_distances,
    nearest_region_and_distance, REGION_CENTROIDS, PROXIMITY_THRESHOLDS_KM,
    hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_PROX  = _p("REVP_V1SR_OUT_PROX",  DATASETS / "protocol_c_inmet_station_region_proximity_v1sr.csv")
OUT_SUM   = _p("REVP_V1SR_OUT_SUM",   DATASETS / "protocol_c_inmet_station_region_proximity_summary_v1sr.csv")
SCHEMA_P  = _p("REVP_V1SR_SCHEMA_P",  SCHEMAS  / "protocol_c_inmet_station_region_proximity_v1sr_schema.csv")
SCHEMA_S  = _p("REVP_V1SR_SCHEMA_S",  SCHEMAS  / "protocol_c_inmet_station_region_proximity_summary_v1sr_schema.csv")
DOC       = _p("REVP_V1SR_DOC",       DOCS     / "revp_v1sr_inmet_station_regional_proximity_matcher.md")

PROX_FIELDS = [
    "station_id", "station_code", "station_name", "uf", "lat", "lon",
    "nearest_region", "distance_km",
    "within_25km", "within_50km", "within_100km",
    "proximity_status", "source_zip",
    "review_only", "does_not_validate_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_MAX_STATIONS = 1200
_HEADER_BYTES = 400   # read only first N bytes of each file for speed


def _read_stations_from_zip(zip_path: Path) -> dict[str, dict[str, Any]]:
    """Extract station metadata (code, name, UF, lat, lon) from ZIP headers."""
    stations: dict[str, dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.upper().endswith(".CSV"):
                    continue
                if len(stations) >= _MAX_STATIONS:
                    break
                try:
                    data = z.read(name)
                    text = data[:_HEADER_BYTES].decode("latin-1", errors="replace")
                    info: dict[str, Any] = {"zip": zip_path.name}
                    for ln in text.splitlines()[:10]:
                        lo = ln.lower()
                        val = ln.split(";", 1)[-1].strip() if ";" in ln else ""
                        if lo.startswith("codigo"):
                            info["code"] = val
                        elif lo.startswith("estacao"):
                            info["name"] = val[:60]
                        elif lo.startswith("uf"):
                            info["uf"] = val[:4]
                        elif lo.startswith("latitude"):
                            info["lat"] = parse_float_safe(val)
                        elif lo.startswith("longitude"):
                            info["lon"] = parse_float_safe(val)
                    if info.get("code") and info.get("lat", 0.0) != 0.0:
                        stations[info["code"]] = info
                except Exception:
                    continue
    except Exception:
        pass
    return stations


def run() -> dict[str, Any]:
    inmet_dir = raw_root() / "inmet" / "historical"
    all_stations: dict[str, dict[str, Any]] = {}

    # Read from the most complete year zip available; deduplicate by code.
    for year in ("2022", "2023", "2024", "2021", "2020", "2025", "2026"):
        zp = inmet_dir / f"inmet_{year}.zip"
        if not zp.exists():
            continue
        found = _read_stations_from_zip(zp)
        for code, info in found.items():
            if code not in all_stations:
                all_stations[code] = info
        if len(all_stations) >= _MAX_STATIONS:
            break

    rows: list[dict[str, Any]] = []
    for code, info in all_stations.items():
        lat = float(info.get("lat", 0.0))
        lon = float(info.get("lon", 0.0))
        nearest, dist = nearest_region_and_distance(lat, lon)
        dists = station_region_distances(lat, lon)
        min_dist = min(dists.values())

        if min_dist == float("inf"):
            prox_status = "NO_COORDINATES"
        elif min_dist <= 25:
            prox_status = "WITHIN_25KM"
        elif min_dist <= 50:
            prox_status = "WITHIN_50KM"
        elif min_dist <= 100:
            prox_status = "WITHIN_100KM"
        else:
            prox_status = "BEYOND_100KM"

        row: dict[str, Any] = {
            "station_id":   f"V1SR_ST_{code}",
            "station_code": code,
            "station_name": info.get("name", ""),
            "uf":           info.get("uf", ""),
            "lat":          f"{lat:.6f}",
            "lon":          f"{lon:.6f}",
            "nearest_region": nearest,
            "distance_km":  f"{min_dist:.2f}",
            "within_25km":  "true" if min_dist <= 25 else "false",
            "within_50km":  "true" if min_dist <= 50 else "false",
            "within_100km": "true" if min_dist <= 100 else "false",
            "proximity_status": prox_status,
            "source_zip":   info.get("zip", ""),
            "notes":        "",
        }
        row.update(guardrail_row())
        rows.append(row)

    violations = scan_guardrails(rows, "v1sr_prox")
    if violations:
        raise ValueError(f"Guardrail violations in v1sr: {violations[:3]}")

    write_csv_with_header(OUT_PROX, rows, PROX_FIELDS)
    write_schema(SCHEMA_P, PROX_FIELDS, "v1sr_proximity")

    within_100 = sum(1 for r in rows if r["within_100km"] == "true")
    within_50  = sum(1 for r in rows if r["within_50km"] == "true")
    within_25  = sum(1 for r in rows if r["within_25km"] == "true")
    by_region  = {reg: sum(1 for r in rows if r["nearest_region"] == reg and r["within_100km"] == "true")
                  for reg in REGION_CENTROIDS}
    summary = [
        {"stat_key": "stations_total",         "stat_value": str(len(rows))},
        {"stat_key": "stations_within_25km",   "stat_value": str(within_25)},
        {"stat_key": "stations_within_50km",   "stat_value": str(within_50)},
        {"stat_key": "stations_within_100km",  "stat_value": str(within_100)},
        {"stat_key": "stations_near_recife",   "stat_value": str(by_region.get("RECIFE",0))},
        {"stat_key": "stations_near_pet",      "stat_value": str(by_region.get("PET",0))},
        {"stat_key": "stations_near_curitiba", "stat_value": str(by_region.get("CURITIBA",0))},
        {"stat_key": "stage", "stat_value": "v1sr"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1sr_summary")

    write_doc(DOC, "v1sr — INMET Station Regional Proximity Matcher", [
        "## Objetivo",
        "Reler metadata de estacoes INMET dos ZIPs com parsing correto de "
        "coordenadas (decimal-virgula brasileira). Calcular distancia geodesica "
        "a cada regiao-alvo (Recife, Petropolis, Curitiba). Review-only.",
        "## Resultado",
        f"Estacoes: {len(rows)}. Dentro de 100km: {within_100}. "
        f"Recife: {by_region.get('RECIFE',0)}. PET: {by_region.get('PET',0)}. "
        f"Curitiba: {by_region.get('CURITIBA',0)}.",
        "## Limitacoes",
        "Proximidade de estacao nao valida evento. Distancia e criterio de "
        "relevancia contextual, nao de evidencia causal.",
    ])
    print(f"[v1sr] stations={len(rows)} within_100km={within_100} "
          f"recife={by_region.get('RECIFE',0)} pet={by_region.get('PET',0)} "
          f"curitiba={by_region.get('CURITIBA',0)}")
    return {"stations": len(rows), "within_100km": within_100,
            "by_region": by_region}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sr proximity matcher").parse_args()
    run()
