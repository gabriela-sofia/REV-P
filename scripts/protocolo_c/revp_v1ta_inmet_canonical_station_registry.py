"""REV-P v1ta — INMET canonical station registry.

Re-parses station metadata from ZIPs with correct decimal-comma handling.
Merges v1si (legacy, broken coords) and v1sr (corrected) into a single
canonical registry. Preserves provenance; marks coordinate quality.
"""
from __future__ import annotations
import argparse, zipfile
from pathlib import Path
from typing import Any

from revp_v1ta_v1tf_inmet_canonical_common import (
    DATASETS, DOCS, SCHEMAS, _p, raw_root,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row, scan_guardrails,
    parse_decimal_comma_float, detect_coordinate_anomaly,
    station_coordinate_quality_status, normalize_station_code, normalize_uf,
    nearest_region_and_distance, station_region_distances, PROXIMITY_THRESHOLDS_KM,
    hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_REG  = _p("REVP_V1TA_OUT_REG",  DATASETS / "protocol_c_inmet_canonical_station_registry_v1ta.csv")
OUT_SUM  = _p("REVP_V1TA_OUT_SUM",  DATASETS / "protocol_c_inmet_canonical_station_registry_summary_v1ta.csv")
SCHEMA_R = _p("REVP_V1TA_SCHEMA_R", SCHEMAS  / "protocol_c_inmet_canonical_station_registry_v1ta_schema.csv")
SCHEMA_S = _p("REVP_V1TA_SCHEMA_S", SCHEMAS  / "protocol_c_inmet_canonical_station_registry_summary_v1ta_schema.csv")
DOC      = _p("REVP_V1TA_DOC",      DOCS     / "revp_v1ta_inmet_canonical_station_registry.md")

REG_FIELDS = [
    "station_id", "station_code", "station_name", "uf",
    "latitude", "longitude", "raw_lat_text", "raw_lon_text",
    "coordinate_quality_status", "source_years", "provenance_sources",
    "nearest_region", "nearest_region_distance_km",
    "within_25km", "within_50km", "within_100km",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_HEADER_BYTES = 400
_MAX_STATIONS = 2000


def _parse_zip_stations(zip_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.upper().endswith(".CSV") or len(out) >= _MAX_STATIONS:
                    continue
                try:
                    text = z.read(name)[:_HEADER_BYTES].decode("latin-1", errors="replace")
                    info: dict[str, Any] = {}
                    for ln in text.splitlines()[:10]:
                        lo = ln.lower()
                        val = ln.split(";", 1)[-1].strip() if ";" in ln else ""
                        if lo.startswith("codigo"):    info["raw_code"] = val
                        elif lo.startswith("estacao"): info["name"] = val[:80]
                        elif lo.startswith("uf"):      info["uf"] = val[:4]
                        elif lo.startswith("latitude"):  info["raw_lat"] = val
                        elif lo.startswith("longitude"): info["raw_lon"] = val
                    code = normalize_station_code(info.get("raw_code", ""))
                    if not code:
                        continue
                    lat = parse_decimal_comma_float(info.get("raw_lat", ""))
                    lon = parse_decimal_comma_float(info.get("raw_lon", ""))
                    if lat == 0.0 and lon == 0.0:
                        continue
                    year_m = name.split("_")[0] if name else ""
                    # extract year from filename pattern *_YYYY-*
                    import re
                    ym = re.search(r"(\d{4})", name)
                    year = ym.group(1) if ym else "?"
                    if code not in out:
                        out[code] = {
                            "code": code, "name": info.get("name", ""),
                            "uf": normalize_uf(info.get("uf", "")),
                            "lat": lat, "lon": lon,
                            "raw_lat": info.get("raw_lat", ""),
                            "raw_lon": info.get("raw_lon", ""),
                            "years": {year},
                        }
                    else:
                        out[code]["years"].add(year)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def run() -> dict[str, Any]:
    inmet_dir = raw_root() / "inmet" / "historical"

    # Collect from ZIPs
    zip_stations: dict[str, dict[str, Any]] = {}
    for year in ("2022", "2023", "2024", "2021", "2020", "2025", "2026"):
        zp = inmet_dir / f"inmet_{year}.zip"
        if zp.exists():
            found = _parse_zip_stations(zp)
            for code, info in found.items():
                if code in zip_stations:
                    zip_stations[code]["years"].update(info["years"])
                else:
                    zip_stations[code] = info

    # Load v1si (legacy) and v1sr (corrected)
    v1si_rows = read_csv_safe(DATASETS / "protocol_c_inmet_station_candidates_v1si.csv")
    v1sr_rows = read_csv_safe(DATASETS / "protocol_c_inmet_station_region_proximity_v1sr.csv")

    v1si_by_code: dict[str, dict] = {normalize_station_code(r["station_code"]): r
                                      for r in v1si_rows if r.get("station_code")}
    v1sr_by_code: dict[str, dict] = {normalize_station_code(r["station_code"]): r
                                      for r in v1sr_rows if r.get("station_code")}

    rows: list[dict[str, Any]] = []
    for code, info in zip_stations.items():
        lat, lon = info["lat"], info["lon"]
        anomaly = detect_coordinate_anomaly(lat, lon)
        coord_status = station_coordinate_quality_status(
            lat, lon, info["raw_lat"], info["raw_lon"]
        )

        nearest, dist = nearest_region_and_distance(lat, lon)
        dists = station_region_distances(lat, lon)
        min_dist = min(dists.values()) if dists else float("inf")

        # Build provenance list
        sources = ["raw_zip"]
        if code in v1si_by_code:
            sources.append("v1si")
        if code in v1sr_by_code:
            sources.append("v1sr")

        row: dict[str, Any] = {
            "station_id":   f"V1TA_ST_{code}",
            "station_code": code,
            "station_name": info["name"],
            "uf":           info["uf"],
            "latitude":     f"{lat:.6f}",
            "longitude":    f"{lon:.6f}",
            "raw_lat_text": info["raw_lat"],
            "raw_lon_text": info["raw_lon"],
            "coordinate_quality_status": coord_status,
            "source_years":     ";".join(sorted(info["years"])),
            "provenance_sources": ";".join(sources),
            "nearest_region":   nearest,
            "nearest_region_distance_km": f"{min_dist:.2f}",
            "within_25km":  "true" if min_dist <= 25  else "false",
            "within_50km":  "true" if min_dist <= 50  else "false",
            "within_100km": "true" if min_dist <= 100 else "false",
            "notes":        anomaly if anomaly != "OK" else "",
        }
        row.update(guardrail_row())
        rows.append(row)

    write_csv_with_header(OUT_REG, rows, REG_FIELDS)
    write_schema(SCHEMA_R, REG_FIELDS, "v1ta_registry")

    ok      = sum(1 for r in rows if r["coordinate_quality_status"] != "COORD_ANOMALY_ZERO_COORDS")
    w100    = sum(1 for r in rows if r["within_100km"] == "true")
    by_src  = {s: sum(1 for r in rows if s in r["provenance_sources"]) for s in ("v1si","v1sr","raw_zip")}
    summary = [
        {"stat_key": "canonical_stations_total",   "stat_value": str(len(rows))},
        {"stat_key": "coord_quality_ok",           "stat_value": str(ok)},
        {"stat_key": "within_100km",               "stat_value": str(w100)},
        {"stat_key": "sourced_from_v1si",          "stat_value": str(by_src["v1si"])},
        {"stat_key": "sourced_from_v1sr",          "stat_value": str(by_src["v1sr"])},
        {"stat_key": "sourced_from_raw_zip",       "stat_value": str(by_src["raw_zip"])},
        {"stat_key": "stage",                      "stat_value": "v1ta"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1ta_summary")

    write_doc(DOC, "v1ta — INMET Canonical Station Registry", [
        "## Objetivo",
        "Registry canônico de estações INMET com coordenadas corrigidas "
        "(decimal-vírgula → ponto). Preserva provenance de v1si, v1sr e ZIPs brutos.",
        f"## Resultado\nEstações: {len(rows)}. OK coords: {ok}. "
        f"Dentro de 100km: {w100}.",
        "## Nota metodológica",
        "v1si tinha bug de parse: vírgula decimal interpretada como milhar. "
        "v1ta corrige usando os ZIPs originais. v1si não é modificado.",
    ])
    print(f"[v1ta] stations={len(rows)} coord_ok={ok} within_100km={w100}")
    return {"stations": len(rows), "coord_ok": ok, "within_100km": w100}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ta canonical station registry").parse_args()
    run()
