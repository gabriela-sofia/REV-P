"""REV-P v1si — INMET station and precipitation extractor.

Reads downloaded INMET ZIPs/CSVs, extracts station metadata and daily
precipitation, filters by region proximity. Never creates labels. Review-only.
"""
from __future__ import annotations
import argparse, csv, io, re, zipfile
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, forbidden_guardrail_scan,
    raw_root, read_csv_safe, safe_relpath, hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_STATIONS = _p("REVP_V1SI_OUT_STATIONS", DATASETS / "protocol_c_inmet_station_candidates_v1si.csv")
OUT_PRECIP = _p("REVP_V1SI_OUT_PRECIP", DATASETS / "protocol_c_inmet_precipitation_daily_review_v1si.csv")
OUT_SUMMARY = _p("REVP_V1SI_OUT_SUMMARY", DATASETS / "protocol_c_inmet_extraction_summary_v1si.csv")
SCHEMA_ST = _p("REVP_V1SI_SCHEMA_ST", SCHEMAS / "protocol_c_inmet_station_candidates_v1si_schema.csv")
SCHEMA_PR = _p("REVP_V1SI_SCHEMA_PR", SCHEMAS / "protocol_c_inmet_precipitation_daily_review_v1si_schema.csv")
SCHEMA_SM = _p("REVP_V1SI_SCHEMA_SM", SCHEMAS / "protocol_c_inmet_extraction_summary_v1si_schema.csv")
DOC = _p("REVP_V1SI_DOC", DOCS / "revp_v1si_inmet_station_precipitation_extractor.md")

STATION_FIELDS = ["station_id", "station_code", "station_name", "uf", "latitude",
                  "longitude", "region_candidate", "source_file", "review_only", "notes"]
PRECIP_FIELDS = [
    "record_id", "source_name", "station_code", "station_name", "uf",
    "region_candidate", "date", "precipitation_mm", "temporal_precision",
    "spatial_precision", "provenance_status", "review_only",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

# Region bounding boxes (approximate)
_REGIONS = {
    "RECIFE": (-8.2, -8.0, -35.1, -34.8),
    "PET": (-22.6, -22.4, -43.3, -43.1),
    "CURITIBA": (-25.6, -25.3, -49.4, -49.2),
}

def _region_from_coords(lat: float, lon: float) -> str:
    for name, (lat_min, lat_max, lon_min, lon_max) in _REGIONS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    # Looser match by UF
    return "UNKNOWN"

def _region_from_uf(uf: str) -> str:
    u = str(uf).strip().upper()
    if u == "PE": return "RECIFE"
    if u == "RJ": return "PET"
    if u == "PR": return "CURITIBA"
    return "UNKNOWN"

def _extract_from_csv_text(text: str, source_file: str,
                           stations: dict, precip: list, max_precip: int = 5000) -> None:
    """Extract stations and precipitation from INMET CSV content."""
    lines = text.splitlines()
    # INMET CSVs have metadata lines before the header; look for patterns
    station_code = station_name = uf = lat_s = lon_s = ""
    for line in lines[:20]:
        lo = line.lower().replace(";", ",")
        if "codigo" in lo and "estacao" not in lo:
            station_code = lo.split(",")[-1].strip().strip('"')
        elif "estacao" in lo or "station" in lo:
            station_name = line.split(",")[-1].strip().strip('"')[:60]
        elif lo.startswith("uf"):
            uf = line.split(",")[-1].strip().strip('"')[:4]
        elif "latitude" in lo:
            lat_s = line.split(",")[-1].strip().strip('"').replace(",", ".")
        elif "longitude" in lo:
            lon_s = line.split(",")[-1].strip().strip('"').replace(",", ".")

    lat = lon = 0.0
    try: lat = float(lat_s)
    except: pass
    try: lon = float(lon_s)
    except: pass

    region = _region_from_coords(lat, lon) if (lat and lon) else _region_from_uf(uf)
    sid = station_code or hash_short(station_name + uf, 8)

    if sid not in stations:
        stations[sid] = {
            "station_id": f"V1SI_ST_{sid}", "station_code": station_code,
            "station_name": station_name, "uf": uf,
            "latitude": f"{lat:.6f}" if lat else "", "longitude": f"{lon:.6f}" if lon else "",
            "region_candidate": region, "source_file": safe_relpath(Path(source_file)),
            "review_only": "true", "notes": "",
        }

    # Find data header
    header_idx = -1
    for i, line in enumerate(lines):
        if "DATA" in line.upper() and ("PRECIPITACAO" in line.upper() or "CHUVA" in line.upper()):
            header_idx = i
            break
    if header_idx < 0:
        # try to find any CSV-like header with date
        for i, line in enumerate(lines):
            if re.match(r".*\d{4}[/-]\d{2}[/-]\d{2}.*", line):
                header_idx = max(0, i - 1)
                break
    if header_idx < 0:
        return

    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])), delimiter=";")
        for j, row in enumerate(reader):
            if len(precip) >= max_precip:
                break
            date_val = ""
            precip_val = ""
            for k, v in row.items():
                kl = (k or "").upper()
                if "DATA" in kl and not date_val:
                    date_val = str(v or "").strip()[:10]
                if ("PRECIPITACAO" in kl or "CHUVA" in kl or "PRECIP" in kl) and not precip_val:
                    precip_val = str(v or "").strip().replace(",", ".")
            if date_val and precip_val:
                pr = {
                    "record_id": f"V1SI_PR_{hash_short(sid+date_val, 10)}",
                    "source_name": "INMET", "station_code": station_code,
                    "station_name": station_name, "uf": uf,
                    "region_candidate": region, "date": date_val,
                    "precipitation_mm": precip_val,
                    "temporal_precision": "DAY", "spatial_precision": "POINT",
                    "provenance_status": "OFFICIAL_INMET_REVIEW_ONLY",
                    "notes": "",
                }
                pr.update(guardrail_row())
                precip.append(pr)
    except Exception:
        pass


def run(datasets: Path | None = None) -> dict[str, Any]:
    inmet_dir = raw_root() / "inmet" / "historical"
    stations: dict[str, dict] = {}
    precip: list[dict[str, Any]] = []

    if inmet_dir.exists():
        for zf in sorted(inmet_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(zf) as z:
                    for name in z.namelist():
                        if name.lower().endswith(".csv"):
                            text = z.read(name).decode("latin-1", errors="replace")
                            _extract_from_csv_text(text, str(zf), stations, precip)
            except Exception:
                pass
        for cf in sorted(inmet_dir.glob("*.csv")):
            try:
                text = cf.read_text(encoding="latin-1", errors="replace")
                _extract_from_csv_text(text, str(cf), stations, precip)
            except Exception:
                pass

    st_rows = list(stations.values())
    forbidden_guardrail_scan(precip, "v1si_precip")

    write_csv_with_header(OUT_STATIONS, st_rows, STATION_FIELDS)
    write_csv_with_header(OUT_PRECIP, precip, PRECIP_FIELDS)
    write_schema_for(SCHEMA_ST, STATION_FIELDS, "v1si_stations")
    write_schema_for(SCHEMA_PR, PRECIP_FIELDS, "v1si_precip")

    summary = [
        {"stat_key": "stations_found", "stat_value": str(len(st_rows))},
        {"stat_key": "precipitation_records", "stat_value": str(len(precip))},
        {"stat_key": "regions_detected", "stat_value": ";".join(sorted({s["region_candidate"] for s in st_rows}))},
        {"stat_key": "stage", "stat_value": "v1si"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_SM, SUM_FIELDS, "v1si_summary")

    write_doc(DOC, "v1si — INMET Station and Precipitation Extractor", [
        "## Objetivo",
        "Extrair metadata de estacoes e precipitacao diaria dos ZIPs/CSVs INMET baixados. "
        "Filtrar por regiao. Review-only; nunca label.",
        "## Resultado",
        f"Estacoes: {len(st_rows)}. Registros de precipitacao: {len(precip)}.",
    ])
    print(f"[v1si] stations={len(st_rows)} precip_records={len(precip)}")
    return {"stations": len(st_rows), "precip": len(precip)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1si INMET extractor").parse_args()
    run()
