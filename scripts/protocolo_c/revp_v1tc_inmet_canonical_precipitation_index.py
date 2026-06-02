"""REV-P v1tc — INMET canonical precipitation index.

Reads precipitation from local INMET ZIPs for stations near target regions.
Uses canonical station registry (v1ta) for correct coordinates. Aggregates
hourly → daily. Capped via env vars. Review-only; never creates events.
"""
from __future__ import annotations
import argparse, csv, io, os, zipfile
from pathlib import Path
from typing import Any

from revp_v1ta_v1tf_inmet_canonical_common import (
    DATASETS, DOCS, SCHEMAS, _p, raw_root,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row, normalize_date, parse_decimal_comma_float, hash_short,
    normalize_station_code,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_IDX  = _p("REVP_V1TC_OUT_IDX",  DATASETS / "protocol_c_inmet_canonical_precipitation_index_v1tc.csv")
OUT_SUM  = _p("REVP_V1TC_OUT_SUM",  DATASETS / "protocol_c_inmet_canonical_precipitation_index_summary_v1tc.csv")
SCHEMA_I = _p("REVP_V1TC_SCHEMA_I", SCHEMAS  / "protocol_c_inmet_canonical_precipitation_index_v1tc_schema.csv")
SCHEMA_S = _p("REVP_V1TC_SCHEMA_S", SCHEMAS  / "protocol_c_inmet_canonical_precipitation_index_summary_v1tc_schema.csv")
DOC      = _p("REVP_V1TC_DOC",      DOCS     / "revp_v1tc_inmet_canonical_precipitation_index.md")

IDX_FIELDS = [
    "precip_record_id", "station_code", "station_name", "uf",
    "nearest_region", "date", "precipitation_mm",
    "temporal_precision", "spatial_precision",
    "source_year", "provenance_status",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative",
    "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_PRECIP_COL_IDX = 2   # 0-indexed after splitting by ";"

def _max_daily_rows() -> int:
    return int(os.environ.get("REVP_INMETmax_rows", "250000"))


def _parse_daily_precip_from_file(text: str) -> dict[str, float]:
    """Parse hourly CSV text → daily totals {ISO_date: mm}. Comma-decimal aware."""
    lines = text.splitlines()
    hdr_idx = -1
    for i, ln in enumerate(lines):
        if "DATA" in ln.upper() and "HORA" in ln.upper():
            hdr_idx = i
            break
    if hdr_idx < 0:
        return {}

    daily: dict[str, float] = {}
    try:
        reader = csv.reader(io.StringIO("\n".join(lines[hdr_idx:])), delimiter=";")
        next(reader, None)  # skip header row
        for row in reader:
            if not row:
                continue
            date_val = row[0].strip() if row else ""
            d_iso = normalize_date(date_val)
            if not d_iso:
                continue
            # Precip is 3rd field (index 2)
            raw_p = row[_PRECIP_COL_IDX].strip() if len(row) > _PRECIP_COL_IDX else ""
            if not raw_p:
                continue
            mm = parse_decimal_comma_float(raw_p, -1.0)
            if mm < 0:
                continue
            daily[d_iso] = daily.get(d_iso, 0.0) + mm
    except Exception:
        pass
    return daily


def run() -> dict[str, Any]:
    # Load canonical stations; filter to within 100km
    station_rows = read_csv_safe(DATASETS / "protocol_c_inmet_canonical_station_registry_v1ta.csv")
    nearby: dict[str, dict[str, str]] = {
        normalize_station_code(r["station_code"]): r
        for r in station_rows
        if r.get("within_100km") == "true"
    }

    inmet_dir = raw_root() / "inmet" / "historical"
    rows: list[dict[str, Any]] = []
    years_processed: set[str] = set()
    stations_hit: set[str] = set()
    max_rows = _max_daily_rows()

    for year in ("2022", "2021", "2020", "2023", "2024", "2025", "2026"):
        if len(rows) >= max_rows:
            break
        zp = inmet_dir / f"inmet_{year}.zip"
        if not zp.exists():
            continue
        years_processed.add(year)

        try:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if len(rows) >= max_rows:
                        break
                    if not name.upper().endswith(".CSV"):
                        continue
                    # Extract code from filename  e.g. INMET_SE_RJ_A601_...
                    parts = name.split("_")
                    code = None
                    for p in parts:
                        if p.startswith("A") and p[1:].isdigit():
                            code = p
                            break
                    if code not in nearby:
                        continue
                    try:
                        text = z.read(name).decode("latin-1", errors="replace")
                        daily = _parse_daily_precip_from_file(text)
                        st = nearby[code]
                        for d_iso, mm in sorted(daily.items()):
                            if len(rows) >= max_rows:
                                break
                            row: dict[str, Any] = {
                                "precip_record_id": f"V1TC_{hash_short(code+d_iso, 10)}",
                                "station_code":  code,
                                "station_name":  st.get("station_name", ""),
                                "uf":            st.get("uf", ""),
                                "nearest_region": st.get("nearest_region", ""),
                                "date":          d_iso,
                                "precipitation_mm": f"{mm:.2f}",
                                "temporal_precision": "DAY",
                                "spatial_precision":  "POINT",
                                "source_year":   year,
                                "provenance_status": "OFFICIAL_INMET_CANONICAL_REVIEW_ONLY",
                                "notes":         "",
                            }
                            row.update(guardrail_row())
                            rows.append(row)
                        stations_hit.add(code)
                    except Exception:
                        continue
        except Exception:
            continue

    if not rows:
        rows = [{
            "precip_record_id": "FAIL_CLOSED_NO_DATA",
            "station_code": "", "station_name": "", "uf": "",
            "nearest_region": "", "date": "", "precipitation_mm": "",
            "temporal_precision": "N/A", "spatial_precision": "N/A",
            "source_year": "", "provenance_status": "FAIL_CLOSED_NO_CANONICAL_STATIONS",
            "notes": "", **guardrail_row(),
        }]

    write_csv_with_header(OUT_IDX, rows, IDX_FIELDS)
    write_schema(SCHEMA_I, IDX_FIELDS, "v1tc_precip_index")

    real_rows = [r for r in rows if r["provenance_status"] == "OFFICIAL_INMET_CANONICAL_REVIEW_ONLY"]
    summary = [
        {"stat_key": "total_daily_records",   "stat_value": str(len(real_rows))},
        {"stat_key": "stations_with_data",    "stat_value": str(len(stations_hit))},
        {"stat_key": "years_processed",       "stat_value": ";".join(sorted(years_processed))},
        {"stat_key": "max_daily_rows_cap",    "stat_value": str(max_rows)},
        {"stat_key": "validates_events",      "stat_value": "false"},
        {"stat_key": "creates_negatives",     "stat_value": "false"},
        {"stat_key": "stage",                 "stat_value": "v1tc"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tc_summary")

    write_doc(DOC, "v1tc — INMET Canonical Precipitation Index", [
        "## Objetivo",
        "Índice canônico de precipitação diária INMET para estações dentro de 100km "
        "das regiões-alvo. Parse correto de decimal-vírgula. Review-only.",
        f"## Resultado\nRegistros diários: {len(real_rows)}. "
        f"Estações: {len(stations_hit)}. Anos: {sorted(years_processed)}.",
        "## Limitação",
        "Precipitação não valida eventos. Ausência de dados não é evidência negativa.",
    ])
    print(f"[v1tc] daily_records={len(real_rows)} stations={len(stations_hit)} "
          f"years={sorted(years_processed)}")
    return {"records": len(real_rows), "stations": len(stations_hit),
            "years": sorted(years_processed)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tc canonical precipitation index").parse_args()
    run()
