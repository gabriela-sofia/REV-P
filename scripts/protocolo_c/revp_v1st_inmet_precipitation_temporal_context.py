"""REV-P v1st — INMET precipitation temporal context table.

Reads 2022 INMET precipitation data for stations within 100km of target
regions, filtered to event date windows from v1ss. All outputs review-only.
Precipitação never validates events; absence is not negative evidence.
"""
from __future__ import annotations
import argparse, csv, io, zipfile
from pathlib import Path
from typing import Any
from datetime import date

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p, raw_root,
    write_csv_with_header, write_schema, write_doc,
    guardrail_row, scan_guardrails, read_csv_safe,
    parse_float_safe, parse_date_safe, normalize_date, window_position,
    hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_CTX  = _p("REVP_V1ST_OUT_CTX",  DATASETS / "protocol_c_inmet_precipitation_event_window_context_v1st.csv")
OUT_SUM  = _p("REVP_V1ST_OUT_SUM",  DATASETS / "protocol_c_inmet_precipitation_event_window_summary_v1st.csv")
SCHEMA_C = _p("REVP_V1ST_SCHEMA_C", SCHEMAS  / "protocol_c_inmet_precipitation_event_window_context_v1st_schema.csv")
SCHEMA_S = _p("REVP_V1ST_SCHEMA_S", SCHEMAS  / "protocol_c_inmet_precipitation_event_window_summary_v1st_schema.csv")
DOC      = _p("REVP_V1ST_DOC",      DOCS     / "revp_v1st_inmet_precipitation_temporal_context.md")

CTX_FIELDS = [
    "context_id", "event_window_id", "event_candidate_id", "region",
    "station_code", "station_name", "distance_km",
    "date", "precipitation_mm", "window_position",
    "precipitation_context_status", "evidence_role",
    "review_only", "does_not_validate_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_MAX_ROWS = 5000
_MAX_STATIONS_PER_REGION = 10
_HEADER_BYTES = 400


def _parse_precip_from_zip(zip_path: Path, station_codes: set[str],
                            date_set: set[str], max_rows: int = 5000
                            ) -> dict[str, dict[str, float]]:
    """Extract {station_code: {ISO_date: daily_precip_mm}} from a ZIP.

    Only reads files for stations in station_codes. Aggregates hourly → daily.
    Caps at max_rows total records for speed.
    """
    result: dict[str, dict[str, float]] = {}
    total = 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.upper().endswith(".CSV"):
                    continue
                # Quick code check from filename (e.g. INMET_SE_RJ_A627_PETROPOLIS_...)
                upper = name.upper()
                code = None
                for c in station_codes:
                    if f"_{c}_" in upper or upper.startswith(c):
                        code = c
                        break
                if code is None:
                    continue
                try:
                    data = z.read(name)
                    text = data.decode("latin-1", errors="replace")
                    lines = text.splitlines()
                    # Find data header
                    hdr_idx = -1
                    for i, ln in enumerate(lines):
                        if "DATA" in ln.upper() and ("PRECIPIT" in ln.upper() or "HORA" in ln.upper()):
                            hdr_idx = i
                            break
                    if hdr_idx < 0:
                        continue
                    reader = csv.DictReader(io.StringIO("\n".join(lines[hdr_idx:])), delimiter=";")
                    daily: dict[str, float] = result.setdefault(code, {})
                    for row in reader:
                        if total >= max_rows:
                            break
                        # Find date and precip columns
                        date_val = precip_val = ""
                        for k, v in row.items():
                            ku = (k or "").upper()
                            if "DATA" in ku and not date_val:
                                date_val = str(v or "").strip()[:10]
                            if ("PRECIPIT" in ku or "CHUVA" in ku) and not precip_val:
                                precip_val = str(v or "").strip().replace(",", ".")
                        d_iso = normalize_date(date_val)
                        if d_iso and d_iso in date_set:
                            mm = parse_float_safe(precip_val, 0.0)
                            if mm < 0:
                                mm = 0.0
                            daily[d_iso] = daily.get(d_iso, 0.0) + mm
                            total += 1
                except Exception:
                    continue
                if total >= max_rows:
                    break
    except Exception:
        pass
    return result


def run() -> dict[str, Any]:
    prox_rows = read_csv_safe(DATASETS / "protocol_c_inmet_station_region_proximity_v1sr.csv")
    win_rows  = read_csv_safe(DATASETS / "protocol_c_event_date_windows_v1ss.csv")

    # Build window date sets per region
    windows_by_region: dict[str, list[dict[str, str]]] = {}
    all_dates: set[str] = set()
    for w in win_rows:
        if w.get("blocked_reason"):
            continue
        region = w.get("region", "")
        ws = parse_date_safe(w.get("window_start", ""))
        we = parse_date_safe(w.get("window_end", ""))
        if not (ws and we):
            continue
        windows_by_region.setdefault(region, []).append(w)
        d = ws
        from datetime import timedelta
        while d <= we:
            all_dates.add(d.isoformat())
            d += timedelta(days=1)

    # Get nearby stations per region (within 100km, top N closest)
    from collections import defaultdict
    stations_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in prox_rows:
        if r.get("within_100km") == "true":
            stations_by_region[r["nearest_region"]].append(r)
    # Sort by distance, cap per region
    station_codes_needed: set[str] = set()
    for reg, stlist in stations_by_region.items():
        stlist.sort(key=lambda x: float(x.get("distance_km", "9999") or "9999"))
        for st in stlist[:_MAX_STATIONS_PER_REGION]:
            station_codes_needed.add(st["station_code"])

    # Read precipitation from available ZIPs
    inmet_dir = raw_root() / "inmet" / "historical"
    precip_by_station: dict[str, dict[str, float]] = {}
    for year in ("2022", "2021", "2020", "2023", "2024"):
        zp = inmet_dir / f"inmet_{year}.zip"
        if not zp.exists() or not station_codes_needed:
            continue
        found = _parse_precip_from_zip(zp, station_codes_needed, all_dates)
        for code, daily in found.items():
            precip_by_station.setdefault(code, {}).update(daily)
        # Stop if we have data for all needed stations
        if set(precip_by_station.keys()) >= station_codes_needed:
            break

    # Build station lookup
    station_meta: dict[str, dict[str, str]] = {r["station_code"]: r for r in prox_rows}

    rows: list[dict[str, Any]] = []
    for region, win_list in windows_by_region.items():
        nearby = [s for s in stations_by_region.get(region, [])
                  if s["station_code"] in station_codes_needed][:_MAX_STATIONS_PER_REGION]
        for w in win_list:
            ws = parse_date_safe(w.get("window_start", ""))
            we = parse_date_safe(w.get("window_end", ""))
            ev_d = parse_date_safe(w.get("parsed_date", ""))
            if not (ws and we and ev_d):
                continue
            for st in nearby:
                code = st["station_code"]
                dist_km = st.get("distance_km", "")
                daily = precip_by_station.get(code, {})
                from datetime import timedelta
                d = ws
                while d <= we:
                    if len(rows) >= _MAX_ROWS:
                        break
                    d_iso = d.isoformat()
                    mm = daily.get(d_iso)
                    # Include row even with no data (mm=None → not available)
                    status = "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY" if mm is not None else "NO_DATA_AVAILABLE"
                    row: dict[str, Any] = {
                        "context_id":        f"V1ST_C{hash_short(code+d_iso+w['event_window_id'],8)}",
                        "event_window_id":   w["event_window_id"],
                        "event_candidate_id": w.get("event_candidate_id", ""),
                        "region":            region,
                        "station_code":      code,
                        "station_name":      station_meta.get(code, {}).get("station_name", ""),
                        "distance_km":       dist_km,
                        "date":              d_iso,
                        "precipitation_mm":  f"{mm:.2f}" if mm is not None else "",
                        "window_position":   window_position(ev_d, d),
                        "precipitation_context_status": status,
                        "evidence_role":     "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
                        "notes":             "",
                    }
                    row.update(guardrail_row())
                    rows.append(row)
                    d += timedelta(days=1)
                if len(rows) >= _MAX_ROWS:
                    break
            if len(rows) >= _MAX_ROWS:
                break

    if not rows:
        rows = [{
            "context_id": "FAIL_CLOSED_NO_CONTEXT", "event_window_id": "",
            "event_candidate_id": "", "region": "", "station_code": "",
            "station_name": "", "distance_km": "", "date": "",
            "precipitation_mm": "", "window_position": "",
            "precipitation_context_status": "FAIL_CLOSED_NO_WINDOWS_OR_STATIONS",
            "evidence_role": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
            "notes": "", **guardrail_row(),
        }]

    violations = scan_guardrails(rows, "v1st_ctx")
    if violations:
        raise ValueError(f"Guardrail violations in v1st: {violations[:3]}")

    write_csv_with_header(OUT_CTX, rows, CTX_FIELDS)
    write_schema(SCHEMA_C, CTX_FIELDS, "v1st_context")

    with_data = sum(1 for r in rows if r["precipitation_mm"])
    summary = [
        {"stat_key": "context_rows",           "stat_value": str(len(rows))},
        {"stat_key": "rows_with_precip_data",  "stat_value": str(with_data)},
        {"stat_key": "stations_queried",       "stat_value": str(len(station_codes_needed))},
        {"stat_key": "event_windows_used",     "stat_value": str(sum(len(v) for v in windows_by_region.values()))},
        {"stat_key": "validates_events",       "stat_value": "false"},
        {"stat_key": "creates_negative",       "stat_value": "false"},
        {"stat_key": "stage",                  "stat_value": "v1st"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1st_summary")

    write_doc(DOC, "v1st — INMET Precipitation Temporal Context Table", [
        "## Objetivo",
        "Tabela de contexto de precipitacao INMET para janelas de eventos. "
        "Dados filtrados para estacoes dentro de 100km das regioes-alvo.",
        "## Resultado",
        f"Linhas de contexto: {len(rows)}. Com dados de precip: {with_data}.",
        "## Declaracao obrigatoria",
        "Precipitacao observada, proximidade de estacao ou janela temporal "
        "compativel nao validam automaticamente evento, nao criam ground truth "
        "operacional, nao criam negativo formal e nao substituem revisao supervisora.",
    ])
    print(f"[v1st] rows={len(rows)} with_data={with_data} stations_queried={len(station_codes_needed)}")
    return {"rows": len(rows), "with_data": with_data}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1st precipitation context").parse_args()
    run()
