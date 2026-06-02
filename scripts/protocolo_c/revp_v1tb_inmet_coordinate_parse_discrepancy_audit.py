"""REV-P v1tb — INMET coordinate parse discrepancy audit.

Compares v1si (legacy broken coords) against v1ta canonical registry.
Documents the decimal-comma parse bug as a QA artefact. Does NOT modify v1si.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1ta_v1tf_inmet_canonical_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row,
    parse_decimal_comma_float, normalize_station_code,
    compare_station_records, detect_coordinate_anomaly,
    hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_AUD  = _p("REVP_V1TB_OUT_AUD",  DATASETS / "protocol_c_inmet_coordinate_parse_discrepancy_audit_v1tb.csv")
OUT_SUM  = _p("REVP_V1TB_OUT_SUM",  DATASETS / "protocol_c_inmet_coordinate_parse_discrepancy_summary_v1tb.csv")
SCHEMA_A = _p("REVP_V1TB_SCHEMA_A", SCHEMAS  / "protocol_c_inmet_coordinate_parse_discrepancy_audit_v1tb_schema.csv")
SCHEMA_S = _p("REVP_V1TB_SCHEMA_S", SCHEMAS  / "protocol_c_inmet_coordinate_parse_discrepancy_summary_v1tb_schema.csv")
DOC      = _p("REVP_V1TB_DOC",      DOCS     / "revp_v1tb_inmet_coordinate_parse_discrepancy_audit.md")

AUD_FIELDS = [
    "discrepancy_id", "station_code", "station_name",
    "v1si_lat", "v1si_lon", "canonical_lat", "canonical_lon",
    "delta_km", "discrepancy_type", "affects_region_matching",
    "correction_status",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    v1si = read_csv_safe(DATASETS / "protocol_c_inmet_station_candidates_v1si.csv")
    v1ta = read_csv_safe(DATASETS / "protocol_c_inmet_canonical_station_registry_v1ta.csv")

    canon_by_code: dict[str, dict] = {
        normalize_station_code(r["station_code"]): r for r in v1ta
        if r.get("station_code")
    }

    rows: list[dict[str, Any]] = []
    for r in v1si:
        code = normalize_station_code(r.get("station_code", ""))
        if not code:
            continue
        canon = canon_by_code.get(code)
        if canon is None:
            continue

        v1si_lat = r.get("latitude", "")
        v1si_lon = r.get("longitude", "")
        canon_lat = canon.get("latitude", "")
        canon_lon = canon.get("longitude", "")

        disc = compare_station_records(v1si_lat, v1si_lon, canon_lat, canon_lon)
        dtype = disc["discrepancy_type"]

        # Determine if region matching is affected
        si_lat_f = parse_decimal_comma_float(v1si_lat, 9999.0)
        si_lon_f = parse_decimal_comma_float(v1si_lon, 9999.0)
        si_anomaly = detect_coordinate_anomaly(si_lat_f, si_lon_f)
        affects = "true" if si_anomaly != "OK" or dtype == "DECIMAL_COMMA_CORRECTION" else "false"

        row: dict[str, Any] = {
            "discrepancy_id":        f"V1TB_D{len(rows):04d}",
            "station_code":          code,
            "station_name":          r.get("station_name", canon.get("station_name", "")),
            "v1si_lat":              v1si_lat,
            "v1si_lon":              v1si_lon,
            "canonical_lat":         canon_lat,
            "canonical_lon":         canon_lon,
            "delta_km":              disc.get("delta_km", ""),
            "discrepancy_type":      dtype,
            "affects_region_matching": affects,
            "correction_status":     "CORRECTED_IN_V1TA" if dtype != "NO_DISCREPANCY" else "NO_CORRECTION_NEEDED",
            "notes":                 "",
        }
        row.update(guardrail_row())
        rows.append(row)

    if not rows:
        rows = [{
            "discrepancy_id": "FAIL_CLOSED_NO_V1SI_DATA",
            "station_code": "", "station_name": "",
            "v1si_lat": "", "v1si_lon": "", "canonical_lat": "", "canonical_lon": "",
            "delta_km": "", "discrepancy_type": "NO_V1SI_INPUT",
            "affects_region_matching": "false", "correction_status": "N/A",
            "notes": "v1si not found or empty",
            **guardrail_row(),
        }]

    write_csv_with_header(OUT_AUD, rows, AUD_FIELDS)
    write_schema(SCHEMA_A, AUD_FIELDS, "v1tb_audit")

    corrected = sum(1 for r in rows if r["correction_status"] == "CORRECTED_IN_V1TA")
    affected  = sum(1 for r in rows if r["affects_region_matching"] == "true")
    no_disc   = sum(1 for r in rows if r["discrepancy_type"] == "NO_DISCREPANCY")
    summary = [
        {"stat_key": "stations_compared",          "stat_value": str(len(rows))},
        {"stat_key": "corrected_in_v1ta",          "stat_value": str(corrected)},
        {"stat_key": "affects_region_matching",    "stat_value": str(affected)},
        {"stat_key": "no_discrepancy",             "stat_value": str(no_disc)},
        {"stat_key": "v1si_not_modified",          "stat_value": "true"},
        {"stat_key": "stage",                      "stat_value": "v1tb"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tb_summary")

    write_doc(DOC, "v1tb — INMET Coordinate Parse Discrepancy Audit", [
        "## Objetivo",
        "Documentar discrepância entre coordenadas v1si (parse quebrado) e "
        "v1ta (parse correto com decimal-vírgula). v1si não é modificado.",
        "## Causa do bug em v1si",
        "O extrator v1si interpretou a vírgula decimal dos CSVs INMET como "
        "separador de milhar (ex: -22,75 lido como -2275.0 ou '-22,' + '75'). "
        "v1sr e v1ta corrigem substituindo vírgula por ponto antes do parse.",
        f"## Resultado\nEstações comparadas: {len(rows)}. "
        f"Corrigidas em v1ta: {corrected}. "
        f"Afeta matching de região: {affected}.",
    ])
    print(f"[v1tb] compared={len(rows)} corrected={corrected} region_affected={affected}")
    return {"compared": len(rows), "corrected": corrected, "region_affected": affected}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tb coordinate discrepancy audit").parse_args()
    run()
