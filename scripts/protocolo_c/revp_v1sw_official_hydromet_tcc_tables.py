"""REV-P v1sw — Official hydromet TCC tables.

Generates three TCC-ready tables: station coverage, event window availability,
and a methodological limitations summary. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_schema, write_doc,
    guardrail_row, scan_guardrails, read_csv_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_COV   = _p("REVP_V1SW_OUT_COV",   DATASETS / "protocol_c_tcc_table_hydromet_station_coverage_v1sw.csv")
OUT_WIN   = _p("REVP_V1SW_OUT_WIN",   DATASETS / "protocol_c_tcc_table_hydromet_event_windows_v1sw.csv")
OUT_LIM   = _p("REVP_V1SW_OUT_LIM",   DATASETS / "protocol_c_tcc_table_hydromet_context_limitations_v1sw.csv")
SCHEMA_CO = _p("REVP_V1SW_SCHEMA_CO", SCHEMAS  / "protocol_c_tcc_table_hydromet_station_coverage_v1sw_schema.csv")
SCHEMA_WI = _p("REVP_V1SW_SCHEMA_WI", SCHEMAS  / "protocol_c_tcc_table_hydromet_event_windows_v1sw_schema.csv")
SCHEMA_LI = _p("REVP_V1SW_SCHEMA_LI", SCHEMAS  / "protocol_c_tcc_table_hydromet_context_limitations_v1sw_schema.csv")
DOC       = _p("REVP_V1SW_DOC",       DOCS     / "revp_v1sw_official_hydromet_tcc_tables.md")

COV_FIELDS = ["region", "stations_within_100km", "stations_within_50km",
              "stations_within_25km", "nearest_distance_km", "coverage_status",
              "review_only", "notes"]
WIN_FIELDS = ["region", "event_windows_available", "date_range",
              "hydromet_context_rows", "feature_rows",
              "review_only", "notes"]
LIM_FIELDS = ["limitation_id", "aspect", "description", "implication",
              "review_only", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    prox  = read_csv_safe(DATASETS / "protocol_c_inmet_station_region_proximity_v1sr.csv")
    wins  = read_csv_safe(DATASETS / "protocol_c_event_date_windows_v1ss.csv")
    ctx   = read_csv_safe(DATASETS / "protocol_c_inmet_precipitation_event_window_context_v1st.csv")
    feats = read_csv_safe(DATASETS / "protocol_c_rolling_rainfall_context_features_v1su.csv")

    # Station coverage table
    cov_rows: list[dict[str, Any]] = []
    from collections import defaultdict
    by_reg: dict[str, list[dict]] = defaultdict(list)
    for r in prox:
        if r.get("within_100km") == "true":
            by_reg[r["nearest_region"]].append(r)

    for region in ("RECIFE", "PET", "CURITIBA"):
        stlist = by_reg.get(region, [])
        stlist_sorted = sorted(stlist, key=lambda x: float(x.get("distance_km","9999") or "9999"))
        n100 = len(stlist)
        n50  = sum(1 for s in stlist if s.get("within_50km") == "true")
        n25  = sum(1 for s in stlist if s.get("within_25km") == "true")
        nearest = stlist_sorted[0].get("distance_km", "") if stlist_sorted else ""
        cov_rows.append({
            "region": region,
            "stations_within_100km": str(n100),
            "stations_within_50km":  str(n50),
            "stations_within_25km":  str(n25),
            "nearest_distance_km":   nearest,
            "coverage_status":       "ADEQUATE_CONTEXT" if n100 >= 1 else "NO_NEARBY_STATIONS",
            "review_only":           "true", "notes": "",
        })

    write_csv_with_header(OUT_COV, cov_rows, COV_FIELDS)
    write_schema(SCHEMA_CO, COV_FIELDS, "v1sw_coverage")

    # Event window table
    win_rows_tcc: list[dict[str, Any]] = []
    ctx_by_reg: dict[str, int] = defaultdict(int)
    feat_by_reg: dict[str, int] = defaultdict(int)
    for r in ctx:
        if r.get("precipitation_context_status") == "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY":
            ctx_by_reg[r["region"]] += 1
    for r in feats:
        if r.get("feature_status") == "ROLLING_CONTEXT_REVIEW_ONLY":
            feat_by_reg[r["region"]] += 1

    for region in ("RECIFE", "PET", "CURITIBA"):
        region_wins = [w for w in wins if w.get("region") == region and not w.get("blocked_reason")]
        dates = sorted({w["parsed_date"] for w in region_wins if w.get("parsed_date")})
        win_rows_tcc.append({
            "region":                  region,
            "event_windows_available": str(len(region_wins)),
            "date_range":              f"{dates[0]} – {dates[-1]}" if dates else "N/A",
            "hydromet_context_rows":   str(ctx_by_reg.get(region, 0)),
            "feature_rows":            str(feat_by_reg.get(region, 0)),
            "review_only": "true", "notes": "",
        })

    write_csv_with_header(OUT_WIN, win_rows_tcc, WIN_FIELDS)
    write_schema(SCHEMA_WI, WIN_FIELDS, "v1sw_windows")

    # Methodological limitations table
    lim_rows: list[dict[str, Any]] = [
        {"limitation_id": "LIM01", "aspect": "temporal_coverage",
         "description": "Dados INMET 2020-2026; precipitacao horaria agregada a diaria.",
         "implication": "Eventos sub-horarios nao captados.", "review_only": "true", "notes": ""},
        {"limitation_id": "LIM02", "aspect": "spatial_representativity",
         "description": "Precipitacao pontual nao representa area de bacia.",
         "implication": "Variabilidade espacial nao modelada.", "review_only": "true", "notes": ""},
        {"limitation_id": "LIM03", "aspect": "event_validation",
         "description": "Precipitacao compativel nao valida evento de deslizamento ou inundacao.",
         "implication": "Revisao humana obrigatoria para qualquer interpretacao causal.", "review_only": "true", "notes": ""},
        {"limitation_id": "LIM04", "aspect": "negative_evidence",
         "description": "Ausencia de precipitacao nao constitui evidencia negativa.",
         "implication": "Evento pode ter ocorrido por acumulados anteriores ou fontes locais.", "review_only": "true", "notes": ""},
        {"limitation_id": "LIM05", "aspect": "station_density",
         "description": "Densidade de estacoes varia por regiao.",
         "implication": "Areas sem estacao proxima tem lacuna de contexto, nao evidencia negativa.", "review_only": "true", "notes": ""},
        {"limitation_id": "LIM06", "aspect": "license",
         "description": "Dados INMET: fonte publica oficial, necessita revisao de licenca antes de publicacao.",
         "implication": "Verificar termos no portal INMET antes de citar em publicacao.", "review_only": "true", "notes": ""},
    ]
    write_csv_with_header(OUT_LIM, lim_rows, LIM_FIELDS)
    write_schema(SCHEMA_LI, LIM_FIELDS, "v1sw_limitations")

    write_doc(DOC, "v1sw — Official Hydromet TCC Tables", [
        "## Objetivo",
        "Tabelas prontas para TCC: cobertura de estacoes, disponibilidade de "
        "janelas de eventos e limitacoes metodologicas.",
        f"## Resultado\nCobertura: {len(cov_rows)} regioes. "
        f"Janelas: {sum(int(w['event_windows_available']) for w in win_rows_tcc)}. "
        f"Limitacoes documentadas: {len(lim_rows)}.",
    ])
    print(f"[v1sw] coverage_regions={len(cov_rows)} limitations={len(lim_rows)}")
    return {"coverage_regions": len(cov_rows), "limitations": len(lim_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sw TCC tables").parse_args()
    run()
