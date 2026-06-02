"""REV-P v1ss — Protocol C event date window builder.

Reads event candidates and official documented events to build temporal
windows for hydrometeorological context lookup. Fail-closed when no
parseable dates exist. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_schema, write_doc,
    guardrail_row, scan_guardrails, read_csv_safe,
    parse_date_safe, normalize_date, normalize_region, build_window,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_WIN  = _p("REVP_V1SS_OUT_WIN",  DATASETS / "protocol_c_event_date_windows_v1ss.csv")
OUT_SUM  = _p("REVP_V1SS_OUT_SUM",  DATASETS / "protocol_c_event_date_windows_summary_v1ss.csv")
SCHEMA_W = _p("REVP_V1SS_SCHEMA_W", SCHEMAS  / "protocol_c_event_date_windows_v1ss_schema.csv")
SCHEMA_S = _p("REVP_V1SS_SCHEMA_S", SCHEMAS  / "protocol_c_event_date_windows_summary_v1ss_schema.csv")
DOC      = _p("REVP_V1SS_DOC",      DOCS     / "revp_v1ss_protocol_c_event_date_window_builder.md")

WIN_FIELDS = [
    "event_window_id", "event_candidate_id", "region", "hazard_type",
    "event_date_text", "parsed_date", "window_start", "window_end",
    "window_days", "temporal_precision_status", "source_block",
    "review_only", "does_not_validate_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

# Input datasets (in priority order for date extraction)
_INPUTS = [
    ("official_documented_event_unit_registry.csv", "event_date", "documented_event_unit_id", "region", "FLOOD_LANDSLIDE", "v1ir"),
    ("ground_reference_event_registry.csv",         "event_or_survey_date", "event_id",                "region", "FLOOD_LANDSLIDE", "v1qx"),
    ("protocol_c_ground_reference_candidate_master_registry.csv", "event_date_text", "candidate_id", "region", "FLOOD_LANDSLIDE", "v1qu"),
]


def _load_events() -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for fname, date_field, id_field, region_field, hazard, source in _INPUTS:
        path = DATASETS / fname
        rows = read_csv_safe(path)
        for r in rows:
            raw_date = r.get(date_field, "")
            parsed = normalize_date(raw_date)
            events.append({
                "candidate_id": r.get(id_field, ""),
                "region":       normalize_region(r.get(region_field, "")),
                "hazard_type":  hazard,
                "date_text":    raw_date,
                "parsed_date":  parsed,
                "source_block": source,
            })
    return events


def run() -> dict[str, Any]:
    events = _load_events()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    no_date = 0

    for i, ev in enumerate(events):
        parsed = ev["parsed_date"]
        if not parsed:
            no_date += 1
            continue

        # Deduplicate by region + parsed_date
        key = f"{ev['region']}_{parsed}"
        if key in seen:
            continue
        seen.add(key)

        d = parse_date_safe(parsed)
        if d is None:
            continue
        w_start, w_end = build_window(d, before_days=7, after_days=1)

        row: dict[str, Any] = {
            "event_window_id":   f"V1SS_W{len(rows):04d}",
            "event_candidate_id": ev["candidate_id"],
            "region":            ev["region"],
            "hazard_type":       ev["hazard_type"],
            "event_date_text":   ev["date_text"],
            "parsed_date":       parsed,
            "window_start":      w_start.isoformat(),
            "window_end":        w_end.isoformat(),
            "window_days":       str((w_end - w_start).days + 1),
            "temporal_precision_status": "DATE_PARSED_OK",
            "source_block":      ev["source_block"],
            "blocked_reason":    "",
            "notes":             "",
        }
        row.update(guardrail_row())
        rows.append(row)

    # Fail-closed: emit header even when empty
    if not rows:
        rows = [{
            "event_window_id": "FAIL_CLOSED_NO_PARSEABLE_DATES",
            "event_candidate_id": "", "region": "", "hazard_type": "",
            "event_date_text": "", "parsed_date": "",
            "window_start": "", "window_end": "", "window_days": "0",
            "temporal_precision_status": "FAIL_CLOSED_NO_INPUT",
            "source_block": "", "blocked_reason": "NO_PARSEABLE_DATES", "notes": "",
            **guardrail_row(),
        }]

    violations = scan_guardrails(rows, "v1ss_windows")
    if violations:
        raise ValueError(f"Guardrail violations in v1ss: {violations[:3]}")

    write_csv_with_header(OUT_WIN, rows, WIN_FIELDS)
    write_schema(SCHEMA_W, WIN_FIELDS, "v1ss_windows")

    parsed_count = sum(1 for r in rows if r["temporal_precision_status"] == "DATE_PARSED_OK")
    regions_covered = set(r["region"] for r in rows if r["region"])
    summary = [
        {"stat_key": "windows_total",     "stat_value": str(len(rows))},
        {"stat_key": "parseable_dates",   "stat_value": str(parsed_count)},
        {"stat_key": "no_date_skipped",   "stat_value": str(no_date)},
        {"stat_key": "regions_covered",   "stat_value": ";".join(sorted(regions_covered))},
        {"stat_key": "stage",             "stat_value": "v1ss"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1ss_summary")

    write_doc(DOC, "v1ss — Protocol C Event Date Window Builder", [
        "## Objetivo",
        "Construir janelas temporais (T-7 a T+1) para cada evento candidato "
        "com data parseavel. Fail-closed quando sem datas.",
        "## Resultado",
        f"Janelas: {len(rows)}. Parseadas: {parsed_count}. "
        f"Regioes: {', '.join(sorted(regions_covered))}.",
        "## Limitacao",
        "Janela temporal compativel nao valida evento automaticamente.",
    ])
    print(f"[v1ss] windows={len(rows)} parsed={parsed_count} regions={sorted(regions_covered)}")
    return {"windows": len(rows), "parsed": parsed_count, "regions": sorted(regions_covered)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ss event date window builder").parse_args()
    run()
