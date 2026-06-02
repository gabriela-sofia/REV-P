"""REV-P v1sv — Hydromet evidence to intake crosswalk.

Creates crosswalk entries linking hydrometeorological context data to v1rb
intake format. All entries require manual review; no automatic intake.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_schema, write_doc,
    guardrail_row, scan_guardrails, read_csv_safe,
    hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_CW  = _p("REVP_V1SV_OUT_CW",  DATASETS / "protocol_c_hydromet_evidence_intake_crosswalk_v1sv.csv")
OUT_SUM = _p("REVP_V1SV_OUT_SUM", DATASETS / "protocol_c_hydromet_evidence_intake_crosswalk_summary_v1sv.csv")
SCHEMA_C = _p("REVP_V1SV_SCHEMA_C", SCHEMAS / "protocol_c_hydromet_evidence_intake_crosswalk_v1sv_schema.csv")
SCHEMA_S = _p("REVP_V1SV_SCHEMA_S", SCHEMAS / "protocol_c_hydromet_evidence_intake_crosswalk_summary_v1sv_schema.csv")
DOC      = _p("REVP_V1SV_DOC",      DOCS    / "revp_v1sv_hydromet_evidence_intake_crosswalk.md")

CW_FIELDS = [
    "crosswalk_id", "event_window_id", "event_candidate_id", "region",
    "source_name", "source_family", "evidence_type", "context_summary",
    "intake_status", "manual_review_required", "license_note",
    "review_only", "does_not_validate_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    win_rows  = read_csv_safe(DATASETS / "protocol_c_event_date_windows_v1ss.csv")
    feat_rows = read_csv_safe(DATASETS / "protocol_c_rolling_rainfall_context_features_v1su.csv")

    # Aggregate feature info per event window
    feat_by_win: dict[str, list[dict[str, str]]] = {}
    for r in feat_rows:
        if r.get("feature_status") == "ROLLING_CONTEXT_REVIEW_ONLY":
            feat_by_win.setdefault(r["event_window_id"], []).append(r)

    rows: list[dict[str, Any]] = []
    for w in win_rows:
        if w.get("blocked_reason"):
            continue
        wid = w["event_window_id"]
        feats = feat_by_win.get(wid, [])
        n_st = len(feats)

        if n_st == 0:
            context_summary = "NO_HYDROMET_CONTEXT_AVAILABLE"
        else:
            rain_7d_vals = [float(f["rain_7d"]) for f in feats if f.get("rain_7d")]
            max_r7 = max(rain_7d_vals, default=0.0)
            context_summary = (
                f"n_stations={n_st}; max_rain_7d={max_r7:.1f}mm; "
                f"region={w.get('region','')}; date={w.get('parsed_date','')}"
            )

        row: dict[str, Any] = {
            "crosswalk_id":       f"V1SV_CW{hash_short(wid, 10)}",
            "event_window_id":    wid,
            "event_candidate_id": w.get("event_candidate_id", ""),
            "region":             w.get("region", ""),
            "source_name":        "INMET",
            "source_family":      "OFFICIAL_HYDROMETEOROLOGICAL",
            "evidence_type":      "HYDROMETEOROLOGICAL_CONTEXT",
            "context_summary":    context_summary,
            "intake_status":      "HYDROMET_CONTEXT_READY_FOR_MANUAL_INTAKE",
            "manual_review_required": "true",
            "license_note":       "PUBLIC_OFFICIAL_SOURCE_NEEDS_LICENSE_REVIEW",
            "notes":              "",
        }
        row.update(guardrail_row())
        rows.append(row)

    if not rows:
        rows = [{
            "crosswalk_id": "FAIL_CLOSED_NO_WINDOWS", "event_window_id": "",
            "event_candidate_id": "", "region": "", "source_name": "INMET",
            "source_family": "OFFICIAL_HYDROMETEOROLOGICAL",
            "evidence_type": "HYDROMETEOROLOGICAL_CONTEXT",
            "context_summary": "FAIL_CLOSED_NO_WINDOWS",
            "intake_status": "FAIL_CLOSED", "manual_review_required": "true",
            "license_note": "", "notes": "", **guardrail_row(),
        }]

    violations = scan_guardrails(rows, "v1sv_cw")
    if violations:
        raise ValueError(f"Guardrail violations in v1sv: {violations[:3]}")

    write_csv_with_header(OUT_CW, rows, CW_FIELDS)
    write_schema(SCHEMA_C, CW_FIELDS, "v1sv_crosswalk")

    manual = sum(1 for r in rows if r["manual_review_required"] == "true")
    summary = [
        {"stat_key": "crosswalk_rows",         "stat_value": str(len(rows))},
        {"stat_key": "manual_review_required", "stat_value": str(manual)},
        {"stat_key": "auto_intake_rows",       "stat_value": "0"},
        {"stat_key": "stage",                  "stat_value": "v1sv"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1sv_summary")

    write_doc(DOC, "v1sv — Hydromet Evidence Intake Crosswalk", [
        "## Objetivo",
        "Criar crosswalk entre contexto hidrometeorologico (v1sr-v1su) e "
        "o template de intake v1rb/v1rc. Todos os itens requerem revisao manual.",
        f"## Resultado\nCrosswalk rows: {len(rows)}. Manual review: {manual}.",
        "## Limitacao",
        "Crosswalk nao implica validacao automatica de evento.",
    ])
    print(f"[v1sv] crosswalk={len(rows)} manual={manual}")
    return {"crosswalk": len(rows), "manual": manual}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sv hydromet intake crosswalk").parse_args()
    run()
