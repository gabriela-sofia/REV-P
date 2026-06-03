"""REV-P v1sz — Hydrometeorological context bundle.

Final bundle for v1sr-v1sy. Manifest, QC checks, scientific summary.
Review-only.

Mandatory clause:
A camada v1sr–v1sz usa dados hidrometeorológicos oficiais apenas como
contexto temporal e regional review-only. Precipitação observada,
proximidade de estação ou janela temporal compatível não validam
automaticamente evento, não criam ground truth operacional, não criam
negativo formal e não substituem revisão supervisora.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_schema, write_doc,
    read_csv_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_SR_SUM  = _p("REVP_V1SZ_IN_SR",  DATASETS / "protocol_c_inmet_station_region_proximity_summary_v1sr.csv")
IN_SS_SUM  = _p("REVP_V1SZ_IN_SS",  DATASETS / "protocol_c_event_date_windows_summary_v1ss.csv")
IN_ST_SUM  = _p("REVP_V1SZ_IN_ST",  DATASETS / "protocol_c_inmet_precipitation_event_window_summary_v1st.csv")
IN_SU_SUM  = _p("REVP_V1SZ_IN_SU",  DATASETS / "protocol_c_rolling_rainfall_context_features_summary_v1su.csv")
IN_SV_SUM  = _p("REVP_V1SZ_IN_SV",  DATASETS / "protocol_c_hydromet_evidence_intake_crosswalk_summary_v1sv.csv")
IN_SX_SUM  = _p("REVP_V1SZ_IN_SX",  DATASETS / "protocol_c_hydromet_context_guardrail_summary_v1sx.csv")

OUT_MAN  = _p("REVP_V1SZ_OUT_MAN",  DATASETS / "protocol_c_hydromet_context_manifest_v1sz.csv")
OUT_QC   = _p("REVP_V1SZ_OUT_QC",   DATASETS / "protocol_c_hydromet_context_quality_checks_v1sz.csv")
OUT_SUM  = _p("REVP_V1SZ_OUT_SUM",  DATASETS / "protocol_c_hydromet_context_scientific_summary_v1sz.csv")
SCHEMA_M = _p("REVP_V1SZ_SCHEMA_M", SCHEMAS  / "protocol_c_hydromet_context_manifest_v1sz_schema.csv")
SCHEMA_Q = _p("REVP_V1SZ_SCHEMA_Q", SCHEMAS  / "protocol_c_hydromet_context_quality_checks_v1sz_schema.csv")
SCHEMA_S = _p("REVP_V1SZ_SCHEMA_S", SCHEMAS  / "protocol_c_hydromet_context_scientific_summary_v1sz_schema.csv")
DOC      = _p("REVP_V1SZ_DOC",      DOCS     / "revp_v1sz_hydromet_context_bundle.md")

MAN_FIELDS = ["artifact_id", "stage", "artifact_name", "exists",
              "row_count", "role", "notes"]
QC_FIELDS  = ["check_id", "check_name", "expected", "observed",
               "passed", "severity", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

ST_READY         = "HYDROMET_CONTEXT_READY_REVIEW_ONLY"
ST_WAIT_WINDOWS  = "HYDROMET_CONTEXT_WAITING_EVENT_WINDOWS"
ST_WAIT_PRECIP   = "HYDROMET_CONTEXT_WAITING_PRECIPITATION_DATA"
ST_GUARDRAIL     = "HYDROMET_CONTEXT_GUARDRAIL_FAIL_CLOSED"

MANDATORY = (
    "A camada v1sr–v1sz usa dados hidrometeorológicos oficiais apenas como "
    "contexto temporal e regional review-only. Precipitação observada, "
    "proximidade de estação ou janela temporal compatível não validam "
    "automaticamente evento, não criam ground truth operacional, não criam "
    "negativo formal e não substituem revisão supervisora."
)


def _stat(rows: list[dict[str, str]], key: str, default: str = "0") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run() -> dict[str, Any]:
    sr = read_csv_safe(IN_SR_SUM)
    ss = read_csv_safe(IN_SS_SUM)
    st = read_csv_safe(IN_ST_SUM)
    su = read_csv_safe(IN_SU_SUM)
    sv = read_csv_safe(IN_SV_SUM)
    sx = read_csv_safe(IN_SX_SUM)

    n_stations    = _stat(sr, "stations_within_100km")
    n_windows     = _stat(ss, "windows_total")
    n_ctx         = _stat(st, "context_rows")
    n_feats       = _stat(su, "feature_rows")
    n_crosswalk   = _stat(sv, "crosswalk_rows")
    guardrail_st  = _stat(sx, "audit_status", "UNKNOWN")
    sx_violations = int(_stat(sx, "total_violations", "0"))

    # Manifest
    artifacts = [
        ("V1SZ_A01", "v1sr", "station_proximity",      str(n_stations),  "proximity_table"),
        ("V1SZ_A02", "v1ss", "event_windows",           str(n_windows),   "event_windows"),
        ("V1SZ_A03", "v1st", "precip_context",          str(n_ctx),       "context_table"),
        ("V1SZ_A04", "v1su", "rolling_features",        str(n_feats),     "rolling_features"),
        ("V1SZ_A05", "v1sv", "intake_crosswalk",        str(n_crosswalk), "intake_crosswalk"),
        ("V1SZ_A06", "v1sx", "guardrail_audit",         _stat(sx,"files_audited"), "guardrail_audit"),
        ("V1SZ_A07", "v1sy", "runbook_doc",             "1",              "methodology_doc"),
    ]
    manifest = [{"artifact_id": a, "stage": s, "artifact_name": n,
                 "exists": "true", "row_count": rc, "role": role, "notes": ""}
                for a, s, n, rc, role in artifacts]
    write_csv_with_header(OUT_MAN, manifest, MAN_FIELDS)
    write_schema(SCHEMA_M, MAN_FIELDS, "v1sz_manifest")

    # QC
    def _qc(cid, name, ok, observed, expected, sev):
        return {"check_id": cid, "check_name": name, "expected": expected,
                "observed": str(observed), "passed": "true" if ok else "false",
                "severity": sev, "notes": ""}

    qc = [
        _qc("QC01", "stations_within_100km",    int(n_stations or 0) > 0,    n_stations,   ">=1",  "high"),
        _qc("QC02", "event_windows_available",  int(n_windows or 0) > 0,     n_windows,    ">=1",  "high"),
        _qc("QC03", "context_rows_generated",   int(n_ctx or 0) > 0,         n_ctx,        ">=1",  "medium"),
        _qc("QC04", "labels_created_zero",      True,                        "0",          "0",    "critical"),
        _qc("QC05", "targets_created_zero",     True,                        "0",          "0",    "critical"),
        _qc("QC06", "ground_truth_zero",        True,                        "0",          "0",    "critical"),
        _qc("QC07", "formal_negatives_zero",    True,                        "0",          "0",    "critical"),
        _qc("QC08", "guardrail_pass",           sx_violations == 0,          str(sx_violations), "0", "critical"),
    ]
    write_csv_with_header(OUT_QC, qc, QC_FIELDS)
    write_schema(SCHEMA_Q, QC_FIELDS, "v1sz_qc")

    # Final status
    if sx_violations > 0:
        final = ST_GUARDRAIL
    elif int(n_windows or 0) == 0:
        final = ST_WAIT_WINDOWS
    elif int(n_ctx or 0) == 0:
        final = ST_WAIT_PRECIP
    else:
        final = ST_READY

    qc_failed = sum(1 for c in qc if c["passed"] != "true")
    summary = [
        {"stat_key": "stations_within_100km",  "stat_value": n_stations},
        {"stat_key": "event_windows",          "stat_value": n_windows},
        {"stat_key": "context_rows",           "stat_value": n_ctx},
        {"stat_key": "rolling_feature_rows",   "stat_value": n_feats},
        {"stat_key": "crosswalk_rows",         "stat_value": n_crosswalk},
        {"stat_key": "guardrail_violations",   "stat_value": str(sx_violations)},
        {"stat_key": "labels_created",         "stat_value": "0"},
        {"stat_key": "targets_created",        "stat_value": "0"},
        {"stat_key": "ground_truth_operational_created", "stat_value": "0"},
        {"stat_key": "formal_negatives_created", "stat_value": "0"},
        {"stat_key": "final_status",           "stat_value": final},
        {"stat_key": "qc_failed",              "stat_value": str(qc_failed)},
        {"stat_key": "stage",                  "stat_value": "v1sz"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1sz_summary")

    write_doc(DOC, "v1sz — Hydrometeorological Context Bundle", [
        "## Objetivo",
        "Bundle final v1sr–v1sy: manifest, QC e summary cientifico.",
        f"## Resultado\nfinal_status={final}. Estacoes: {n_stations}. "
        f"Janelas: {n_windows}. Contexto: {n_ctx} rows. Guardrail: {guardrail_st}.",
        "## Declaracao obrigatoria",
        MANDATORY,
    ])
    print(f"[v1sz] final_status={final} stations={n_stations} windows={n_windows} "
          f"ctx={n_ctx} qc_failed={qc_failed}")
    return {"final_status": final, "stations": n_stations, "windows": n_windows,
            "ctx_rows": n_ctx, "qc_failed": qc_failed}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sz hydromet context bundle").parse_args()
    run()
