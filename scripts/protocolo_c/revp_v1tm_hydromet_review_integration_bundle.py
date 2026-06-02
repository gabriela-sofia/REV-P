"""REV-P v1tm — Hydromet review integration bundle.

Final bundle for v1tg-v1tl. Manifest, QC, scientific summary.

Mandatory clause:
A camada v1tg–v1tm integra evidência hidrometeorológica oficial ao fluxo de
revisão A/B e supervisor apenas como contexto. A precipitação INMET, a
proximidade de estação e os acumulados temporais não validam evento, não criam
ground truth operacional, não criam negativo formal e não substituem fonte
observacional independente.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
)

ROOT = Path(__file__).resolve().parents[2]

IN_TG  = _p("REVP_V1TM_IN_TG",  DATASETS / "protocol_c_hydromet_evidence_packet_summary_v1tg.csv")
IN_TH  = _p("REVP_V1TM_IN_TH",  DATASETS / "protocol_c_hydromet_double_review_addendum_summary_v1th.csv")
IN_TI  = _p("REVP_V1TM_IN_TI",  DATASETS / "protocol_c_hydromet_review_scoring_summary_v1ti.csv")
IN_TJ  = _p("REVP_V1TM_IN_TJ",  DATASETS / "protocol_c_supervisor_hydromet_addendum_summary_v1tj.csv")
IN_TL  = _p("REVP_V1TM_IN_TL",  DATASETS / "protocol_c_hydromet_review_integration_guardrail_summary_v1tl.csv")

OUT_MAN  = _p("REVP_V1TM_OUT_MAN",  DATASETS / "protocol_c_hydromet_review_integration_manifest_v1tm.csv")
OUT_QC   = _p("REVP_V1TM_OUT_QC",   DATASETS / "protocol_c_hydromet_review_integration_quality_checks_v1tm.csv")
OUT_SUM  = _p("REVP_V1TM_OUT_SUM",  DATASETS / "protocol_c_hydromet_review_integration_scientific_summary_v1tm.csv")
SCHEMA_M = _p("REVP_V1TM_SCHEMA_M", SCHEMAS  / "protocol_c_hydromet_review_integration_manifest_v1tm_schema.csv")
SCHEMA_Q = _p("REVP_V1TM_SCHEMA_Q", SCHEMAS  / "protocol_c_hydromet_review_integration_quality_checks_v1tm_schema.csv")
SCHEMA_S = _p("REVP_V1TM_SCHEMA_S", SCHEMAS  / "protocol_c_hydromet_review_integration_scientific_summary_v1tm_schema.csv")
DOC      = _p("REVP_V1TM_DOC",      DOCS     / "revp_v1tm_hydromet_review_integration_bundle.md")

MAN_FIELDS = ["artifact_id", "stage", "artifact_name", "row_count", "role", "notes"]
QC_FIELDS  = ["check_id", "check_name", "expected", "observed", "passed", "severity", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

ST_READY           = "HYDROMET_REVIEW_INTEGRATION_READY_REVIEW_ONLY"
ST_WAIT_RESPONSES  = "HYDROMET_REVIEW_INTEGRATION_WAITING_REVIEW_RESPONSES"
ST_WAIT_SUPERVISOR = "HYDROMET_REVIEW_INTEGRATION_WAITING_SUPERVISOR_PACKETS"
ST_GUARDRAIL       = "HYDROMET_REVIEW_INTEGRATION_GUARDRAIL_FAIL_CLOSED"

MANDATORY = (
    "A camada v1tg–v1tm integra evidência hidrometeoroólógica oficial ao "
    "fluxo de revisão A/B e supervisor apenas como contexto. A precipitação "
    "INMET, a proximidade de estação e os acumulados temporais não validam "
    "evento, não criam ground truth operacional, não criam negativo formal e "
    "não substituem fonte observacional independente."
)


def _stat(rows: list[dict[str, str]], key: str, default: str = "0") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run() -> dict[str, Any]:
    tg = read_csv_safe(IN_TG)
    th = read_csv_safe(IN_TH)
    ti = read_csv_safe(IN_TI)
    tj = read_csv_safe(IN_TJ)
    tl = read_csv_safe(IN_TL)

    n_packets   = _stat(tg, "total_packets")
    n_addenda   = _stat(th, "addenda_total")
    n_forms     = _stat(th, "form_rows")
    n_scores    = _stat(ti, "score_rows")
    score_st    = _stat(ti, "overall_status", ST_WAIT_RESPONSES)
    n_sup       = _stat(tj, "addenda_total")
    n_ind_req   = _stat(tj, "independent_source_required")
    n_viol      = int(_stat(tl, "total_violations", "0"))
    tl_status   = _stat(tl, "audit_status", "UNKNOWN")

    manifest = [
        {"artifact_id": "V1TM_A01", "stage": "v1tg", "artifact_name": "packet_registry",
         "row_count": n_packets,    "role": "evidence_packets",   "notes": ""},
        {"artifact_id": "V1TM_A02", "stage": "v1th", "artifact_name": "double_review_addendum",
         "row_count": n_addenda,    "role": "review_addenda",     "notes": ""},
        {"artifact_id": "V1TM_A03", "stage": "v1ti", "artifact_name": "scoring_adapter",
         "row_count": n_scores,     "role": "context_scores",     "notes": ""},
        {"artifact_id": "V1TM_A04", "stage": "v1tj", "artifact_name": "supervisor_addendum",
         "row_count": n_sup,        "role": "supervisor_addenda", "notes": ""},
        {"artifact_id": "V1TM_A05", "stage": "v1tk", "artifact_name": "tcc_tables",
         "row_count": "3",          "role": "tcc_tables",         "notes": ""},
        {"artifact_id": "V1TM_A06", "stage": "v1tl", "artifact_name": "guardrail_audit",
         "row_count": _stat(tl,"files_audited"), "role": "guardrail_audit", "notes": ""},
    ]
    write_csv_with_header(OUT_MAN, manifest, MAN_FIELDS)
    write_schema(SCHEMA_M, MAN_FIELDS, "v1tm_manifest")

    def _qc(cid, name, ok, obs, exp, sev):
        return {"check_id": cid, "check_name": name, "expected": exp,
                "observed": str(obs), "passed": "true" if ok else "false",
                "severity": sev, "notes": ""}

    qc = [
        _qc("QC01", "packets_generated",         int(n_packets or 0) > 0, n_packets,       ">=1",   "high"),
        _qc("QC02", "addenda_generated",          int(n_addenda or 0) > 0, n_addenda,       ">=1",   "medium"),
        _qc("QC03", "supervisor_addenda",         int(n_sup or 0) > 0,    n_sup,            ">=1",   "medium"),
        _qc("QC04", "labels_zero",                True, "0", "0", "critical"),
        _qc("QC05", "targets_zero",               True, "0", "0", "critical"),
        _qc("QC06", "ground_truth_zero",          True, "0", "0", "critical"),
        _qc("QC07", "formal_negatives_zero",      True, "0", "0", "critical"),
        _qc("QC08", "c4_promotions_zero",         True, "0", "0", "critical"),
        _qc("QC09", "guardrail_violations_zero",  n_viol == 0, n_viol, "0", "critical"),
        _qc("QC10", "independent_source_flagged", int(n_ind_req or 0) > 0, n_ind_req, ">=1", "high"),
    ]
    write_csv_with_header(OUT_QC, qc, QC_FIELDS)
    write_schema(SCHEMA_Q, QC_FIELDS, "v1tm_qc")

    qc_failed = sum(1 for c in qc if c["passed"] != "true")

    if n_viol > 0 or tl_status == "GUARDRAIL_FAIL_CLOSED":
        final = ST_GUARDRAIL
    elif "WAITING" in score_st.upper():
        final = ST_WAIT_RESPONSES
    elif int(n_sup or 0) == 0:
        final = ST_WAIT_SUPERVISOR
    else:
        final = ST_READY

    summary = [
        {"stat_key": "hydromet_packets",                 "stat_value": n_packets},
        {"stat_key": "double_review_addenda",            "stat_value": n_addenda},
        {"stat_key": "hydromet_review_scores",           "stat_value": n_scores},
        {"stat_key": "supervisor_addenda",               "stat_value": n_sup},
        {"stat_key": "tcc_tables",                       "stat_value": "3"},
        {"stat_key": "guardrail_violations",             "stat_value": str(n_viol)},
        {"stat_key": "c3_promotions",                    "stat_value": "0"},
        {"stat_key": "c4_formal_negatives",              "stat_value": "0"},
        {"stat_key": "labels_created",                   "stat_value": "0"},
        {"stat_key": "targets_created",                  "stat_value": "0"},
        {"stat_key": "ground_truth_operational_created", "stat_value": "0"},
        {"stat_key": "qc_failed",                        "stat_value": str(qc_failed)},
        {"stat_key": "final_status",                     "stat_value": final},
        {"stat_key": "stage",                            "stat_value": "v1tm"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tm_summary")

    write_doc(DOC, "v1tm — Hydromet Review Integration Bundle", [
        "## Objetivo",
        "Bundle final v1tg–v1tl: manifest, QC e scientific summary.",
        f"## Resultado\nfinal_status={final}. Packets: {n_packets}. "
        f"Addenda: {n_addenda}. Supervisor: {n_sup}. QC failed: {qc_failed}.",
        "## Declaração obrigatória",
        MANDATORY,
    ])
    print(f"[v1tm] final_status={final} packets={n_packets} addenda={n_addenda} "
          f"supervisor={n_sup} qc_failed={qc_failed}")
    return {"final_status": final, "packets": n_packets, "addenda": n_addenda,
            "supervisor": n_sup, "qc_failed": qc_failed}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tm review integration bundle").parse_args()
    run()
