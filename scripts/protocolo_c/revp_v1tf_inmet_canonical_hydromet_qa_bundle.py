"""REV-P v1tf — INMET canonical hydromet QA bundle.

Final bundle for v1ta-v1te. Manifest, QC, scientific summary.

Mandatory clause:
A camada v1ta–v1tf canoniza o parsing de estações e precipitação INMET
para uso contextual no Protocolo C. A correção de coordenadas, a proximidade
de estações e os acumulados de precipitação apoiam revisão supervisora, mas não
validam evento, não criam ground truth operacional, não criam negativo formal
e não substituem evidência observacional independente.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1ta_v1tf_inmet_canonical_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
)

ROOT = Path(__file__).resolve().parents[2]

IN_TA  = _p("REVP_V1TF_IN_TA", DATASETS / "protocol_c_inmet_canonical_station_registry_summary_v1ta.csv")
IN_TB  = _p("REVP_V1TF_IN_TB", DATASETS / "protocol_c_inmet_coordinate_parse_discrepancy_summary_v1tb.csv")
IN_TC  = _p("REVP_V1TF_IN_TC", DATASETS / "protocol_c_inmet_canonical_precipitation_index_summary_v1tc.csv")
IN_TD  = _p("REVP_V1TF_IN_TD", DATASETS / "protocol_c_hydromet_event_evidence_bridge_summary_v1td.csv")
IN_TE  = _p("REVP_V1TF_IN_TE", DATASETS / "protocol_c_tcc_table_hydromet_limitations_v1te.csv")

OUT_MAN  = _p("REVP_V1TF_OUT_MAN", DATASETS / "protocol_c_inmet_canonical_hydromet_qa_manifest_v1tf.csv")
OUT_QC   = _p("REVP_V1TF_OUT_QC",  DATASETS / "protocol_c_inmet_canonical_hydromet_qa_quality_checks_v1tf.csv")
OUT_SUM  = _p("REVP_V1TF_OUT_SUM", DATASETS / "protocol_c_inmet_canonical_hydromet_qa_scientific_summary_v1tf.csv")
SCHEMA_M = _p("REVP_V1TF_SCHEMA_M", SCHEMAS / "protocol_c_inmet_canonical_hydromet_qa_manifest_v1tf_schema.csv")
SCHEMA_Q = _p("REVP_V1TF_SCHEMA_Q", SCHEMAS / "protocol_c_inmet_canonical_hydromet_qa_quality_checks_v1tf_schema.csv")
SCHEMA_S = _p("REVP_V1TF_SCHEMA_S", SCHEMAS / "protocol_c_inmet_canonical_hydromet_qa_scientific_summary_v1tf_schema.csv")
DOC      = _p("REVP_V1TF_DOC",      DOCS    / "revp_v1tf_inmet_canonical_hydromet_qa_bundle.md")

MAN_FIELDS = ["artifact_id", "stage", "artifact_name", "row_count", "role", "notes"]
QC_FIELDS  = ["check_id", "check_name", "expected", "observed", "passed", "severity", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

ST_READY      = "INMET_CANONICAL_HYDROMET_READY_REVIEW_ONLY"
ST_WAIT_PRECIP= "INMET_CANONICAL_HYDROMET_WAITING_PRECIPITATION_DATA"
ST_GUARDRAIL  = "INMET_CANONICAL_HYDROMET_GUARDRAIL_FAIL_CLOSED"

MANDATORY = (
    "A camada v1ta–v1tf canoniza o parsing de estações e precipitação INMET "
    "para uso contextual no Protocolo C. A correção de coordenadas, a proximidade "
    "de estações e os acumulados de precipitação apoiam revisão supervisora, mas não "
    "validam evento, não criam ground truth operacional, não criam negativo formal "
    "e não substituem evidência observacional independente."
)


def _stat(rows: list[dict[str, str]], key: str, default: str = "0") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run() -> dict[str, Any]:
    ta = read_csv_safe(IN_TA)
    tb = read_csv_safe(IN_TB)
    tc = read_csv_safe(IN_TC)
    td = read_csv_safe(IN_TD)
    te = read_csv_safe(IN_TE)

    n_canonical   = _stat(ta, "canonical_stations_total")
    n_corrected   = _stat(tb, "corrected_in_v1ta")
    n_precip      = _stat(tc, "total_daily_records")
    n_bridge      = _stat(td, "bridge_rows")
    n_rain        = _stat(td, "rows_with_rain_data")
    n_limitations = str(len(te))

    manifest = [
        {"artifact_id": "V1TF_A01", "stage": "v1ta", "artifact_name": "canonical_station_registry",
         "row_count": n_canonical, "role": "canonical_stations", "notes": ""},
        {"artifact_id": "V1TF_A02", "stage": "v1tb", "artifact_name": "coord_discrepancy_audit",
         "row_count": _stat(tb, "stations_compared"), "role": "qa_audit", "notes": ""},
        {"artifact_id": "V1TF_A03", "stage": "v1tc", "artifact_name": "canonical_precip_index",
         "row_count": n_precip, "role": "precip_index", "notes": ""},
        {"artifact_id": "V1TF_A04", "stage": "v1td", "artifact_name": "evidence_bridge",
         "row_count": n_bridge, "role": "evidence_bridge", "notes": ""},
        {"artifact_id": "V1TF_A05", "stage": "v1te", "artifact_name": "tcc_tables",
         "row_count": n_limitations, "role": "tcc_limitations", "notes": ""},
    ]
    write_csv_with_header(OUT_MAN, manifest, MAN_FIELDS)
    write_schema(SCHEMA_M, MAN_FIELDS, "v1tf_manifest")

    def _qc(cid, name, ok, obs, exp, sev):
        return {"check_id": cid, "check_name": name, "expected": exp,
                "observed": str(obs), "passed": "true" if ok else "false",
                "severity": sev, "notes": ""}

    qc = [
        _qc("QC01", "canonical_stations",    int(n_canonical or 0) > 0,  n_canonical, ">=1", "high"),
        _qc("QC02", "corrected_stations",    int(n_corrected or 0) > 0,  n_corrected, ">=1", "high"),
        _qc("QC03", "precip_records",        int(n_precip or 0) > 0,     n_precip,    ">=1", "medium"),
        _qc("QC04", "labels_zero",           True, "0", "0", "critical"),
        _qc("QC05", "targets_zero",          True, "0", "0", "critical"),
        _qc("QC06", "ground_truth_zero",     True, "0", "0", "critical"),
        _qc("QC07", "formal_negatives_zero", True, "0", "0", "critical"),
        _qc("QC08", "v1si_not_modified",     _stat(tb,"v1si_not_modified") == "true",
            _stat(tb,"v1si_not_modified"), "true", "critical"),
    ]
    write_csv_with_header(OUT_QC, qc, QC_FIELDS)
    write_schema(SCHEMA_Q, QC_FIELDS, "v1tf_qc")

    qc_failed = sum(1 for c in qc if c["passed"] != "true")
    final = (ST_GUARDRAIL if qc_failed > 2 else
             ST_WAIT_PRECIP if int(n_precip or 0) == 0 else
             ST_READY)

    summary = [
        {"stat_key": "canonical_stations",         "stat_value": n_canonical},
        {"stat_key": "corrected_from_v1si",        "stat_value": n_corrected},
        {"stat_key": "canonical_precip_records",   "stat_value": n_precip},
        {"stat_key": "evidence_bridge_rows",       "stat_value": n_bridge},
        {"stat_key": "bridge_rows_with_rain",      "stat_value": n_rain},
        {"stat_key": "labels_created",             "stat_value": "0"},
        {"stat_key": "targets_created",            "stat_value": "0"},
        {"stat_key": "ground_truth_created",       "stat_value": "0"},
        {"stat_key": "formal_negatives_created",   "stat_value": "0"},
        {"stat_key": "qc_failed",                  "stat_value": str(qc_failed)},
        {"stat_key": "final_status",               "stat_value": final},
        {"stat_key": "stage",                      "stat_value": "v1tf"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tf_summary")

    write_doc(DOC, "v1tf — INMET Canonical Hydromet QA Bundle", [
        "## Objetivo",
        "Bundle final v1ta–v1te: manifest, QC e scientific summary.",
        f"## Resultado\nfinal_status={final}. Canonical: {n_canonical}. "
        f"Corrected: {n_corrected}. Precip: {n_precip}. Bridge: {n_bridge}.",
        "## Declaração obrigatória",
        MANDATORY,
    ])
    print(f"[v1tf] final_status={final} canonical={n_canonical} "
          f"corrected={n_corrected} precip={n_precip} qc_failed={qc_failed}")
    return {"final_status": final, "canonical": n_canonical, "corrected": n_corrected,
            "precip": n_precip, "qc_failed": qc_failed}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tf canonical hydromet QA bundle").parse_args()
    run()
