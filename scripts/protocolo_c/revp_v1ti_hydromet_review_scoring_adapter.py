"""REV-P v1ti — Hydromet review scoring adapter.

Reads human responses from REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH
if set; otherwise emits WAITING status with header. Scoring is contextual
only — never influences label or C4.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_extended, scan_guardrails_extended,
    hash_short, parse_float_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_SCR  = _p("REVP_V1TI_OUT_SCR",  DATASETS / "protocol_c_hydromet_review_scores_v1ti.csv")
OUT_SUM  = _p("REVP_V1TI_OUT_SUM",  DATASETS / "protocol_c_hydromet_review_scoring_summary_v1ti.csv")
SCHEMA_SC= _p("REVP_V1TI_SCHEMA_SC",SCHEMAS  / "protocol_c_hydromet_review_scores_v1ti_schema.csv")
SCHEMA_S = _p("REVP_V1TI_SCHEMA_S", SCHEMAS  / "protocol_c_hydromet_review_scoring_summary_v1ti_schema.csv")
DOC      = _p("REVP_V1TI_DOC",      DOCS     / "revp_v1ti_hydromet_review_scoring_adapter.md")

SCR_FIELDS = [
    "score_id", "event_candidate_id", "region", "addendum_id",
    "hydromet_context_score", "station_relevance_score",
    "temporal_context_score", "overclaim_risk_score",
    "scoring_status",
    "review_only", "hydromet_validates_event", "hydromet_is_negative_evidence",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

ST_WAITING = "HYDROMET_REVIEW_RESPONSES_WAITING"
ST_SCORED  = "HYDROMET_REVIEW_CONTEXT_SCORED_REVIEW_ONLY"
ST_FAIL    = "HYDROMET_REVIEW_SCORING_FAIL_CLOSED"

# Mapping from question_key to score dimension
_Q_TO_DIM = {
    "hydromet_precipitation_present":    "hydromet_context_score",
    "hydromet_station_proximity_adequate": "station_relevance_score",
    "hydromet_temporal_compatibility":   "temporal_context_score",
    "hydromet_overclaim_risk":           "overclaim_risk_score",
}


def _score_response(response: str) -> float:
    """Convert yes/no/uncertain to 1.0/0.0/0.5. Contextual only."""
    lo = str(response or "").strip().lower()
    if lo in ("sim", "yes", "s", "1", "true"):
        return 1.0
    if lo in ("nao", "no", "n", "0", "false", "nao"):
        return 0.0
    return 0.5  # incerto / empty


def _load_responses(path: Path) -> list[dict[str, str]]:
    try:
        return read_csv_safe(path)
    except Exception:
        return []


def run() -> dict[str, Any]:
    responses_path_env = os.environ.get("REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH", "")
    addenda = read_csv_safe(DATASETS / "protocol_c_hydromet_double_review_addendum_manifest_v1th.csv")
    forms   = read_csv_safe(DATASETS / "protocol_c_hydromet_double_review_addendum_forms_v1th.csv")

    rows: list[dict[str, Any]] = []

    if not responses_path_env:
        # Fail-closed: waiting
        for a in addenda:
            if not a.get("event_candidate_id"):
                continue
            row: dict[str, Any] = {
                "score_id":               f"V1TI_SC_{hash_short(a['event_candidate_id'],10)}",
                "event_candidate_id":     a["event_candidate_id"],
                "region":                 a.get("region", ""),
                "addendum_id":            a.get("addendum_id", ""),
                "hydromet_context_score": "",
                "station_relevance_score": "",
                "temporal_context_score": "",
                "overclaim_risk_score":   "",
                "scoring_status":         ST_WAITING,
                "notes":                  "REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH not set",
            }
            row.update(guardrail_row_extended())
            rows.append(row)
    else:
        # Load responses and compute contextual scores
        resp_rows = _load_responses(Path(responses_path_env))
        # Index by (addendum_id, question_key)
        resp_index: dict[tuple[str, str], str] = {}
        for r in resp_rows:
            aid = r.get("addendum_id", "")
            qk  = r.get("question_key", "")
            rv  = r.get("response_value", "")
            if aid and qk:
                resp_index[(aid, qk)] = rv

        # One score row per addendum
        scored_ids: set[str] = set()
        for a in addenda:
            cid = a.get("event_candidate_id", "")
            aid = a.get("addendum_id", "")
            if not cid or aid in scored_ids:
                continue
            scored_ids.add(aid)

            scores: dict[str, str] = {}
            for qk, dim in _Q_TO_DIM.items():
                rv = resp_index.get((aid, qk), "")
                scores[dim] = f"{_score_response(rv):.1f}" if rv else ""

            status = ST_SCORED if any(scores.values()) else ST_FAIL
            row = {
                "score_id":                f"V1TI_SC_{hash_short(cid, 10)}",
                "event_candidate_id":      cid,
                "region":                  a.get("region", ""),
                "addendum_id":             aid,
                "hydromet_context_score":  scores.get("hydromet_context_score", ""),
                "station_relevance_score": scores.get("station_relevance_score", ""),
                "temporal_context_score":  scores.get("temporal_context_score", ""),
                "overclaim_risk_score":    scores.get("overclaim_risk_score", ""),
                "scoring_status":          status,
                "notes":                   "",
            }
            row.update(guardrail_row_extended())
            rows.append(row)

    if not rows:
        rows = [{
            "score_id": "FAIL_CLOSED_NO_ADDENDA", "event_candidate_id": "",
            "region": "", "addendum_id": "",
            "hydromet_context_score": "", "station_relevance_score": "",
            "temporal_context_score": "", "overclaim_risk_score": "",
            "scoring_status": ST_FAIL,
            "notes": "no addenda found", **guardrail_row_extended(),
        }]

    violations = scan_guardrails_extended(rows, "v1ti_scores")
    if violations:
        raise ValueError(f"Guardrail violations in v1ti: {violations[:3]}")

    write_csv_with_header(OUT_SCR, rows, SCR_FIELDS)
    write_schema(SCHEMA_SC, SCR_FIELDS, "v1ti_scores")

    waiting = sum(1 for r in rows if r["scoring_status"] == ST_WAITING)
    scored  = sum(1 for r in rows if r["scoring_status"] == ST_SCORED)
    overall = ST_WAITING if not responses_path_env else (ST_SCORED if scored > 0 else ST_FAIL)
    summary = [
        {"stat_key": "score_rows",            "stat_value": str(len(rows))},
        {"stat_key": "waiting_responses",     "stat_value": str(waiting)},
        {"stat_key": "scored_review_only",    "stat_value": str(scored)},
        {"stat_key": "influences_label",      "stat_value": "false"},
        {"stat_key": "influences_c4",         "stat_value": "false"},
        {"stat_key": "overall_status",        "stat_value": overall},
        {"stat_key": "stage",                 "stat_value": "v1ti"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1ti_summary")

    write_doc(DOC, "v1ti — Hydromet Review Scoring Adapter", [
        "## Objetivo",
        "Estrutura de pontuação para respostas do revisor ao addendum hidromet. "
        "Scoring contextual apenas — nunca influencia label ou C4.",
        f"## Resultado\nStatus: {overall}. Rows: {len(rows)}. "
        f"Waiting: {waiting}. Scored: {scored}.",
        "## Notas",
        "Definir REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH para ativar scoring.",
    ])
    print(f"[v1ti] status={overall} rows={len(rows)} waiting={waiting} scored={scored}")
    return {"status": overall, "rows": len(rows), "waiting": waiting}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ti hydromet review scoring").parse_args()
    run()
