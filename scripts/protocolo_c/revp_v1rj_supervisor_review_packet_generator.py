"""REV-P v1rj — Supervisor review packet generator.

Generates supervisor/orientador packets for completed reviews that reach a
C3-candidate or disagreement state and still require a final human decision.
Waiting status when no completed reviews exist. Never creates a label.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rg_v1rm_review_response_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    C3_NEEDS_SUPERVISOR,
    SUP_APPROVE_C3,
    SUP_BLOCK_SOURCE,
    SUP_BLOCK_SPATIAL,
    SUP_BLOCK_TEMPORAL,
    SUP_KEEP_C2,
    SUP_REQUEST_MORE,
    _p,
    assert_clean_rows,
    guardrail_row,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_SCORES = _p("REVP_V1RJ_IN_SCORES", DATASETS / "protocol_c_completed_review_scores_v1ri.csv")
OUT_MANIFEST = _p("REVP_V1RJ_OUT_MANIFEST", DATASETS / "protocol_c_supervisor_review_packet_manifest_v1rj.csv")
OUT_FORMS = _p("REVP_V1RJ_OUT_FORMS", DATASETS / "protocol_c_supervisor_review_forms_v1rj.csv")
OUT_SUMMARY = _p("REVP_V1RJ_OUT_SUMMARY", DATASETS / "protocol_c_supervisor_review_summary_v1rj.csv")
SCHEMA_MANIFEST = _p("REVP_V1RJ_SCHEMA_MANIFEST", SCHEMAS / "protocol_c_supervisor_review_packet_manifest_v1rj_schema.csv")
SCHEMA_FORMS = _p("REVP_V1RJ_SCHEMA_FORMS", SCHEMAS / "protocol_c_supervisor_review_forms_v1rj_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RJ_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_supervisor_review_summary_v1rj_schema.csv")
DOC = _p("REVP_V1RJ_DOC", DOCS / "revp_v1rj_supervisor_review_packet_generator.md")

MANIFEST_FIELDS = [
    "supervisor_packet_id", "review_sample_id", "event_id", "patch_id", "region",
    "composite_review_score", "disagreement_flag", "source_evidence_summary",
    "temporal_evidence_summary", "spatial_evidence_summary", "recommended_decision",
    "supervisor_action_required", "can_promote_to_c3_candidate",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "notes",
]

FORM_FIELDS = [
    "form_id", "supervisor_packet_id", "review_sample_id", "question_key",
    "question_text", "answer_placeholder", "response_value", "review_only", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

WAITING = "SUPERVISOR_PACKETS_WAITING_COMPLETED_REVIEWS"
READY = "SUPERVISOR_PACKETS_READY"

SUPERVISOR_QUESTIONS = [
    ("supervisor_decision", "Decisao do supervisor (APPROVE_C3_CANDIDATE_REVIEW_ONLY/KEEP_C2_REVIEW_ONLY/BLOCK_*/REQUEST_ADDITIONAL_REVIEW)"),
    ("decision_confidence_0_4", "Confianca da decisao (0 a 4)"),
    ("required_followup", "Acao de followup necessaria (texto livre)"),
    ("decision_note", "Justificativa da decisao (texto livre)"),
]


def _f(value: str | None) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _recommend_action(s: dict[str, str]) -> tuple[str, str]:
    """Return (supervisor_action_required, can_promote_to_c3_candidate)."""
    composite = _f(s.get("composite_review_score"))
    temporal = _f(s.get("temporal_support_score"))
    spatial = _f(s.get("spatial_support_score"))
    source = _f(s.get("source_support_score"))
    disagreement = str(s.get("disagreement_flag", "")).lower() == "true"
    rec = s.get("recommended_decision", "")

    if disagreement:
        return SUP_REQUEST_MORE, "false"
    if source < 0.75:
        return SUP_BLOCK_SOURCE, "false"
    if temporal < 0.6:
        return SUP_BLOCK_TEMPORAL, "false"
    if spatial < 0.6:
        return SUP_BLOCK_SPATIAL, "false"
    if rec == C3_NEEDS_SUPERVISOR and composite >= 0.75:
        return SUP_APPROVE_C3, "true"
    return SUP_KEEP_C2, "false"


def _needs_packet(s: dict[str, str]) -> bool:
    both = s.get("reviewer_a_present") == "true" and s.get("reviewer_b_present") == "true"
    if not both:
        return False
    return (s.get("supervisor_review_required") == "true"
            or str(s.get("disagreement_flag", "")).lower() == "true")


def build(scores: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    forms: list[dict[str, Any]] = []
    for s in scores:
        if not _needs_packet(s):
            continue
        rsid = s.get("review_sample_id", "")
        packet_id = f"V1RJ_SPKT_{rsid}"
        action, can_promote = _recommend_action(s)
        row = {
            "supervisor_packet_id": packet_id, "review_sample_id": rsid,
            "event_id": s.get("event_id", ""), "patch_id": s.get("patch_id", ""),
            "region": s.get("region", ""),
            "composite_review_score": s.get("composite_review_score", ""),
            "disagreement_flag": s.get("disagreement_flag", "false"),
            "source_evidence_summary": f"source_support={s.get('source_support_score','')}",
            "temporal_evidence_summary": f"temporal_support={s.get('temporal_support_score','')}",
            "spatial_evidence_summary": f"spatial_support={s.get('spatial_support_score','')}",
            "recommended_decision": s.get("recommended_decision", ""),
            "supervisor_action_required": action,
            "can_promote_to_c3_candidate": can_promote,
            "notes": "review_only_supervisor_decision_pending",
        }
        row.update(guardrail_row())
        manifest.append(row)
        for q, text in SUPERVISOR_QUESTIONS:
            forms.append({
                "form_id": f"V1RJ_FORM_{rsid}_{q}", "supervisor_packet_id": packet_id,
                "review_sample_id": rsid, "question_key": q, "question_text": text,
                "answer_placeholder": "<TO_BE_FILLED_BY_SUPERVISOR>", "response_value": "",
                "review_only": "true", "notes": "",
            })
    return manifest, forms


def run(datasets: Path | None = None) -> dict[str, Any]:
    scores = read_csv_safe(IN_SCORES)
    manifest, forms = build(scores)
    assert_clean_rows(manifest, "v1rj_manifest")

    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv_with_header(OUT_FORMS, forms, FORM_FIELDS)
    write_schema_safe(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1rj_manifest")
    write_schema_safe(SCHEMA_FORMS, FORM_FIELDS, "v1rj_forms")

    status = READY if manifest else WAITING
    approve = sum(1 for r in manifest if r["supervisor_action_required"] == SUP_APPROVE_C3)
    promote = sum(1 for r in manifest if r["can_promote_to_c3_candidate"] == "true")
    summary = [
        {"stat_key": "supervisor_status", "stat_value": status},
        {"stat_key": "supervisor_packets", "stat_value": str(len(manifest))},
        {"stat_key": "supervisor_forms", "stat_value": str(len(forms))},
        {"stat_key": "approve_c3_candidate_actions", "stat_value": str(approve)},
        {"stat_key": "can_promote_to_c3_candidate", "stat_value": str(promote)},
        {"stat_key": "stage", "stat_value": "v1rj"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rj_summary")

    write_doc(
        DOC,
        "v1rj — Supervisor Review Packet Generator",
        [
            "## Objetivo",
            "Gerar pacotes para o supervisor quando uma revisao completa alcanca estado "
            "C3-candidate ou desacordo e ainda precisa decisao humana final. Sem revisoes "
            "completas, SUPERVISOR_PACKETS_WAITING_COMPLETED_REVIEWS.",
            "## Acoes do supervisor",
            "APPROVE_C3_CANDIDATE_REVIEW_ONLY, KEEP_C2_REVIEW_ONLY, BLOCK_C3_NEEDS_MORE_SOURCE, "
            "BLOCK_C3_NEEDS_BETTER_TEMPORAL_PRECISION, BLOCK_C3_NEEDS_BETTER_SPATIAL_PRECISION, "
            "REQUEST_ADDITIONAL_REVIEW.",
            "## Resultado",
            f"Status: {status}. Pacotes: {len(manifest)}. Promoviveis a C3 candidate: {promote}.",
            "## Guardrails",
            "can_promote_to_c3_candidate=true significa apenas elegibilidade review-only. "
            "can_create_operational_label=false sempre. Desacordo exige revisao adicional.",
        ],
    )
    print(f"[v1rj] status={status} packets={len(manifest)} promote={promote}")
    return {"status": status, "packets": len(manifest), "promote": promote}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rj supervisor packet generator").parse_args()
    run()
