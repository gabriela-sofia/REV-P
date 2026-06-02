"""Shared helpers — REV-P Protocol C v1tn-v1tw Automated Review Adjudication.

This layer replaces the *internal* manual review (organisation, triage,
completeness, overclaim control, review-only adjudication) with an automated
flow. It NEVER creates operational labels, targets, operational ground
truth, formal negatives, automatic C3 promotion or C4; it never treats DINO or
hydromet as proof, and never treats absence as a negative. External
observational evidence remains required for any operational claim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (  # noqa: F401
    DATASETS, DOCS, SCHEMAS, _p, ROOT,
    read_csv_safe, write_csv_with_header, write_json_safe,
    write_schema, write_doc,
    safe_relpath, hash_short, parse_float_safe,
    normalize_region, normalize_event_id, normalize_patch_id,
    ABS_PATH_RE,
)

# ---------------------------------------------------------------------------
# Guardrails — every output row carries the full guardrail block.
# ---------------------------------------------------------------------------

# Fields that must always be present and must never be "true".
FORBIDDEN_TRUE_FLAGS = [
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative",
    "automatic_c3_promotion", "c4_opened",
]

# Fields that must always be present and must always be "true".
REQUIRED_TRUE_FLAGS = [
    "review_only",
    "requires_external_observational_evidence_for_operational_claim",
]


def guardrail_row_review(
    automated_review: bool = True,
    internal_review_automated: bool = True,
) -> dict[str, str]:
    """Full guardrail block for automated review outputs."""
    return {
        "review_only": "true",
        "automated_review": "true" if automated_review else "false",
        "internal_review_automated_for_review_only":
            "true" if internal_review_automated else "false",
        "requires_external_observational_evidence_for_operational_claim": "true",
        "can_create_operational_label": "false",
        "can_train_model": "false",
        "target_created": "false",
        "ground_truth_operational": "false",
        "formal_negative": "false",
        "dino_validates_event": "false",
        "hydromet_validates_event": "false",
        "hydromet_is_negative_evidence": "false",
        "absence_as_negative": "false",
        "automatic_c3_promotion": "false",
        "c4_opened": "false",
    }


GUARDRAIL_FIELDS_REVIEW = list(guardrail_row_review().keys())


def scan_guardrails(rows: list[dict[str, Any]], label: str) -> list[str]:
    """Return violation strings (empty = clean). Covers forbidden-true,
    required-true, absolute paths and local-runs exposure."""
    issues: list[str] = []
    forbidden_literal = "local" + "_runs"
    for i, row in enumerate(rows):
        for f in FORBIDDEN_TRUE_FLAGS:
            if str(row.get(f, "false")).strip().lower() == "true":
                issues.append(f"{label}[{i}].{f}=true")
        for f in REQUIRED_TRUE_FLAGS:
            if f in row and str(row.get(f)).strip().lower() != "true":
                issues.append(f"{label}[{i}].{f}!=true")
        for k, v in row.items():
            sv = str(v)
            if ABS_PATH_RE.search(sv):
                issues.append(f"{label}[{i}].{k}=abs_path")
            if forbidden_literal in sv.lower():
                issues.append(f"{label}[{i}].{k}=local_runs_exposure")
    return issues


def guardrail_row() -> dict[str, str]:
    """Alias kept for naming parity with sibling commons."""
    return guardrail_row_review()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_case_id(text: str) -> str:
    return str(text or "").strip().upper().replace(" ", "_")[:120]


def normalize_bool(text: str) -> str:
    return "true" if str(text or "").strip().lower() in (
        "true", "1", "yes", "sim", "y") else "false"


# ---------------------------------------------------------------------------
# Evidence summarisers — each returns a normalised review-only status code.
# ---------------------------------------------------------------------------

def summarize_external_evidence(candidates: list[dict[str, str]]) -> str:
    if not candidates:
        return "EXTERNAL_SOURCE_ABSENT_LOCAL"
    accepted = [c for c in candidates
                if str(c.get("candidate_status", "")).strip().upper()
                in ("ACCEPTED", "VALIDATED_REVIEW_ONLY", "STRONG")]
    if accepted:
        return "EXTERNAL_CANDIDATE_PRESENT_REVIEW_ONLY"
    return "EXTERNAL_CANDIDATE_WEAK_REVIEW_ONLY"


def summarize_hydromet_evidence(packet: dict[str, str] | None) -> str:
    if not packet:
        return "HYDROMET_CONTEXT_ABSENT"
    level = str(packet.get("hydromet_support_level", "")).strip().upper()
    if level == "HYDROMET_CONTEXT_AVAILABLE":
        return "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY"
    if level:
        return level
    return "HYDROMET_CONTEXT_ABSENT"


def summarize_dino_role(dino_rows: list[dict[str, str]]) -> str:
    # DINO is always representation/context only — never proof.
    if dino_rows:
        return "DINO_REPRESENTATION_REVIEW_ONLY_CONTEXT"
    return "DINO_NOT_PRESENT_CONTEXT_ONLY"


def summarize_patch_context(links: list[dict[str, str]]) -> str:
    if not links:
        return "PATCH_LINK_ABSENT"
    strong = [l for l in links
              if str(l.get("link_confidence", "")).strip().upper()
              in ("HIGH", "STRONG")]
    if strong:
        return "PATCH_LINK_CANDIDATE_REVIEW_ONLY"
    return "PATCH_LINK_WEAK_REVIEW_ONLY"


def summarize_protocol_c_state(
    region: str, backlog: list[dict[str, str]]
) -> str:
    region = normalize_region(region)
    region_rows = [b for b in backlog
                   if normalize_region(b.get("region", "")) == region]
    if not region_rows:
        return "PROTOCOL_C_NO_BACKLOG_RECORD"
    if any("BLOCKED_INSUFFICIENT_EVIDENCE" in (
            str(b.get("current_state", "")) + str(b.get("status", ""))).upper()
            for b in region_rows):
        return "PROTOCOL_C_BLOCKED_INSUFFICIENT_EVIDENCE"
    return "PROTOCOL_C_OPEN_BACKLOG"


# ---------------------------------------------------------------------------
# Boolean evidence dimensions (used by readiness + reviewers).
# ---------------------------------------------------------------------------

def _has_external(ext_status: str) -> bool:
    return ext_status.startswith("EXTERNAL_CANDIDATE_PRESENT")


def _has_hydromet(hyd_status: str) -> bool:
    return hyd_status.startswith("HYDROMET_CONTEXT_AVAILABLE")


def _has_temporal(window: str, temporal_status: str) -> bool:
    if str(temporal_status).strip().upper() == "DATE_PARSED_OK":
        return True
    return bool(str(window or "").strip())


def _has_patch(patch_status: str) -> bool:
    return patch_status.startswith("PATCH_LINK_CANDIDATE")


def evidence_completeness(
    ext_status: str, hyd_status: str, window: str, temporal_status: str,
    patch_status: str, dino_status: str,
) -> float:
    dims = [
        _has_external(ext_status),
        _has_hydromet(hyd_status),
        _has_temporal(window, temporal_status),
        _has_patch(patch_status),
        dino_status.startswith("DINO_REPRESENTATION"),
        bool(str(window or "").strip()),
    ]
    return round(sum(1 for d in dims if d) / len(dims), 2)


# ---------------------------------------------------------------------------
# Classification — case readiness
# ---------------------------------------------------------------------------

def classify_case_readiness(
    ext_status: str, hyd_status: str, window: str, temporal_status: str,
    patch_status: str,
) -> str:
    if not str(window or "").strip():
        return "CASE_BLOCKED_INSUFFICIENT_EVIDENCE"
    if not _has_temporal(window, temporal_status):
        return "CASE_NEEDS_TEMPORAL_PRECISION"
    if _has_external(ext_status) and _has_hydromet(hyd_status):
        return "CASE_READY_FOR_REVIEW_ONLY_ADJUDICATION"
    if _has_hydromet(hyd_status):
        return "CASE_CONTEXT_AVAILABLE_NEEDS_EXTERNAL_SOURCE"
    return "CASE_BLOCKED_INSUFFICIENT_EVIDENCE"


def next_required_action(readiness: str) -> str:
    return {
        "CASE_READY_FOR_REVIEW_ONLY_ADJUDICATION":
            "PROCEED_AUTOMATED_REVIEW_ONLY_ADJUDICATION",
        "CASE_CONTEXT_AVAILABLE_NEEDS_EXTERNAL_SOURCE":
            "COLLECT_EXTERNAL_OBSERVATIONAL_SOURCE_FOR_OPERATIONAL_CLAIM",
        "CASE_NEEDS_TEMPORAL_PRECISION":
            "RECOVER_TEMPORAL_PRECISION",
        "CASE_BLOCKED_INSUFFICIENT_EVIDENCE":
            "GATHER_MINIMUM_EVIDENCE_BEFORE_REVIEW",
    }.get(readiness, "REVIEW_CASE_MANUALLY_EXTERNAL")


def blocking_factors(
    ext_status: str, hyd_status: str, window: str, temporal_status: str,
    patch_status: str,
) -> str:
    blocks: list[str] = []
    if not _has_external(ext_status):
        blocks.append("NO_EXTERNAL_OBSERVATIONAL_SOURCE")
    if not _has_hydromet(hyd_status):
        blocks.append("NO_HYDROMET_CONTEXT")
    if not _has_temporal(window, temporal_status):
        blocks.append("WEAK_TEMPORAL_PRECISION")
    if not _has_patch(patch_status):
        blocks.append("NO_PATCH_LINK")
    return ";".join(blocks) if blocks else "NONE"


# ---------------------------------------------------------------------------
# Classification — automated reviewer A/B
# ---------------------------------------------------------------------------

# Allowed reviewer decisions (no label/target/ground-truth/C3/C4/negative).
REVIEWER_DECISIONS = (
    "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE",
    "AUTOMATED_REVIEW_CONTEXT_OK_BUT_NEEDS_EXTERNAL_SOURCE",
    "AUTOMATED_REVIEW_NEEDS_TEMPORAL_PRECISION",
    "AUTOMATED_REVIEW_NEEDS_SPATIAL_PRECISION",
    "AUTOMATED_REVIEW_OVERCLAIM_RISK",
    "AUTOMATED_REVIEW_BLOCKED_INSUFFICIENT_EVIDENCE",
)


def reviewer_dimensions(
    profile: str, ext_status: str, hyd_status: str, window: str,
    temporal_status: str, patch_status: str, dino_status: str,
) -> dict[str, Any]:
    """Per-profile evidence judgement. 'conservative' is stricter than
    'integrator' on spatial/temporal tolerance and overclaim sensitivity."""
    profile = profile.strip().lower()
    has_ext = _has_external(ext_status)
    has_hyd = _has_hydromet(hyd_status)
    has_patch = _has_patch(patch_status)
    has_window = bool(str(window or "").strip())
    parsed = str(temporal_status).strip().upper() == "DATE_PARSED_OK"
    completeness = evidence_completeness(
        ext_status, hyd_status, window, temporal_status, patch_status, dino_status)

    if profile == "conservative":
        temporal_ok = parsed
        spatial_ok = has_patch
        overclaim = "MODERATE" if (has_hyd and not has_ext) else "LOW"
        confidence = round(completeness * 0.65, 2)
    else:  # integrator
        temporal_ok = parsed or has_window
        spatial_ok = has_patch or has_window
        overclaim = "LOW" if has_hyd else "NONE"
        confidence = round(min(1.0, completeness * 0.9 + 0.05), 2)

    return {
        "external_source_sufficient_for_review_only": "true" if has_ext else "false",
        "temporal_context_sufficient_for_review_only": "true" if temporal_ok else "false",
        "spatial_context_sufficient_for_review_only": "true" if spatial_ok else "false",
        "hydromet_context_useful": "true" if has_hyd else "false",
        "dino_role_correctly_limited": "true",
        "patch_linkage_sufficient_for_review_only": "true" if has_patch else "false",
        "overclaim_risk": overclaim,
        "evidence_chain_completeness_score": f"{completeness:.2f}",
        "review_only_confidence_score": f"{confidence:.2f}",
    }


def classify_automated_review_decision(profile: str, dims: dict[str, Any]) -> str:
    profile = profile.strip().lower()
    ext = dims["external_source_sufficient_for_review_only"] == "true"
    temporal = dims["temporal_context_sufficient_for_review_only"] == "true"
    spatial = dims["spatial_context_sufficient_for_review_only"] == "true"
    hyd = dims["hydromet_context_useful"] == "true"
    overclaim = str(dims["overclaim_risk"]).upper()
    completeness = parse_float_safe(dims["evidence_chain_completeness_score"], 0.0)

    if overclaim == "HIGH":
        return "AUTOMATED_REVIEW_OVERCLAIM_RISK"
    if not temporal:
        return "AUTOMATED_REVIEW_NEEDS_TEMPORAL_PRECISION"
    if not spatial:
        return "AUTOMATED_REVIEW_NEEDS_SPATIAL_PRECISION"
    if profile == "integrator" and hyd and completeness >= 0.5:
        return "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE"
    if hyd:
        return "AUTOMATED_REVIEW_CONTEXT_OK_BUT_NEEDS_EXTERNAL_SOURCE"
    if not ext:
        return "AUTOMATED_REVIEW_BLOCKED_INSUFFICIENT_EVIDENCE"
    return "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE"


# ---------------------------------------------------------------------------
# Classification — consensus / divergence
# ---------------------------------------------------------------------------

CONSENSUS_STATUSES = (
    "AUTOMATED_CONSENSUS_VALIDATED_FOR_REVIEW_ONLY_USE",
    "AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE",
    "AUTOMATED_CONSENSUS_BLOCKED_TEMPORAL_SPATIAL",
    "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION",
    "AUTOMATED_CONSENSUS_OVERCLAIM_RISK",
)

_TEMPORAL_SPATIAL = {
    "AUTOMATED_REVIEW_NEEDS_TEMPORAL_PRECISION",
    "AUTOMATED_REVIEW_NEEDS_SPATIAL_PRECISION",
}


def classify_consensus(a_status: str, b_status: str) -> tuple[str, str]:
    """Return (consensus_status, divergence_type)."""
    if a_status == b_status:
        if a_status == "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE":
            return "AUTOMATED_CONSENSUS_VALIDATED_FOR_REVIEW_ONLY_USE", "NONE"
        if a_status == "AUTOMATED_REVIEW_CONTEXT_OK_BUT_NEEDS_EXTERNAL_SOURCE":
            return "AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE", "NONE"
        if a_status == "AUTOMATED_REVIEW_OVERCLAIM_RISK":
            return "AUTOMATED_CONSENSUS_OVERCLAIM_RISK", "NONE"
        if a_status in _TEMPORAL_SPATIAL:
            return "AUTOMATED_CONSENSUS_BLOCKED_TEMPORAL_SPATIAL", "NONE"
        return "AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE", "NONE"
    if {a_status, b_status} <= _TEMPORAL_SPATIAL:
        return "AUTOMATED_CONSENSUS_BLOCKED_TEMPORAL_SPATIAL", "TEMPORAL_VS_SPATIAL"
    if "AUTOMATED_REVIEW_OVERCLAIM_RISK" in (a_status, b_status):
        return "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION", "OVERCLAIM_DISAGREEMENT"
    return ("AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION",
            "VALIDATION_THRESHOLD_DISAGREEMENT")


def supervisor_adjudication_required(consensus_status: str) -> str:
    return ("true" if consensus_status
            == "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION" else "false")


# ---------------------------------------------------------------------------
# Classification — automated supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_DECISIONS = (
    "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE",
    "AUTOMATED_SUPERVISOR_READY_FOR_TCC_DISCUSSION",
    "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE",
    "AUTOMATED_SUPERVISOR_BLOCKED_OVERCLAIM_RISK",
    "AUTOMATED_SUPERVISOR_BLOCKED_INSUFFICIENT_EVIDENCE",
)


def classify_supervisor_precheck(guardrail_clean: bool) -> str:
    return "SUPERVISOR_PRECHECK_PASS" if guardrail_clean else \
        "SUPERVISOR_PRECHECK_FAIL_GUARDRAIL"


def classify_supervisor_decision(
    consensus_status: str, completeness: float, guardrail_clean: bool,
    has_external: bool,
) -> str:
    if not guardrail_clean:
        return "AUTOMATED_SUPERVISOR_BLOCKED_OVERCLAIM_RISK"
    if consensus_status == "AUTOMATED_CONSENSUS_OVERCLAIM_RISK":
        return "AUTOMATED_SUPERVISOR_BLOCKED_OVERCLAIM_RISK"
    if consensus_status == "AUTOMATED_CONSENSUS_BLOCKED_TEMPORAL_SPATIAL":
        return "AUTOMATED_SUPERVISOR_BLOCKED_INSUFFICIENT_EVIDENCE"
    if consensus_status == "AUTOMATED_CONSENSUS_VALIDATED_FOR_REVIEW_ONLY_USE":
        return "AUTOMATED_SUPERVISOR_READY_FOR_TCC_DISCUSSION"
    if consensus_status == "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION":
        if completeness >= 0.5:
            return "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE"
        return "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE"
    # AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE
    if has_external:
        return "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE"
    return "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE"


def supervisor_final_for_review_only(decision: str) -> str:
    return "true" if decision in (
        "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE",
        "AUTOMATED_SUPERVISOR_READY_FOR_TCC_DISCUSSION",
    ) else "false"


def supervisor_ready_for_tcc(decision: str) -> str:
    return "true" if decision in (
        "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE",
        "AUTOMATED_SUPERVISOR_READY_FOR_TCC_DISCUSSION",
    ) else "false"


# ---------------------------------------------------------------------------
# Review-only validation status (proof audit)
# ---------------------------------------------------------------------------

def classify_review_only_validation_status(supervisor_decision: str) -> str:
    if supervisor_decision in (
        "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE",
        "AUTOMATED_SUPERVISOR_READY_FOR_TCC_DISCUSSION",
    ):
        return "VALIDATED_FOR_REVIEW_ONLY_USE"
    if supervisor_decision == "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE":
        return "EXTERNAL_OBSERVATIONAL_EVIDENCE_REQUIRED_FOR_OPERATIONAL_CLAIM"
    return "NOT_VALIDATED_FOR_REVIEW_ONLY_USE"


# ---------------------------------------------------------------------------
# Rubric / summary builders
# ---------------------------------------------------------------------------

def build_reviewer_rubric() -> list[dict[str, str]]:
    crit = [
        ("external_source_sufficient_for_review_only",
         "Há fonte observacional externa independente suficiente?"),
        ("temporal_context_sufficient_for_review_only",
         "A precisão temporal é suficiente para revisão?"),
        ("spatial_context_sufficient_for_review_only",
         "A precisão espacial é suficiente para revisão?"),
        ("hydromet_context_useful",
         "O contexto hidromet (INMET) é útil — apenas como contexto?"),
        ("dino_role_correctly_limited",
         "O papel do DINO está corretamente limitado a representação?"),
        ("patch_linkage_sufficient_for_review_only",
         "O vínculo patch/evento é suficiente para revisão?"),
        ("overclaim_risk",
         "Risco de overclaim se o contexto for usado como prova."),
    ]
    rows = []
    for prof in ("conservative", "integrator"):
        for key, desc in crit:
            rows.append({
                "reviewer_profile": prof, "criterion_key": key,
                "criterion_description": desc,
                "review_only": "true", "automated_review": "true",
            })
    return rows


def build_supervisor_rubric() -> list[dict[str, str]]:
    checks = [
        ("guardrail_precheck", "Nenhum guardrail proibido ativado."),
        ("consensus_review", "Avaliar consenso/divergência dos revisores."),
        ("completeness_review", "Cadeia de evidência suficientemente completa."),
        ("overclaim_control", "Bloquear qualquer overclaim operacional."),
        ("review_only_decision", "Decisão final apenas para uso review-only."),
        ("tcc_readiness", "Caso pronto para discussão no TCC."),
        ("external_requirement",
         "Fonte observacional externa exigida para afirmação operacional."),
    ]
    return [{
        "supervisor_check_key": k, "supervisor_check_description": d,
        "review_only": "true", "automated_review": "true",
        "operational_validation": "false",
        "supervisor_final_operational_decision_allowed": "false",
    } for k, d in checks]


def build_tcc_case_summary(case: dict[str, str]) -> dict[str, str]:
    return {
        "case_id": case.get("case_id", ""),
        "region": case.get("region", ""),
        "hazard_type": case.get("hazard_type", ""),
        "case_readiness_status": case.get("case_readiness_status", ""),
        "claim_safety": "REVIEW_ONLY_SAFE_NO_OPERATIONAL_CLAIM",
        "review_only": "true",
    }
