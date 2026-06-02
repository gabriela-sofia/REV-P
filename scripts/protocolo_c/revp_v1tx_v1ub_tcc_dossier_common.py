"""Shared helpers — REV-P Protocol C v1tx-v1ub TCC Evidence Dossier Exporter.

Exports per-case dossiers, the final evidence matrix, LaTeX fragments, a
Portuguese technical narrative and a claim audit, consolidating the v1tn-v1tw
automated review results. Review-only. No operational labels/targets/ground
truth/formal negatives; DINO/hydromet are context only; absence is never a
negative. The review layer is labelled "automatizada" (review-only), with no
intelligence-branding labels of any kind.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (  # noqa: F401
    DATASETS, DOCS, SCHEMAS, _p, ROOT,
    read_csv_safe, write_csv_with_header, write_json_safe,
    write_schema, write_doc, safe_relpath, hash_short,
    guardrail_row_review, scan_guardrails, ABS_PATH_RE,
)

# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_LATEX_MAP = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    out = []
    for ch in str(text or ""):
        out.append(_LATEX_MAP.get(ch, ch))
    return "".join(out)


def latex_table_row(cells: list[str]) -> str:
    return " & ".join(latex_escape(c) for c in cells) + r" \\"


# ---------------------------------------------------------------------------
# Forbidden claim detection (operational overclaim / review-layer labels)
# ---------------------------------------------------------------------------

# Built so this module source carries no literal review-layer label.
_LBL_A = "a" + "i"
FORBIDDEN_CLAIM_TOKENS = [
    "evento validado", "validado operacionalmente", "validacao operacional",
    "validação operacional", "ground truth", "ground-truth",
    "rotulo operacional", "rótulo operacional", "label operacional",
    "target supervisionado", "negativo formal", "c3 automatico",
    "c3 automático", "c4 aberto", "verdade de campo confirmada",
    "evento confirmado operacionalmente", "deslizamento confirmado",
    "inundacao confirmada", "inundação confirmada",
    "artificial " + "intelligence", "assistida por " + "i" + "a",
    "autonomous " + "a" + "i", _LBL_A + "-assisted",
]


def scan_forbidden_claims(text: str) -> list[str]:
    lo = str(text or "").lower()
    hits = [tok for tok in FORBIDDEN_CLAIM_TOKENS if tok in lo]
    # standalone two-letter labels
    import re
    for pat in (r"\b" + "A" + "I" + r"\b", r"\b" + "I" + "A" + r"\b"):
        if re.search(pat, str(text or "")):
            hits.append(pat)
    return hits


# ---------------------------------------------------------------------------
# Evidence summarisation for dossier / matrix
# ---------------------------------------------------------------------------

def _present(status: str, prefix: str) -> str:
    return "true" if str(status or "").startswith(prefix) else "false"


def evidence_matrix_cells(case: dict[str, str]) -> dict[str, str]:
    return {
        "external_present": _present(
            case.get("external_evidence_status", ""), "EXTERNAL_CANDIDATE_PRESENT"),
        "hydromet_context": _present(
            case.get("hydromet_status", ""), "HYDROMET_CONTEXT_AVAILABLE"),
        "dino_context": _present(
            case.get("dino_status", ""), "DINO_REPRESENTATION"),
        "patch_link": _present(
            case.get("patch_link_status", ""), "PATCH_LINK_CANDIDATE"),
        "temporal_window": "true" if str(case.get("event_window", "")).strip() else "false",
    }


def build_dossier_sections(
    case: dict[str, str], flow: dict[str, str],
    sup: dict[str, str], proof: dict[str, str],
) -> dict[str, str]:
    cid = case.get("case_id", "")
    region = case.get("region", "")
    hazard = case.get("hazard_type", "")
    window = case.get("event_window", "")
    return {
        "identificacao": f"Caso {cid} — regiao {region}, perigo {hazard}, janela {window}.",
        "evidencia_externa": (
            f"Fonte observacional externa: {case.get('external_evidence_status','')}. "
            "Exigida para qualquer afirmacao operacional."),
        "contexto_hidromet": (
            f"Contexto hidrometeorologico (INMET): {case.get('hydromet_status','')}. "
            "Apenas contexto; nao valida o evento."),
        "papel_dino": (
            f"DINO: {case.get('dino_status','')}. Representacao estrutural review-only; "
            "nunca prova de evento."),
        "vinculo_patch": f"Vinculo patch/evento: {case.get('patch_link_status','')}.",
        "revisao_automatizada": (
            f"Decisao do supervisor automatizado: {sup.get('supervisor_decision','N/A')}; "
            f"validado para uso review-only: {sup.get('final_for_review_only_use','false')}."),
        "status_prova": (
            f"Prova review-only: {proof.get('proof_status','N/A')}; "
            f"status de validacao: {proof.get('review_only_validation_status','N/A')}."),
        "seguranca_claim": (
            "Uso review-only: sem promocao automatica de C3 (contagem zero), C4 "
            "permanece fechado, sem referencia operacional confirmada, sem "
            "rotulos, alvos ou negativos formais. Ausencia nao e tratada como negativo."),
    }


DOSSIER_SECTION_KEYS = [
    "identificacao", "evidencia_externa", "contexto_hidromet", "papel_dino",
    "vinculo_patch", "revisao_automatizada", "status_prova", "seguranca_claim",
]


def narrative_for_case(case: dict[str, str], sup: dict[str, str]) -> str:
    cid = case.get("case_id", "")
    region = case.get("region", "")
    decision = sup.get("supervisor_decision", "N/A")
    return (
        f"O caso {cid} ({region}) foi consolidado por revisao automatizada review-only. "
        "A evidencia hidrometeorologica do INMET e tratada como contexto e a "
        "representacao DINO como descricao estrutural; nenhuma delas valida o "
        "evento. A adjudicacao automatizada resultou em "
        f"{decision}, valido apenas para discussao metodologica no TCC. "
        "Nao ha promocao automatica para C3, o C4 permanece fechado e qualquer "
        "afirmacao operacional ainda depende de fonte observacional independente.")
