"""REV-P v2cj - conservative TP2 candidate prioritization."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cj_to_v2cm_common import (
    ALLOWED_CLAIM,
    FORBIDDEN_CLAIM,
    boolish,
    inventory_path,
    pair_path,
    priority_path,
    read_csv,
    write_csv,
    write_text,
)


FIELDS = [
    "rank",
    "candidate_id",
    "region",
    "event_name",
    "source_name",
    "evidence_type",
    "review_priority_score",
    "priority_class",
    "main_strength",
    "main_blocker",
    "recommended_next_action",
    "allowed_claim",
    "forbidden_claim",
    "tp2_status",
    "tp3_ready",
]


def score_candidate(row: dict[str, str], pair: dict[str, str] | None) -> tuple[int, str, str, str]:
    score = 10
    strengths: list[str] = []
    blockers: list[str] = []
    if row.get("source_name") and row.get("source_name") != "UNKNOWN":
        score += 8
        strengths.append("fonte_identificavel")
    else:
        score -= 8
        blockers.append("fonte_ambigua")
    if row.get("event_date"):
        score += 8
        strengths.append("data_explicita")
    else:
        score -= 4
        blockers.append("data_ausente")
    evidence = row.get("evidence_type", "")
    if evidence == "GEOMETRIA_CANDIDATA":
        score += 20
        strengths.append("geometria_candidata")
    elif evidence in {"EVIDENCIA_VISUAL", "EVIDENCIA_DOCUMENTAL"}:
        score += 10
        strengths.append(evidence.lower())
    else:
        score -= 8
        blockers.append("evidencia_textual_ou_fraca")
    if boolish(row.get("can_be_digitized")):
        score += 10
        strengths.append("digitalizacao_possivel")
    if boolish(row.get("has_observed_geometry")):
        score += 15
    else:
        score -= 15
        blockers.append("sem_geometria_observada_validada")
    for key, bonus, label in [
        ("crs_known", 8, "crs"),
        ("provenance_available", 8, "proveniencia"),
        ("hash_available", 8, "hash"),
    ]:
        if boolish(row.get(key)):
            score += bonus
            strengths.append(f"{label}_disponivel")
        else:
            score -= bonus
            blockers.append(f"{label}_ausente")
    if pair:
        score += 8
        strengths.append("par_patch_evento_existente")
        if boolish(pair.get("intersection_test_possible")):
            score += 8
        else:
            score -= 8
            blockers.append("replay_indisponivel")
    else:
        score -= 8
        blockers.append("sem_par_patch_evento")
    score = max(0, min(100, score))
    if score >= 70:
        klass = "HIGH_REVIEW_PRIORITY"
    elif score >= 45:
        klass = "MEDIUM_REVIEW_PRIORITY"
    elif score >= 20:
        klass = "LOW_REVIEW_PRIORITY"
    else:
        klass = "BLOCKED_FOR_REVIEW"
    return score, klass, strengths[0] if strengths else "sem_forca_dominante", blockers[0] if blockers else "sem_bloqueio_dominante"


def build_priority(repo_root: Path) -> list[dict[str, str]]:
    inv = read_csv(inventory_path(repo_root))
    pairs = {row.get("candidate_id", ""): row for row in read_csv(pair_path(repo_root))}
    rows: list[dict[str, str]] = []
    for item in inv:
        pair = pairs.get(item.get("candidate_id", ""))
        score, klass, strength, blocker = score_candidate(item, pair)
        rows.append(
            {
                "candidate_id": item.get("candidate_id", ""),
                "region": item.get("region", ""),
                "event_name": item.get("event_name", ""),
                "source_name": item.get("source_name", ""),
                "evidence_type": item.get("evidence_type", ""),
                "review_priority_score": str(score),
                "priority_class": klass,
                "main_strength": strength,
                "main_blocker": blocker,
                "recommended_next_action": "preparar_pacote_digitizacao" if klass != "BLOCKED_FOR_REVIEW" else "manter_bloqueado_ate_nova_evidencia",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
                "tp2_status": item.get("candidate_status", "TP2_BLOCKED"),
                "tp3_ready": pair.get("tp3_ready", "false") if pair else "false",
            }
        )
    rows.sort(key=lambda r: (-int(r["review_priority_score"]), r["candidate_id"]))
    for idx, row in enumerate(rows, 1):
        row["rank"] = str(idx)
    return rows


def build_report(rows: list[dict[str, str]]) -> str:
    top = rows[0] if rows else {}
    return f"""# REV-P v2cj - priorizacao conservadora de candidatos TP2

Este marco gera prioridade de revisao, nao prioridade de verdade. Mesmo a classe
`HIGH_REVIEW_PRIORITY` nao fecha TP2 e nao cria ground truth operacional.

Total de candidatos priorizados: {len(rows)}.
Candidato no topo: `{top.get('candidate_id', 'NA')}`.
Maior bloqueio dominante: `{top.get('main_blocker', 'NA')}`.

Guardrails: sem label, sem negativo formal, sem treino, sem claim de deteccao,
sem claim de predicao e sem intersecao observada afirmada.
"""


def run(repo_root: Path, force: bool = False) -> int:
    rows = build_priority(repo_root)
    out = priority_path(repo_root)
    if out.exists() and not force:
        raise FileExistsError(out)
    write_csv(out, rows, FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_tp2_candidate_priority_report_v2cj.md", build_report(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run(Path(args.repo_root), args.force)


if __name__ == "__main__":
    raise SystemExit(main())

