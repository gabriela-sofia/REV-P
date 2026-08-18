"""REV-P Protocolo C v1jp -- busca e intake de evidencia negativa formal.

Reconstruido em 2026-08-18 a partir do contrato de teste
`tests/test_revp_v1jp_formal_negative_evidence_discovery_intake.py`, apos o
script original ter sido encontrado sem nunca ter sido versionado no git
(apenas o teste, a documentacao e a saida sanitizada estavam commitados).

Nota de proveniencia: a rodada historica registrada em
`datasets/formal_negative_evidence_intake_registry.csv` (175.824 linhas,
02/06/2026) escaneou majoritariamente um corpus de `local_runs/` -- por
convencao do projeto, essa pasta nunca e versionada. Rodar este script hoje
NAO reproduz esse volume linha a linha, porque o corpus bruto correspondente
nao esta e nunca esteve no repositorio publico. O que este script faz e
documentar e reexecutar o METODO com fidelidade ao comportamento original
(padroes buscados, gates, classificacao), sobre qualquer corpus textual
disponivel em `datasets/` e `docs/metodologia_cientifica/` no momento da
execucao. O registro historico permanece como o resultado real e conferido
(contagens batem com o relatorio v1jp: 0 prontos, 5 em revisao, 175.814
hipoteses invalidas, 5 com evidencia insuficiente).

Regra central do projeto (nunca relaxar): evidencia negativa formal exige
fonte oficial/auditavel, localidade ou coordenada, janela temporal, fenomeno
compativel e declaracao explicita de ausencia, estabilidade ou vistoria sem
ocorrencia. Ausencia de registro, distancia de um anchor positivo, baixo
risco generico, suscetibilidade baixa, pseudo-ausencia, background e
contexto pre-evento NAO bastam -- podem orientar revisao, mas nunca criam
negativo formal nem label de treino.

Guarda de seguranca: se `formal_negative_evidence_intake_registry.csv` ja
existir e for maior que um stub de teste (10 KB), este script NUNCA o
sobrescreve, mesmo com --force -- inclusive quando chamado pela suite de
testes. O resultado historico real (175.824 linhas) so seria substituido
apagando o arquivo manualmente primeiro, decisao que deve ser sempre humana
e deliberada.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REVP_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

REGISTRY_OUT = DATASETS / "formal_negative_evidence_intake_registry.csv"
DECISION_OUT = DATASETS / "formal_negative_candidate_decision_audit.csv"
MATRIX_OUT = DATASETS / "c4_negative_evidence_intake_update_matrix.csv"

# Padroes que, se explicitos, PODEM indicar evidencia negativa formal --
# ainda sujeitos aos 4 gates abaixo antes de qualquer promocao.
VALID_NEGATIVE_PATTERNS = [
    r"sem ind[ií]cios", r"sem evid[eê]ncias?", r"n[aã]o foi observado deslizamento",
    r"aus[eê]ncia de instabilidade", r"[aá]rea est[aá]vel", r"sem movimenta[cç][aã]o",
    r"sem ocorr[eê]ncia", r"vistoria sem ocorr[eê]ncia", r"ponto controle",
    r"[aá]rea controle", r"estabilidade observada", r"sem fei[cç][oõ]es",
]

# Padroes que NUNCA viram negativo formal, mesmo com linguagem de ausencia --
# regra fixa do projeto (nao afrouxar).
INVALID_ASSUMPTION_PATTERNS = [
    r"sem registro", r"aus[eê]ncia de registro", r"baixo risco",
    r"suscetibilidade baixa", r"pseudo[- ]aus[eê]ncia", r"\bpseudo\b",
    r"\bbackground\b", r"contexto pr[eé][- ]evento",
]

PHENOMENON_TERMS = {
    "alagamento": "FLOODING", "enchente": "FLOODING", "inunda": "FLOODING",
    "deslizamento": "MOVEMENT_OF_MASS", "escorregamento": "MOVEMENT_OF_MASS",
    "movimento de massa": "MOVEMENT_OF_MASS", "encosta": "MOVEMENT_OF_MASS",
    "barreira": "MOVEMENT_OF_MASS",
}

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
COORD_RE = re.compile(r"-?\d{1,3}\.\d{3,}")

SCAN_GLOBS = ["*.md", "*.csv", "*.json"]
SCAN_ROOTS = [DATASETS, DOCS]


@dataclass
class Candidate:
    source_document: str
    source_registry: str
    phrase: str
    is_valid_pattern: bool
    is_invalid_assumption: bool
    year: str | None
    coordinate: bool
    phenomenon_group: str | None
    region_hint: str | None


def iter_scan_targets() -> list[Path]:
    targets: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for pattern in SCAN_GLOBS:
            targets.extend(sorted(root.rglob(pattern)))
    return targets


def sanitize(text: str) -> str:
    # Nunca deixar caminho absoluto local vazar para o artefato publico.
    text = re.sub(r"[A-Za-z]:\\[^\s\"']+", "[PATH_SANITIZED]", text)
    text = re.sub(r"/home/[^\s\"']+", "[PATH_SANITIZED]", text)
    return text.replace("\n", " ").strip()


def find_candidates(path: Path) -> list[Candidate]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    lowered = text.lower()
    found: list[Candidate] = []
    for pattern in VALID_NEGATIVE_PATTERNS + INVALID_ASSUMPTION_PATTERNS:
        for match in re.finditer(pattern, lowered):
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            window = sanitize(text[start:end])
            found.append(
                Candidate(
                    source_document=path.name,
                    source_registry=f"DOC:{path.name}",
                    phrase=window,
                    is_valid_pattern=pattern in VALID_NEGATIVE_PATTERNS,
                    is_invalid_assumption=pattern in INVALID_ASSUMPTION_PATTERNS,
                    year=(YEAR_RE.search(window).group(0) if YEAR_RE.search(window) else None),
                    coordinate=bool(COORD_RE.search(window)),
                    phenomenon_group=next(
                        (v for k, v in PHENOMENON_TERMS.items() if k in window.lower()), None
                    ),
                    region_hint=next(
                        (r for r in ("recife", "curitiba", "petrópolis", "petropolis") if r in window.lower()),
                        None,
                    ),
                )
            )
    return found


def classify(candidate: Candidate) -> dict:
    explicit_absence = "PASS" if candidate.is_valid_pattern and not candidate.is_invalid_assumption else "FAIL"
    temporal = "PASS" if candidate.year else "FAIL_OR_MISSING"
    spatial = "PASS_REVIEW_REQUIRED" if candidate.region_hint and not candidate.coordinate else (
        "PASS" if candidate.coordinate else "FAIL_OR_MISSING"
    )
    phenomenon = "PASS" if candidate.phenomenon_group else "FAIL_OR_MISSING"
    patch_linkage_possible = bool(candidate.region_hint)

    if candidate.is_invalid_assumption:
        status = "INSUFFICIENT_EVIDENCE" if not candidate.is_valid_pattern and candidate.phenomenon_group else "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"
        can_be_formal = False
    elif all(g == "PASS" for g in (explicit_absence, temporal, phenomenon)) and spatial.startswith("PASS") and patch_linkage_possible:
        status = "FORMAL_NEGATIVE_READY"
        can_be_formal = True
    elif explicit_absence == "PASS":
        status = "FORMAL_NEGATIVE_CANDIDATE_REVIEW"
        can_be_formal = False
    else:
        status = "INSUFFICIENT_EVIDENCE"
        can_be_formal = False

    return {
        "explicit_absence_gate_status": explicit_absence,
        "temporal_gate_status": temporal,
        "spatial_gate_status": spatial,
        "phenomenon_gate_status": phenomenon,
        "patch_linkage_possible": str(patch_linkage_possible).lower(),
        "leakage_risk_status": "LEAKAGE_REVIEW_REQUIRED",
        "negative_evidence_status": status,
        "can_be_formal_negative": str(can_be_formal).lower(),
    }


def write_registry(rows: list[dict]) -> None:
    fieldnames = [
        "negative_evidence_id", "source_document_sanitized", "source_registry", "source_institution",
        "region", "municipality", "locality_text_sanitized", "coordinate_status", "latitude", "longitude",
        "temporal_expression", "temporal_precision", "phenomenon_group", "negative_evidence_phrase_sanitized",
        "absence_or_stability_type", "source_confidence", "spatial_precision", "temporal_gate_status",
        "spatial_gate_status", "phenomenon_gate_status", "explicit_absence_gate_status", "patch_linkage_possible",
        "leakage_risk_status", "negative_evidence_status", "can_be_formal_negative", "can_create_negative_label",
        "can_create_training_label", "can_train_model", "blocking_reason", "minimum_evidence_needed", "notes",
    ]
    REGISTRY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_decision_audit(rows: list[dict]) -> None:
    fieldnames = [
        "decision_id", "negative_evidence_id", "candidate_status", "gates_passed_count", "gates_failed_count",
        "failed_gates", "nearest_positive_anchor_distance_m", "same_locality_as_positive",
        "same_temporal_window_as_positive", "can_pair_with_positive_c3", "can_unlock_c4", "allowed_use",
        "forbidden_use", "decision", "notes",
    ]
    with DECISION_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_update_matrix(counts: dict) -> None:
    ready = counts.get("FORMAL_NEGATIVE_READY", 0)
    fieldnames = [
        "update_id", "formal_negative_ready_count", "formal_negative_candidate_review_count",
        "invalid_negative_absence_assumption_count", "insufficient_evidence_count", "c4_ready_after_intake",
        "summary_decision", "supervised_training_status_after_intake", "notes",
    ]
    row = {
        "update_id": "C4_NEGATIVE_EVIDENCE_INTAKE_UPDATE_V1JP",
        "formal_negative_ready_count": str(ready),
        "formal_negative_candidate_review_count": str(counts.get("FORMAL_NEGATIVE_CANDIDATE_REVIEW", 0)),
        "invalid_negative_absence_assumption_count": str(counts.get("INVALID_NEGATIVE_ABSENCE_ASSUMPTION", 0)),
        "insufficient_evidence_count": str(counts.get("INSUFFICIENT_EVIDENCE", 0)),
        "c4_ready_after_intake": str(ready > 0).lower(),
        "summary_decision": (
            "NEGATIVE_INTAKE_NO_FORMAL_NEGATIVES_FOUND;C4_STILL_BLOCKED" if ready == 0
            else "NEGATIVE_INTAKE_FORMAL_NEGATIVES_FOUND;C4_REVIEW_REQUIRED"
        ),
        "supervised_training_status_after_intake": (
            "SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVE_LABELS" if ready == 0
            else "SUPERVISED_TRAINING_STILL_BLOCKED_PENDING_MANUAL_REVIEW"
        ),
        "notes": "v1jp nao altera v1jn/v1jo e nao cria label; promocao para C4 exige revisao manual.",
    }
    with MATRIX_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


# Abaixo deste tamanho, um registro existente e tratado como stub/teste e pode
# ser regenerado com --force. Acima disso, o arquivo e tratado como resultado
# historico real e NUNCA e sobrescrito automaticamente -- mesmo com --force --
# porque o corpus bruto que originou a rodada historica (largamente vindo de
# local_runs/, nunca versionado) nao esta mais disponivel, e uma rodada fresca
# produziria uma contagem muito menor e destruiria evidencia real sem
# possibilidade de recuperacao. Para regenerar de fato, apague o arquivo
# manualmente primeiro -- isso deve ser uma decisao humana deliberada, nunca
# um efeito colateral de rodar a suite de testes.
HISTORICAL_PRESERVE_THRESHOLD_BYTES = 10_000


def should_regenerate(force: bool) -> bool:
    if not REGISTRY_OUT.exists():
        return True
    if REGISTRY_OUT.stat().st_size > HISTORICAL_PRESERVE_THRESHOLD_BYTES:
        return False
    return force


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--read-c4-queues", action="store_true")
    parser.add_argument("--scan-local-documents", action="store_true")
    parser.add_argument("--scan-registries", action="store_true")
    parser.add_argument("--extract-negative-evidence-candidates", action="store_true")
    parser.add_argument("--classify-negative-evidence", action="store_true")
    parser.add_argument("--emit-intake", action="store_true")
    args = parser.parse_args()

    if should_regenerate(args.force):
        candidates: list[Candidate] = []
        for path in iter_scan_targets():
            candidates.extend(find_candidates(path))

        registry_rows = []
        decision_rows = []
        counts: dict[str, int] = {}
        for idx, candidate in enumerate(candidates, start=1):
            neg_id = f"NEG_INTAKE_V1JP_{idx:04d}"
            classification = classify(candidate)
            counts[classification["negative_evidence_status"]] = counts.get(
                classification["negative_evidence_status"], 0
            ) + 1
            registry_rows.append(
                {
                    "negative_evidence_id": neg_id,
                    "source_document_sanitized": candidate.source_document,
                    "source_registry": candidate.source_registry,
                    "source_institution": "UNKNOWN_NOT_OFFICIAL_EVIDENCE",
                    "region": (candidate.region_hint or "UNKNOWN").upper()[:3],
                    "municipality": candidate.region_hint or "",
                    "locality_text_sanitized": "",
                    "coordinate_status": "PRESENT" if candidate.coordinate else "NOT_DOCUMENTED",
                    "latitude": "",
                    "longitude": "",
                    "temporal_expression": candidate.year or "",
                    "temporal_precision": "YEAR_ONLY" if candidate.year else "NOT_DOCUMENTED",
                    "phenomenon_group": candidate.phenomenon_group or "NOT_DOCUMENTED",
                    "negative_evidence_phrase_sanitized": candidate.phrase,
                    "absence_or_stability_type": "NO_OCCURRENCE" if candidate.is_valid_pattern else "INVALID_ASSUMPTION_TERM",
                    "source_confidence": "REGISTRY_OR_CONTEXT_ONLY",
                    "spatial_precision": "NOT_DOCUMENTED",
                    **classification,
                    "can_create_negative_label": "false",
                    "can_create_training_label": "false",
                    "can_train_model": "false",
                    "blocking_reason": (
                        "INVALID_NEGATIVE_ASSUMPTION" if classification["negative_evidence_status"] == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"
                        else "NEGATIVE_EVIDENCE_REVIEW_REQUIRED"
                    ),
                    "minimum_evidence_needed": (
                        "explicit official absence/stability evidence; not absence of record, distance, "
                        "pseudo-absence or background"
                    ),
                    "notes": "Discovery/intake only; no label or C4 promotion is made by v1jp.",
                }
            )
            decision_rows.append(
                {
                    "decision_id": f"DECISION_{neg_id}",
                    "negative_evidence_id": neg_id,
                    "candidate_status": classification["negative_evidence_status"],
                    "gates_passed_count": sum(
                        1 for k in ("explicit_absence_gate_status", "temporal_gate_status", "spatial_gate_status", "phenomenon_gate_status")
                        if classification[k].startswith("PASS")
                    ),
                    "gates_failed_count": sum(
                        1 for k in ("explicit_absence_gate_status", "temporal_gate_status", "spatial_gate_status", "phenomenon_gate_status")
                        if not classification[k].startswith("PASS")
                    ),
                    "failed_gates": ";".join(
                        k for k in ("explicit_absence_gate_status", "temporal_gate_status", "spatial_gate_status", "phenomenon_gate_status")
                        if not classification[k].startswith("PASS")
                    ),
                    "nearest_positive_anchor_distance_m": "NOT_COMPUTED_METADATA_ONLY",
                    "same_locality_as_positive": "UNKNOWN",
                    "same_temporal_window_as_positive": "UNKNOWN_REVIEW_REQUIRED" if candidate.region_hint else "UNKNOWN",
                    "can_pair_with_positive_c3": str(bool(candidate.region_hint)).lower(),
                    "can_unlock_c4": "false",
                    "allowed_use": "AUDIT_LOG_ONLY" if classification["negative_evidence_status"] == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION" else "FORMAL_NEGATIVE_REVIEW_INTAKE",
                    "forbidden_use": "NEGATIVE_LABEL;TRAINING_LABEL;SUPERVISED_TRAINING;PSEUDO_ABSENCE_PROMOTION;OPERATIONAL_CLAIM",
                    "decision": "REVIEW_ONLY_DO_NOT_LABEL",
                    "notes": "Distance and leakage must be computed only after evidence passes documentary gates.",
                }
            )

        write_registry(registry_rows)
        write_decision_audit(decision_rows)
        write_update_matrix(counts)
        print(f"v1jp: intake escrito em {REGISTRY_OUT} ({len(registry_rows)} linhas)")
    else:
        print(
            f"v1jp: {REGISTRY_OUT.name} ja existe como resultado historico real "
            f"({REGISTRY_OUT.stat().st_size} bytes) -- preservado, nao sobrescrito. "
            "Apague o arquivo manualmente antes de rodar de novo se a intencao "
            "for de fato regenerar."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
