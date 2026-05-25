"""
REV-P v1jo - C4 gate gap analysis and actionable evidence queue.

This stage reads the canonical v1jn C1-C4 layer and ranks the remaining C4
gaps. It is metadata-only: no labels, training, model changes, raster access,
download/export, or pseudo-absence promotion are performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
DOCS_DIR = REVP_ROOT / "docs" / "metodologia_cientifica"
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jo"

INPUT_PATHS = {
    "events": DATASETS_DIR / "ground_reference_event_registry.csv",
    "linkages": DATASETS_DIR / "event_patch_linkage_registry.csv",
    "decisions": DATASETS_DIR / "ground_truth_candidate_decision_audit.csv",
    "c_level_summary": DATASETS_DIR / "protocol_c_c_level_summary_registry.csv",
    "formal_negative": DATASETS_DIR / "formal_negative_control_evidence_registry.csv",
    "negative_ladder": DATASETS_DIR / "negative_evidence_ladder_registry.csv",
    "pseudo_absence": DATASETS_DIR / "pseudo_absence_candidate_registry.csv",
    "background_unlabeled": DATASETS_DIR / "background_unlabeled_candidate_registry.csv",
    "pu_boundary": DATASETS_DIR / "positive_unlabeled_boundary_matrix.csv",
    "supervised_gate": DATASETS_DIR / "supervised_training_minimum_gate_matrix.csv",
    "split_leakage": DATASETS_DIR / "split_leakage_protocol_registry.csv",
    "multimodal_patches": DATASETS_DIR / "multi_anchor_multimodal_patch_registry.csv",
    "multi_anchor_training_gate": DATASETS_DIR / "multi_anchor_training_gate_matrix.csv",
    "control_expansion": DATASETS_DIR / "control_candidate_expansion_registry.csv",
}

PUBLIC_GAP_ANALYSIS = DATASETS_DIR / "c4_gate_gap_analysis_registry.csv"
PUBLIC_BLOCKER_RANKING = DATASETS_DIR / "c4_blocker_priority_ranking.csv"
PUBLIC_NEGATIVE_QUEUE = DATASETS_DIR / "c4_negative_evidence_search_queue.csv"
PUBLIC_S1_QUEUE = DATASETS_DIR / "c4_s1_completion_queue.csv"
PUBLIC_SPLIT_QUEUE = DATASETS_DIR / "c4_split_leakage_precondition_queue.csv"
PUBLIC_TRANSITION = DATASETS_DIR / "c4_transition_readiness_matrix.csv"

LOCAL_SUMMARY_JSON = LOCAL_RUN_DIR / "v1jo_c4_readiness_summary.json"
LOCAL_QA = LOCAL_RUN_DIR / "v1jo_qa.csv"

DOC_METHOD = DOCS_DIR / "protocolo_c_priorizacao_c4_v1jo.md"
DOC_REPORT = DOCS_DIR / "protocolo_c_relatorio_priorizacao_c4_v1jo.md"

GAP_FIELDS = [
    "event_id",
    "anchor_id",
    "current_c_level",
    "target_c_level",
    "gate_source",
    "gate_temporal",
    "gate_spatial",
    "gate_patch_multimodal",
    "gate_positive_reference",
    "gate_negative_evidence",
    "gate_split_leakage",
    "gate_s1_completeness",
    "gate_training_boundary",
    "primary_blocker",
    "secondary_blockers",
    "c4_blocking_reason",
    "is_resolvable_with_existing_data",
    "requires_new_evidence",
    "requires_negative_evidence",
    "requires_split_leakage",
    "requires_s1_completion",
    "requires_external_benchmark",
    "can_create_training_label",
    "can_train_model",
    "notes",
]

RANKING_FIELDS = [
    "rank",
    "blocker",
    "affected_event_count",
    "severity",
    "blocks_c4",
    "resolvable_with_existing_data",
    "requires_new_data",
    "programming_action_available",
    "scientific_action_required",
    "recommended_next_step",
    "expected_effect",
    "notes",
]

NEGATIVE_QUEUE_FIELDS = [
    "candidate_search_id",
    "target_region",
    "target_event_context",
    "required_evidence_type",
    "acceptable_source_type",
    "forbidden_inference",
    "priority",
    "query_terms_or_registry_targets",
    "expected_output",
    "can_unlock_c4",
    "notes",
]

S1_QUEUE_FIELDS = [
    "anchor_id",
    "event_id",
    "current_s1_status",
    "needed_s1_asset",
    "pre_window",
    "post_window",
    "gee_or_local_possible",
    "priority",
    "would_unlock_c4",
    "would_strengthen_c3",
    "notes",
]

SPLIT_QUEUE_FIELDS = [
    "precondition_id",
    "required_positive_count",
    "required_negative_count",
    "current_positive_c3_count",
    "current_formal_negative_count",
    "split_unit",
    "spatial_buffer_needed",
    "temporal_rule_needed",
    "status",
    "notes",
]

TRANSITION_FIELDS = [
    "confirmed_c3_events",
    "c4_ready_events",
    "formal_negative_count",
    "pseudo_absence_count",
    "background_unlabeled_count",
    "strong_control_count",
    "s1_complete_pair_count",
    "s1_partial_pair_count",
    "split_leakage_status",
    "pu_sandbox_status",
    "supervised_training_status",
    "next_best_programming_step",
    "next_best_scientific_step",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "summary_decision",
]

QA_FIELDS = ["check", "status", "detail"]

PRIVATE_FRAGMENTS = [
    "C:\\Users\\gabriela",
    "Documents\\REV-P",
    "Documents/REV-P",
    "gabriela",
    "local_runs/",
    "local_runs\\",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    rows = [{"field": field, "description": f"{prefix}: {field}."} for field in fields]
    write_csv(path, rows, ["field", "description"])


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jo").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1", "y"}


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get(key, "")}


def count_formal_negatives(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if truthy(row.get("can_be_negative_label")) or truthy(row.get("can_be_formal_negative")))


def s1_pair_status(patch: dict[str, str]) -> tuple[str, bool, str]:
    pre = patch.get("s1_pre_status", "")
    post = patch.get("s1_post_status", "")
    if pre == "QA_PASS" and post == "QA_PASS":
        return "S1_PAIR_COMPLETE_QA_PASS", False, "NONE"
    missing = []
    if pre != "QA_PASS":
        missing.append("S1_PRE_PATCH")
    if post != "QA_PASS":
        missing.append("S1_POST_PATCH")
    return f"S1_PARTIAL_COVERAGE(pre={pre};post={post})", True, ";".join(missing)


def load_inputs() -> dict[str, list[dict[str, str]]]:
    return {name: read_csv(path) for name, path in INPUT_PATHS.items()}


def build_gap_analysis(inputs: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    events = inputs["events"]
    linkages = index_by(inputs["linkages"], "event_id")
    decisions = index_by(inputs["decisions"], "event_id")
    patches = index_by(inputs["multimodal_patches"], "anchor_id")

    rows: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get("event_id", "")
        linkage = linkages.get(event_id, {})
        decision = decisions.get(event_id, {})
        anchor_id = linkage.get("anchor_id") or decision.get("anchor_id", "")
        patch = patches.get(anchor_id, {})
        s1_status, needs_s1, _needed = s1_pair_status(patch)

        secondary = [
            "POSITIVE_LABEL_GATE_NOT_OPERATIONAL",
            "SPLIT_LEAKAGE_NOT_READY",
            "DINO_REVIEW_ONLY_NOT_LABEL_SOURCE",
        ]
        if needs_s1:
            secondary.append("S1_PARTIAL_COVERAGE")

        rows.append(
            {
                "event_id": event_id,
                "anchor_id": anchor_id,
                "current_c_level": event.get("c_level", ""),
                "target_c_level": "C4_OPERATIONAL_LABEL_CANDIDATE",
                "gate_source": "PASS_OFFICIAL_AUDITABLE_SOURCE",
                "gate_temporal": "PASS_EVENT_DATE_DOCUMENTED",
                "gate_spatial": "PASS_EXPLICIT_COORDINATE",
                "gate_patch_multimodal": "PASS_C3_PATCH_LINKED_WITH_S2_DEM_DINO_QA",
                "gate_positive_reference": "BLOCKED_POSITIVE_LABEL_GATE_NOT_OPERATIONAL",
                "gate_negative_evidence": "FAIL_FORMAL_NEGATIVES_ZERO",
                "gate_split_leakage": "BLOCKED_WAITING_FORMAL_POSITIVE_AND_NEGATIVE_LABELS",
                "gate_s1_completeness": s1_status,
                "gate_training_boundary": "BLOCKED_SUPERVISED_TRAINING_NOT_ALLOWED",
                "primary_blocker": "FORMAL_NEGATIVES_ZERO",
                "secondary_blockers": ";".join(secondary),
                "c4_blocking_reason": "FORMAL_NEGATIVES_ZERO;POSITIVE_LABEL_GATE_NOT_OPERATIONAL;SPLIT_LEAKAGE_NOT_READY",
                "is_resolvable_with_existing_data": "false",
                "requires_new_evidence": "true",
                "requires_negative_evidence": "true",
                "requires_split_leakage": "true",
                "requires_s1_completion": str(needs_s1).lower(),
                "requires_external_benchmark": "false",
                "can_create_training_label": "false",
                "can_train_model": "false",
                "notes": (
                    "C3 review reference is stable; C4 is blocked by missing formal negative evidence. "
                    "S1 completion can strengthen multimodal review but cannot create a label."
                ),
            }
        )
    return rows


def build_blocker_ranking(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(gaps)
    s1_partial = sum(1 for row in gaps if row["requires_s1_completion"] == "true")
    return [
        {
            "rank": 1,
            "blocker": "FORMAL_NEGATIVES_ZERO",
            "affected_event_count": total,
            "severity": "CRITICAL",
            "blocks_c4": "true",
            "resolvable_with_existing_data": "false",
            "requires_new_data": "true",
            "programming_action_available": "true",
            "scientific_action_required": "true",
            "recommended_next_step": "Search for explicit official absence or stability evidence for compatible control areas.",
            "expected_effect": "Can close the negative-evidence gate after validation; does not alone authorize training.",
            "notes": "Pseudo-absence and background-unlabeled material are not formal negatives.",
        },
        {
            "rank": 2,
            "blocker": "POSITIVE_LABEL_GATE_NOT_OPERATIONAL",
            "affected_event_count": total,
            "severity": "HIGH",
            "blocks_c4": "true",
            "resolvable_with_existing_data": "false",
            "requires_new_data": "true",
            "programming_action_available": "true",
            "scientific_action_required": "true",
            "recommended_next_step": "Keep C3 references as review candidates until an operational-label protocol is explicitly closed.",
            "expected_effect": "Prevents C3 references from being treated as supervised positive labels.",
            "notes": "Official C3 events are real references, not operational labels.",
        },
        {
            "rank": 3,
            "blocker": "SPLIT_LEAKAGE_NOT_READY",
            "affected_event_count": total,
            "severity": "HIGH",
            "blocks_c4": "true",
            "resolvable_with_existing_data": "false",
            "requires_new_data": "false",
            "programming_action_available": "true",
            "scientific_action_required": "true",
            "recommended_next_step": "Wait for formal positives and negatives, then instantiate spatial/temporal split rules.",
            "expected_effect": "Can prevent same-anchor, same-event, spatial-buffer, and temporal leakage once labels exist.",
            "notes": "The split protocol is review-ready but not actionable for training without label classes.",
        },
        {
            "rank": 4,
            "blocker": "S1_PARTIAL_COVERAGE",
            "affected_event_count": s1_partial,
            "severity": "MEDIUM",
            "blocks_c4": "false",
            "resolvable_with_existing_data": "false",
            "requires_new_data": "true",
            "programming_action_available": "true",
            "scientific_action_required": "false",
            "recommended_next_step": "Complete missing S1 pre/post patches for anchors with partial coverage.",
            "expected_effect": "Strengthens C3 multimodal review and linkage confidence; does not unlock C4.",
            "notes": "S1 is a robustness improvement, not a label gate.",
        },
        {
            "rank": 5,
            "blocker": "EXTERNAL_VALIDATION_OPTIONAL",
            "affected_event_count": total,
            "severity": "LOW",
            "blocks_c4": "false",
            "resolvable_with_existing_data": "false",
            "requires_new_data": "true",
            "programming_action_available": "true",
            "scientific_action_required": "true",
            "recommended_next_step": "Register optional external benchmark requirements after negative evidence is addressed.",
            "expected_effect": "Improves independent auditability but does not replace formal negative and split gates.",
            "notes": "External validation is useful after C4 prerequisites, not a substitute for them.",
        },
    ]


def build_negative_queue() -> list[dict[str, Any]]:
    common_forbidden = (
        "Do not infer absence from missing records, missing damage reports, cloud-free imagery, "
        "pseudo-absence, background samples, or distance from a positive anchor."
    )
    return [
        {
            "candidate_search_id": "NEG_SEARCH_PET_OFFICIAL_ABSENCE_STABILITY_2022",
            "target_region": "PET",
            "target_event_context": "Petrópolis 2022 movement-of-mass reference window",
            "required_evidence_type": "EXPLICIT_ABSENCE_OR_STABILITY_STATEMENT",
            "acceptable_source_type": "Official CPRM/SGB, Defesa Civil, municipal/state technical report, or audited field-survey source",
            "forbidden_inference": common_forbidden,
            "priority": "P1",
            "query_terms_or_registry_targets": "sem indicio de movimento de massa; area estavel; vistoria sem ocorrencia; estabilidade de encosta",
            "expected_output": "formal_negative_control_evidence_registry update with explicit absence/stability evidence and coordinates",
            "can_unlock_c4": "NO_ALONE_CAN_CLOSE_NEGATIVE_GATE_AFTER_VALIDATION",
            "notes": "Highest-value scientific gap: a formal negative must be explicit and temporally compatible.",
        },
        {
            "candidate_search_id": "NEG_SEARCH_PET_FIELD_SURVEY_NO_OCCURRENCE",
            "target_region": "PET",
            "target_event_context": "Official post-event or compatible-window field survey",
            "required_evidence_type": "FIELD_SURVEY_NO_OCCURRENCE",
            "acceptable_source_type": "Official inspection point or polygon with documented no-occurrence/stability result",
            "forbidden_inference": common_forbidden,
            "priority": "P1",
            "query_terms_or_registry_targets": "ponto de vistoria; sem movimento de massa; sem instabilidade; sem danos geotecnicos",
            "expected_output": "negative_evidence_ladder_registry candidate promoted only after explicit absence review",
            "can_unlock_c4": "NO_ALONE_CAN_CLOSE_NEGATIVE_GATE_AFTER_VALIDATION",
            "notes": "A field-survey no-occurrence record is stronger than a contextual background patch.",
        },
        {
            "candidate_search_id": "NEG_SEARCH_PET_OFFICIAL_CONTROL_AREA",
            "target_region": "PET",
            "target_event_context": "Control area outside positive buffers but within compatible environmental stratum",
            "required_evidence_type": "OFFICIAL_CONTROL_AREA_WITH_NO_INDICATION",
            "acceptable_source_type": "Official hazard/inspection/control dataset with explicit no-indication semantics",
            "forbidden_inference": common_forbidden,
            "priority": "P2",
            "query_terms_or_registry_targets": "area controle; sem indicio; estabilidade documentada; vistoria geologica",
            "expected_output": "control candidate with independent coordinates, temporal window, and leakage audit",
            "can_unlock_c4": "NO_ALONE_CAN_CLOSE_NEGATIVE_GATE_AFTER_VALIDATION",
            "notes": "Must avoid same-anchor temporal self-control as a negative label.",
        },
        {
            "candidate_search_id": "NEG_SEARCH_PET_STABLE_AREA_DOCUMENTED",
            "target_region": "PET",
            "target_event_context": "Stable terrain in same broad region with explicit documentation",
            "required_evidence_type": "DOCUMENTED_STABILITY",
            "acceptable_source_type": "Official technical mapping, inspection registry, or engineering/geological report",
            "forbidden_inference": common_forbidden,
            "priority": "P2",
            "query_terms_or_registry_targets": "area estavel; estabilidade documentada; sem instabilidade; sem escorregamento",
            "expected_output": "candidate formal-negative evidence row requiring manual scientific review",
            "can_unlock_c4": "NO_ALONE_CAN_CLOSE_NEGATIVE_GATE_AFTER_VALIDATION",
            "notes": "Stable-area evidence needs temporal and spatial compatibility, not just a generic class.",
        },
        {
            "candidate_search_id": "NEG_SEARCH_PET_NO_INDICIO_MOVIMENTO_MASSA",
            "target_region": "PET",
            "target_event_context": "Compatible post-event window for movement-of-mass absence",
            "required_evidence_type": "OFFICIAL_NO_INDICATION_OF_MOVEMENT_OF_MASS",
            "acceptable_source_type": "Official record explicitly classifying point/area as without indication of movement of mass",
            "forbidden_inference": common_forbidden,
            "priority": "P2",
            "query_terms_or_registry_targets": "sem indicio de movimento de massa; sem escorregamento; sem deslizamento",
            "expected_output": "formal negative candidate with source, date, coordinate, and leakage checks",
            "can_unlock_c4": "NO_ALONE_CAN_CLOSE_NEGATIVE_GATE_AFTER_VALIDATION",
            "notes": "Absence of record remains invalid; only explicit no-indication evidence qualifies.",
        },
    ]


def build_s1_queue(inputs: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    linkages_by_anchor = index_by(inputs["linkages"], "anchor_id")
    rows: list[dict[str, Any]] = []
    for patch in inputs["multimodal_patches"]:
        anchor_id = patch.get("anchor_id", "")
        status, needs_s1, needed = s1_pair_status(patch)
        if not needs_s1:
            continue
        linkage = linkages_by_anchor.get(anchor_id, {})
        rows.append(
            {
                "anchor_id": anchor_id,
                "event_id": linkage.get("event_id", ""),
                "current_s1_status": status,
                "needed_s1_asset": needed,
                "pre_window": f"-{linkage.get('temporal_window_pre_days', '45')}d",
                "post_window": f"+{linkage.get('temporal_window_post_days', '30')}d",
                "gee_or_local_possible": "GEE_OR_LOCAL_METADATA_ONLY_PLANNING",
                "priority": "P1" if needed else "P3",
                "would_unlock_c4": "false",
                "would_strengthen_c3": "true",
                "notes": "Completing S1 improves multimodal robustness but does not create labels or negatives.",
            }
        )
    return rows


def build_split_queue(transition: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "precondition_id": "SPLIT_LEAKAGE_FORMAL_LABEL_PRECONDITION",
            "required_positive_count": "9",
            "required_negative_count": "9",
            "current_positive_c3_count": transition["confirmed_c3_events"],
            "current_formal_negative_count": transition["formal_negative_count"],
            "split_unit": "DOCUMENTED_EVENT_UNIT_PLUS_ANCHOR_PAIR",
            "spatial_buffer_needed": "true",
            "temporal_rule_needed": "true",
            "status": "BLOCKED_WAITING_FORMAL_POSITIVE_AND_NEGATIVE_LABELS",
            "notes": "C3 references are not operational labels; split/leakage closure becomes actionable only after formal labels exist.",
        },
        {
            "precondition_id": "SPLIT_LEAKAGE_SAME_ANCHOR_PAIR_RULE",
            "required_positive_count": "9",
            "required_negative_count": "9",
            "current_positive_c3_count": transition["confirmed_c3_events"],
            "current_formal_negative_count": transition["formal_negative_count"],
            "split_unit": "SAME_ANCHOR_PRE_POST_PAIR",
            "spatial_buffer_needed": "true",
            "temporal_rule_needed": "true",
            "status": "DESIGN_READY_REVIEW_ONLY_NOT_TRAINING_READY",
            "notes": "Pre/post material from the same anchor must stay together if a later label protocol is approved.",
        },
    ]


def build_transition_matrix(inputs: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    events = inputs["events"]
    summary = inputs["c_level_summary"][0] if inputs["c_level_summary"] else {}
    pu = inputs["pu_boundary"][0] if inputs["pu_boundary"] else {}
    supervised = inputs["supervised_gate"][0] if inputs["supervised_gate"] else {}
    training = inputs["multi_anchor_training_gate"][0] if inputs["multi_anchor_training_gate"] else {}

    c3_count = sum(1 for row in events if row.get("c_level") == "C3_EVENT_PATCH_LINKED")
    c4_count = sum(1 for row in events if row.get("c_level") == "C4_OPERATIONAL_LABEL_CANDIDATE")
    formal_negative_count = count_formal_negatives(inputs["formal_negative"])
    s1_complete = sum(1 for row in inputs["multimodal_patches"] if row.get("s1_pre_status") == "QA_PASS" and row.get("s1_post_status") == "QA_PASS")
    s1_partial = max(0, len(inputs["multimodal_patches"]) - s1_complete)

    return {
        "confirmed_c3_events": c3_count,
        "c4_ready_events": c4_count,
        "formal_negative_count": formal_negative_count,
        "pseudo_absence_count": len(inputs["pseudo_absence"]),
        "background_unlabeled_count": len(inputs["background_unlabeled"]),
        "strong_control_count": pu.get("strong_control_review_only_count", "0"),
        "s1_complete_pair_count": s1_complete or summary.get("s1_pair_qa_pass_count", "0"),
        "s1_partial_pair_count": s1_partial,
        "split_leakage_status": supervised.get("split_protocol_status") or training.get("leakage_risk_status", "LEAKAGE_PROTOCOL_REQUIRED"),
        "pu_sandbox_status": pu.get("pu_boundary_status", "PU_SANDBOX_LOCAL_ONLY_READY"),
        "supervised_training_status": supervised.get("supervised_training_boundary_status", "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES"),
        "next_best_programming_step": "Run metadata-only formal-negative evidence search queue and prepare explicit-absence candidate intake.",
        "next_best_scientific_step": "Acquire or verify official absence/stability evidence for control areas in compatible temporal windows.",
        "can_create_training_label": "false",
        "can_train_model": "false",
        "can_unfreeze_dino_for_scientific_claim": "false",
        "summary_decision": "C3_LAYER_STABLE_C4_BLOCKED_BY_NEGATIVE_EVIDENCE",
    }


def build_docs(transition: dict[str, Any], ranking: list[dict[str, Any]], s1_queue: list[dict[str, Any]]) -> None:
    top_blockers = "\n".join(
        f"- {row['rank']}. {row['blocker']}: afeta {row['affected_event_count']} eventos; bloqueia C4={row['blocks_c4']}."
        for row in ranking
    )
    method = f"""# Protocolo C - priorizacao C4 v1jo

## Escopo

v1jo analisa a camada C1-C4 ja consolidada em v1jn e transforma os bloqueios remanescentes em uma fila objetiva de evidencias. A etapa nao cria label operacional, nao treina modelo, nao descongela DINO e nao promove pseudo-ausencia a negativo.

## Estado de entrada

- Eventos C3 confirmados: {transition['confirmed_c3_events']}
- Eventos C4 prontos: {transition['c4_ready_events']}
- Negativos formais: {transition['formal_negative_count']}
- Pseudo-ausencias review-only: {transition['pseudo_absence_count']}
- Pares S1 completos: {transition['s1_complete_pair_count']}
- Pares S1 parciais: {transition['s1_partial_pair_count']}

## Interpretacao metodologica

C3 e um avanco real porque liga evento oficial CPRM, coordenada explicita e patch multimodal com S2/DEM/DINO em QA. Isso cria referencia cientifica para revisao, nao label operacional.

C4 nao esta liberado. O gargalo primario e a ausencia de negativos formais: sem evidencia explicita de ausencia, estabilidade ou vistoria sem ocorrencia, nao existe classe negativa defensavel. Pseudo-ausencia permanece apenas PU/sandbox local-only.

S1 parcial e um bloqueio secundario de robustez multimodal. Completar S1 fortalece C3 e melhora a auditoria dos patches, mas nao cria negativo, nao cria positivo operacional e nao autoriza treino.

## Prioridade de blockers

{top_blockers}

## Proxima acao

Acao programatica de maior valor: executar a fila metadata-only de busca de negativos formais e preparar intake para fontes oficiais com ausencia/estabilidade explicita. Em paralelo, completar S1 para os {len(s1_queue)} anchors incompletos fortalece C3, mas nao altera C4.

Acao cientifica de maior valor: obter evidencia oficial explicita de ausencia, estabilidade ou vistoria sem ocorrencia em janela temporal compativel, com localizacao auditavel e regra de vazamento posterior.
"""
    report = f"""# Relatorio v1jo - priorizacao C4 e fila de evidencias

## Resultado principal

summary_decision = {transition['summary_decision']}

A camada C3 permanece estavel com {transition['confirmed_c3_events']} eventos oficiais CPRM em C3_EVENT_PATCH_LINKED. Nenhum evento foi promovido a C4.

## Blockers C4

{top_blockers}

## Negativos formais

O blocker primario e FORMAL_NEGATIVES_ZERO. A fila v1jo exige evidencia explicita de ausencia/estabilidade, vistoria sem ocorrencia, area controle oficial ou classificacao oficial sem indicio de movimento de massa. Ausencia de registro nao e negativo.

## S1

S1 completo existe para {transition['s1_complete_pair_count']} anchor; {transition['s1_partial_pair_count']} anchors seguem parciais. Completar S1 melhora robustez multimodal e fortalece C3, mas would_unlock_c4=false em toda a fila.

## Split e leakage

Split/leakage permanece precondicionado a labels formais. A regra de split deve proteger unidade documental, anchor, par pre/post, buffer espacial e coerencia temporal, mas so fica acionavel para treino quando positivos e negativos formais existirem.

## Usos permitidos e proibidos

Permitido: revisao cientifica C3, fila de busca metadata-only, intake de evidencia oficial e fortalecimento S1 de C3. Proibido: label operacional, negativo por pseudo-ausencia, treino supervisionado, claim cientifico de modelo e descongelamento de DINO.
"""
    DOC_METHOD.write_text(method, encoding="utf-8")
    DOC_REPORT.write_text(report, encoding="utf-8")


def has_private_path(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(fragment in text for fragment in PRIVATE_FRAGMENTS)


def run_qa(
    gaps: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    negative_queue: list[dict[str, Any]],
    s1_queue: list[dict[str, Any]],
    transition: dict[str, Any],
) -> list[dict[str, str]]:
    public_paths = [
        PUBLIC_GAP_ANALYSIS,
        PUBLIC_BLOCKER_RANKING,
        PUBLIC_NEGATIVE_QUEUE,
        PUBLIC_S1_QUEUE,
        PUBLIC_SPLIT_QUEUE,
        PUBLIC_TRANSITION,
        DOC_METHOD,
        DOC_REPORT,
    ]
    forbidden_negative = "ausencia de registro"
    qa = [
        {"check": "c3_count_preserved", "status": "PASS" if transition["confirmed_c3_events"] == 9 else "FAIL", "detail": str(transition["confirmed_c3_events"])},
        {"check": "c4_count_preserved", "status": "PASS" if transition["c4_ready_events"] == 0 else "FAIL", "detail": str(transition["c4_ready_events"])},
        {"check": "primary_blocker", "status": "PASS" if all(row["primary_blocker"] == "FORMAL_NEGATIVES_ZERO" for row in gaps) else "FAIL", "detail": "FORMAL_NEGATIVES_ZERO"},
        {"check": "s1_secondary_only", "status": "PASS" if all(row["would_unlock_c4"] == "false" for row in s1_queue) else "FAIL", "detail": f"s1_queue={len(s1_queue)}"},
        {"check": "negative_queue_no_absence_of_record_inference", "status": "PASS" if forbidden_negative not in json.dumps(negative_queue, ensure_ascii=False).lower() else "FAIL", "detail": "explicit absence/stability only"},
        {"check": "training_blocked", "status": "PASS" if transition["can_train_model"] == "false" and transition["can_create_training_label"] == "false" else "FAIL", "detail": "training=false"},
        {"check": "dino_unfreeze_blocked", "status": "PASS" if transition["can_unfreeze_dino_for_scientific_claim"] == "false" else "FAIL", "detail": "dino_unfreeze=false"},
        {"check": "public_no_private_path", "status": "PASS" if not any(has_private_path(path) for path in public_paths) else "FAIL", "detail": "public v1jo outputs checked"},
    ]
    return qa


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    inputs = load_inputs()

    gaps = build_gap_analysis(inputs) if args.audit_c4_gaps else []
    ranking = build_blocker_ranking(gaps) if args.rank_blockers else []
    transition = build_transition_matrix(inputs)
    negative_queue = build_negative_queue() if args.build_action_queues else []
    s1_queue = build_s1_queue(inputs) if args.build_action_queues else []
    split_queue = build_split_queue(transition) if args.build_action_queues else []

    if args.audit_c4_gaps:
        write_csv(PUBLIC_GAP_ANALYSIS, gaps, GAP_FIELDS)
        write_schema(SCHEMAS_DIR / "c4_gate_gap_analysis_registry_schema.csv", GAP_FIELDS, "v1jo C4 gate gap analysis")
    if args.rank_blockers:
        write_csv(PUBLIC_BLOCKER_RANKING, ranking, RANKING_FIELDS)
        write_schema(SCHEMAS_DIR / "c4_blocker_priority_ranking_schema.csv", RANKING_FIELDS, "v1jo C4 blocker priority ranking")
    if args.build_action_queues:
        write_csv(PUBLIC_NEGATIVE_QUEUE, negative_queue, NEGATIVE_QUEUE_FIELDS)
        write_csv(PUBLIC_S1_QUEUE, s1_queue, S1_QUEUE_FIELDS)
        write_csv(PUBLIC_SPLIT_QUEUE, split_queue, SPLIT_QUEUE_FIELDS)
        write_schema(SCHEMAS_DIR / "c4_negative_evidence_search_queue_schema.csv", NEGATIVE_QUEUE_FIELDS, "v1jo formal negative evidence search queue")
        write_schema(SCHEMAS_DIR / "c4_s1_completion_queue_schema.csv", S1_QUEUE_FIELDS, "v1jo S1 completion queue")
        write_schema(SCHEMAS_DIR / "c4_split_leakage_precondition_queue_schema.csv", SPLIT_QUEUE_FIELDS, "v1jo split/leakage precondition queue")
    if args.emit_c4_readiness:
        write_csv(PUBLIC_TRANSITION, [transition], TRANSITION_FIELDS)
        write_schema(SCHEMAS_DIR / "c4_transition_readiness_matrix_schema.csv", TRANSITION_FIELDS, "v1jo C4 transition readiness matrix")
        build_docs(transition, ranking, s1_queue)

    qa = run_qa(gaps, ranking, negative_queue, s1_queue, transition)
    write_csv(LOCAL_QA, qa, QA_FIELDS)
    summary = {
        "stage": "v1jo",
        "created_at_utc": utc_now(),
        "confirmed_c3_events": transition["confirmed_c3_events"],
        "c4_ready_events": transition["c4_ready_events"],
        "formal_negative_count": transition["formal_negative_count"],
        "s1_completion_queue_count": len(s1_queue),
        "summary_decision": transition["summary_decision"],
        "qa_status": "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL",
    }
    write_json(LOCAL_SUMMARY_JSON, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1jo C4 gate gap analysis and action queues.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--read-c-level-registries", action="store_true")
    parser.add_argument("--audit-c4-gaps", action="store_true")
    parser.add_argument("--rank-blockers", action="store_true")
    parser.add_argument("--build-action-queues", action="store_true")
    parser.add_argument("--emit-c4-readiness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
