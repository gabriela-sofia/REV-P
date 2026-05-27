"""REV-P v1nn - DINO C3/control review-only triage queue."""

from __future__ import annotations

import argparse
import json

from revp_v1ni_v1nn_common import DATASETS, DOCS, SCHEMAS, read_csv, read_event_rows, read_linkage_by_event, write_doc, write_outputs


OUT_ANCHORS = DATASETS / "dino_c3_anchor_review_triage_registry.csv"
OUT_CONTROLS = DATASETS / "dino_control_candidate_review_queue.csv"
OUT_BOUNDARY = DATASETS / "dino_embedding_training_boundary_matrix.csv"
SCHEMA_ANCHORS = SCHEMAS / "dino_c3_anchor_review_triage_schema.csv"
SCHEMA_CONTROLS = SCHEMAS / "dino_control_candidate_review_queue_schema.csv"
SCHEMA_BOUNDARY = SCHEMAS / "dino_embedding_training_boundary_schema.csv"
DOC = DOCS / "protocolo_c_dino_triagem_review_only_v1nn.md"

ANCHOR_FIELDS = ["triage_id", "event_id", "anchor_id", "patch_candidate_id", "embedding_registry_status", "dino_status", "embedding_values_status", "review_priority", "can_create_label", "can_validate_event", "can_prioritize_review", "notes"]
CONTROL_FIELDS = ["queue_id", "candidate_id", "candidate_source", "region", "dino_metadata_status", "review_priority", "review_purpose", "official_negative_request_relevance", "can_be_formal_negative", "can_create_label", "can_train_model", "notes"]
BOUNDARY_FIELDS = ["boundary_id", "embedding_can_create_label", "embedding_can_validate_event", "embedding_can_prioritize_review", "embedding_can_support_negative_search_queue", "embedding_can_train_classifier_now", "training_blocker", "dino_role", "notes"]


def embedding_anchor_ids() -> set[str]:
    ids: set[str] = set()
    for path in [DATASETS / "multi_anchor_dino_review_embedding_registry.csv", DATASETS / "official_anchor_dino_embedding_readiness_registry.csv"]:
        for row in read_csv(path):
            if row.get("anchor_id"):
                ids.add(row["anchor_id"])
    return ids


def build_anchor_triage() -> list[dict[str, str]]:
    events = read_event_rows()
    linkages = read_linkage_by_event()
    embedded = embedding_anchor_ids()
    rows: list[dict[str, str]] = []
    for idx, event in enumerate(events, 1):
        linkage = linkages.get(event.get("event_id", ""), {})
        anchor_id = linkage.get("anchor_id") or event.get("event_id", "").replace("EVENT_", "ANCHOR_")
        has_embedding = anchor_id in embedded or linkage.get("dino_status") == "DINO_QA_PASS"
        rows.append(
            {
                "triage_id": f"DINO_C3_TRIAGE_V1NN_{idx:03d}",
                "event_id": event.get("event_id", ""),
                "anchor_id": anchor_id,
                "patch_candidate_id": linkage.get("patch_candidate_id", ""),
                "embedding_registry_status": "EMBEDDING_METADATA_AVAILABLE" if has_embedding else "EMBEDDING_METADATA_MISSING",
                "dino_status": linkage.get("dino_status") or "DINO_METADATA_NOT_FOUND",
                "embedding_values_status": "EMBEDDING_VALUES_NOT_VERSIONED_LOCAL_ONLY",
                "review_priority": "HIGH" if not has_embedding else "MEDIUM",
                "can_create_label": "false",
                "can_validate_event": "false",
                "can_prioritize_review": "true",
                "notes": "DINO metadata can prioritize review only. It cannot validate the event, create a negative, or support training.",
            }
        )
    return rows


def build_control_queue() -> list[dict[str, str]]:
    candidates = read_csv(DATASETS / "pseudo_absence_candidate_registry.csv")
    if not candidates:
        candidates = read_csv(DATASETS / "control_dino_readiness_registry.csv")
    rows: list[dict[str, str]] = []
    for idx, candidate in enumerate(candidates[:25], 1):
        candidate_id = candidate.get("candidate_id") or candidate.get("control_candidate_id") or f"CONTROL_CANDIDATE_{idx:03d}"
        rows.append(
            {
                "queue_id": f"DINO_CONTROL_QUEUE_V1NN_{idx:03d}",
                "candidate_id": candidate_id,
                "candidate_source": candidate.get("source_type") or "control_dino_readiness_registry",
                "region": candidate.get("region") or "PET",
                "dino_metadata_status": candidate.get("dino_status") or candidate.get("dino_feature_status") or "METADATA_ONLY",
                "review_priority": "HIGH" if candidate.get("region", "PET") == "PET" else "MEDIUM",
                "review_purpose": "prioritize visual review and official negative evidence request targets",
                "official_negative_request_relevance": "useful_for_official_negative_search_queue_not_label",
                "can_be_formal_negative": "false",
                "can_create_label": "false",
                "can_train_model": "false",
                "notes": "Pseudo-absence/control candidates remain review-only and cannot become negatives without official explicit negative evidence.",
            }
        )
    if not rows:
        rows.append(
            {
                "queue_id": "DINO_CONTROL_QUEUE_V1NN_NONE",
                "candidate_id": "none",
                "candidate_source": "none",
                "region": "PET",
                "dino_metadata_status": "NO_CONTROL_CANDIDATE_METADATA",
                "review_priority": "LOW",
                "review_purpose": "no review queue built",
                "official_negative_request_relevance": "none",
                "can_be_formal_negative": "false",
                "can_create_label": "false",
                "can_train_model": "false",
                "notes": "No candidate metadata was available.",
            }
        )
    return rows


def boundary() -> list[dict[str, str]]:
    return [
        {
            "boundary_id": "DINO_TRAINING_BOUNDARY_V1NN",
            "embedding_can_create_label": "false",
            "embedding_can_validate_event": "false",
            "embedding_can_prioritize_review": "true",
            "embedding_can_support_negative_search_queue": "true",
            "embedding_can_train_classifier_now": "false",
            "training_blocker": "FORMAL_NEGATIVES_ZERO;C4_NOT_OPEN;SPLIT_LEAKAGE_NOT_READY",
            "dino_role": "REVIEW_ONLY_REPRESENTATION",
            "notes": "No classifier training, no supervised separability claim, no operational metric.",
        }
    ]


def write_method_doc(anchor_count: int, control_count: int) -> None:
    write_doc(
        DOC,
        "Protocolo C - DINO triagem review-only v1nn",
        [
            f"Triagem criada para {anchor_count} anchors C3 e {control_count} candidatos controle/pseudo-ausencia.",
            "Embeddings DINO permanecem congelados e metadata-only nos outputs versionaveis. Valores vetoriais brutos nao sao carregados nem versionados.",
            "A fila serve para priorizar revisao visual, vizinhos, outliers e busca oficial de evidencia negativa. Nao cria label, nao valida evento e nao treina classificador.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_ANCHORS.exists() and OUT_CONTROLS.exists() and OUT_BOUNDARY.exists() and not args.force:
        print(json.dumps({"stage": "v1nn", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    anchors = build_anchor_triage()
    controls = build_control_queue()
    bounds = boundary()
    if args.force or args.emit_evidence:
        write_method_doc(len(anchors), len(controls))
        write_outputs(
            [(OUT_ANCHORS, anchors, ANCHOR_FIELDS), (OUT_CONTROLS, controls, CONTROL_FIELDS), (OUT_BOUNDARY, bounds, BOUNDARY_FIELDS)],
            [(SCHEMA_ANCHORS, ANCHOR_FIELDS, "v1nn DINO C3 anchor triage"), (SCHEMA_CONTROLS, CONTROL_FIELDS, "v1nn DINO control candidate review queue"), (SCHEMA_BOUNDARY, BOUNDARY_FIELDS, "v1nn DINO training boundary")],
            [DOC],
        )
    print(json.dumps({"stage": "v1nn", "anchors": len(anchors), "embedding_can_train_classifier_now": "false"}, indent=2))


if __name__ == "__main__":
    main()
