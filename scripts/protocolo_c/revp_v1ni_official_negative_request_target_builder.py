"""REV-P v1ni - official negative evidence request target builder."""

from __future__ import annotations

import argparse
import json

from revp_v1ni_v1nn_common import DATASETS, DOCS, SCHEMAS, read_event_rows, read_gap_rows, safe_join, write_doc, write_outputs


OUT_TARGETS = DATASETS / "official_negative_evidence_request_target_registry.csv"
OUT_FIELDS = DATASETS / "official_negative_evidence_required_fields_matrix.csv"
SCHEMA_TARGETS = SCHEMAS / "official_negative_evidence_request_target_schema.csv"
SCHEMA_FIELDS = SCHEMAS / "official_negative_evidence_required_fields_schema.csv"
DOC = DOCS / "protocolo_c_alvos_pedido_oficial_negativo_v1ni.md"

TARGET_FIELDS = [
    "request_target_id",
    "region",
    "municipality",
    "c4_blocker",
    "target_event_context",
    "target_phenomenon",
    "acceptable_negative_statement_type",
    "required_source_type",
    "required_temporal_window",
    "required_spatial_specificity",
    "required_fields",
    "forbidden_substitutes",
    "priority",
    "can_unlock_c4_alone",
    "notes",
]
REQUIRED_FIELDS = [
    "field_id",
    "request_target_id",
    "required_field",
    "minimum_acceptance_rule",
    "why_needed_for_formal_negative",
    "forbidden_substitute",
    "is_required_for_c4",
]


def build_targets() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    events = read_event_rows()
    gaps = {row.get("event_id", ""): row for row in read_gap_rows()}
    if not events:
        events = [
            {
                "event_id": "PET_C3_EVENTS_AGGREGATE",
                "region": "PET",
                "municipality": "Petropolis",
                "locality_text_sanitized": "Petropolis C3 anchors",
                "event_or_survey_date": "2022",
                "phenomenon_group": "MOVEMENT_OF_MASS",
            }
        ]

    required_names = [
        "official_source",
        "explicit_negative_or_stability_statement",
        "phenomenon_assessed",
        "inspection_or_statement_date",
        "compatible_2022_temporal_window",
        "address_coordinate_or_bairro",
        "issuing_body_or_technical_responsible",
        "uncertainty_or_scope_limits",
    ]
    targets: list[dict[str, str]] = []
    field_rows: list[dict[str, str]] = []
    for idx, event in enumerate(events, 1):
        gap = gaps.get(event.get("event_id", ""), {})
        target_id = f"OFFNEG_TARGET_V1NI_{idx:03d}"
        blocker = gap.get("primary_blocker") or "FORMAL_NEGATIVES_ZERO"
        locality = event.get("locality_text_sanitized") or "Petropolis"
        date = event.get("event_or_survey_date") or "2022"
        targets.append(
            {
                "request_target_id": target_id,
                "region": event.get("region") or "PET",
                "municipality": event.get("municipality") or "Petropolis",
                "c4_blocker": blocker,
                "target_event_context": f"C3 anchor {event.get('event_id', 'unknown')} near {locality}",
                "target_phenomenon": event.get("phenomenon_group") or "MOVEMENT_OF_MASS",
                "acceptable_negative_statement_type": "official explicit absence, stability, no occurrence, no instability, or no geological damage statement",
                "required_source_type": "LAI response, Defesa Civil record, Prefeitura record, SGB/CPRM/DRM technical record, inspection form, official technical table",
                "required_temporal_window": f"compatible with 2022 event or inspection context around {date}",
                "required_spatial_specificity": "address, coordinate, bairro, or locality specific enough to match a review area",
                "required_fields": safe_join(required_names),
                "forbidden_substitutes": "absence of registry;silent gazette;generic background patch;pseudo-absence;distance from positive anchor;positive-only occurrence list",
                "priority": "HIGH",
                "can_unlock_c4_alone": "false",
                "notes": "Target only requests evidence. It does not create a negative label or open C4 without strict adjudication and leakage checks.",
            }
        )
        for required in required_names:
            field_rows.append(
                {
                    "field_id": f"OFFNEG_FIELD_V1NI_{idx:03d}_{required.upper()}",
                    "request_target_id": target_id,
                    "required_field": required,
                    "minimum_acceptance_rule": "must be explicit in the official response or attached official document",
                    "why_needed_for_formal_negative": "formal negative requires official negative semantics with compatible place, time, and phenomenon",
                    "forbidden_substitute": "pseudo-absence, Diario Oficial silence, or lack of a positive occurrence record",
                    "is_required_for_c4": "true",
                }
            )
    return targets, field_rows


def write_method_doc(target_count: int) -> None:
    write_doc(
        DOC,
        "Protocolo C - alvos de pedido oficial negativo v1ni",
        [
            f"Esta etapa criou {target_count} alvos objetivos de pedido oficial para a lacuna C4 FORMAL_NEGATIVES_ZERO.",
            "Negativo formal exige declaracao oficial explicita de ausencia, estabilidade, sem ocorrencia, sem instabilidade ou sem dano geologico, com local, periodo e fenomeno compativeis.",
            "Ausencia de registro, Diario Oficial sem ato tecnico, background patch, distancia de anchor positivo e pseudo-ausencia nao sao negativos formais.",
            "Nenhum alvo abre C4 sozinho. Todos mantem can_unlock_c4_alone=false e dependem de intake, adjudicacao estrita e checagem de leakage.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_TARGETS.exists() and OUT_FIELDS.exists() and not args.force:
        print(json.dumps({"stage": "v1ni", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    targets, field_rows = build_targets()
    if args.force or args.emit_evidence:
        write_method_doc(len(targets))
        write_outputs(
            [(OUT_TARGETS, targets, TARGET_FIELDS), (OUT_FIELDS, field_rows, REQUIRED_FIELDS)],
            [(SCHEMA_TARGETS, TARGET_FIELDS, "v1ni official negative request targets"), (SCHEMA_FIELDS, REQUIRED_FIELDS, "v1ni official negative required fields")],
            [DOC],
        )
    print(json.dumps({"stage": "v1ni", "request_targets": len(targets), "can_unlock_c4_alone": "false"}, indent=2))


if __name__ == "__main__":
    main()
