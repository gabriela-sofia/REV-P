"""REV-P v1nl - strict formal negative adjudicator from official intake."""

from __future__ import annotations

import argparse
import json

from revp_v1ni_v1nn_common import DATASETS, DOCS, SCHEMAS, read_csv, write_doc, write_outputs


PREVALIDATION = DATASETS / "official_negative_response_prevalidation_matrix.csv"
OUT_ADJ = DATASETS / "strict_formal_negative_adjudication_registry.csv"
OUT_GATES = DATASETS / "strict_formal_negative_gate_matrix.csv"
SCHEMA_ADJ = SCHEMAS / "strict_formal_negative_adjudication_schema.csv"
SCHEMA_GATES = SCHEMAS / "strict_formal_negative_gate_matrix_schema.csv"
DOC = DOCS / "protocolo_c_adjudicacao_negativo_formal_v1nl.md"

ADJ_FIELDS = ["candidate_negative_id", "source_response_id", "adjudication_status", "failed_gates", "can_be_formal_negative", "can_be_used_for_c4", "can_create_training_negative_label", "notes"]
GATE_FIELDS = [
    "candidate_negative_id",
    "official_source_pass",
    "explicit_negative_semantics_pass",
    "phenomenon_specificity_pass",
    "temporal_compatibility_pass",
    "spatial_specificity_pass",
    "uncertainty_registered_pass",
    "not_pseudo_absence_pass",
    "not_absence_of_record_pass",
    "leakage_precheck_pass",
    "all_gates_pass",
    "gate_status",
]


def gate_value(row: dict[str, str], key: str) -> str:
    return "PASS" if row.get(key) == "true" else "FAIL"


def build_adjudication() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    pre_rows = read_csv(PREVALIDATION)
    if not pre_rows:
        pre_rows = [{"response_id": "OFFNEG_RESPONSE_NONE_V1NK", "prevalidation_status": "NO_OFFICIAL_RESPONSE_INTAKE"}]
    adj_rows: list[dict[str, str]] = []
    gate_rows: list[dict[str, str]] = []
    for idx, row in enumerate(pre_rows, 1):
        candidate_id = f"STRICT_FORMAL_NEG_V1NL_{idx:03d}"
        no_intake = row.get("prevalidation_status") == "NO_OFFICIAL_RESPONSE_INTAKE"
        gates = {
            "official_source_pass": "FAIL" if no_intake else gate_value(row, "contains_official_source"),
            "explicit_negative_semantics_pass": "FAIL" if no_intake else gate_value(row, "contains_explicit_negative_semantics"),
            "phenomenon_specificity_pass": "FAIL" if no_intake else gate_value(row, "contains_phenomenon"),
            "temporal_compatibility_pass": "FAIL" if no_intake else gate_value(row, "contains_date"),
            "spatial_specificity_pass": "FAIL" if no_intake else gate_value(row, "contains_coordinate_address_or_bairro"),
            "uncertainty_registered_pass": "FAIL" if no_intake else "PASS",
            "not_pseudo_absence_pass": "PASS",
            "not_absence_of_record_pass": "PASS",
            "leakage_precheck_pass": "FAIL",
        }
        failed = [name for name, value in gates.items() if value != "PASS"]
        all_pass = not failed
        if no_intake:
            status = "BLOCKED_NO_OFFICIAL_RESPONSE_INTAKE"
        elif all_pass:
            status = "POTENTIAL_FORMAL_NEGATIVE_READY_FOR_C4_RECHECK"
        else:
            status = "BLOCKED_INSUFFICIENT_OFFICIAL_NEGATIVE_EVIDENCE"
        gate_rows.append(
            {
                "candidate_negative_id": candidate_id,
                **gates,
                "all_gates_pass": "PASS" if all_pass else "FAIL",
                "gate_status": status,
            }
        )
        adj_rows.append(
            {
                "candidate_negative_id": candidate_id,
                "source_response_id": row.get("response_id", ""),
                "adjudication_status": status,
                "failed_gates": ";".join(failed) if failed else "none",
                "can_be_formal_negative": "true" if all_pass else "false",
                "can_be_used_for_c4": "true" if all_pass else "false",
                "can_create_training_negative_label": "false",
                "notes": "Strict adjudication blocks label creation; training remains false even if a future C4 recheck finds usable formal negatives.",
            }
        )
    return adj_rows, gate_rows


def write_method_doc() -> None:
    write_doc(
        DOC,
        "Protocolo C - adjudicacao estrita de negativo formal v1nl",
        [
            "A adjudicacao exige fonte oficial, semantica negativa explicita, fenomeno, janela temporal, especificidade espacial, registro de incerteza, exclusao de pseudo-ausencia, exclusao de ausencia de registro e precheck de leakage.",
            "No estado atual, ausencia de resposta oficial real mantem o bloqueio metodologico e nao cria negativo formal.",
            "Mesmo candidatos futuros aprovados por metadados precisam manter can_create_training_negative_label=false ate fechamento completo de C4 e split/leakage.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_ADJ.exists() and OUT_GATES.exists() and not args.force:
        print(json.dumps({"stage": "v1nl", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    adj, gates = build_adjudication()
    if args.force or args.emit_evidence:
        write_method_doc()
        write_outputs(
            [(OUT_ADJ, adj, ADJ_FIELDS), (OUT_GATES, gates, GATE_FIELDS)],
            [(SCHEMA_ADJ, ADJ_FIELDS, "v1nl strict formal negative adjudication"), (SCHEMA_GATES, GATE_FIELDS, "v1nl strict formal negative gates")],
            [DOC],
        )
    print(json.dumps({"stage": "v1nl", "formal_negative_count": sum(1 for row in adj if row["can_be_formal_negative"] == "true")}, indent=2))


if __name__ == "__main__":
    main()
