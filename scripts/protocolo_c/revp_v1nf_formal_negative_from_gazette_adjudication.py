"""REV-P v1nf - strict formal negative adjudication from gazette acts."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import adjudicate_gazette_negatives, write_simple_doc


OUT_REG = DATASETS / "formal_negative_gazette_candidate_registry.csv"
OUT_MATRIX = DATASETS / "formal_negative_gazette_gate_matrix.csv"
SCHEMA_REG = SCHEMAS / "formal_negative_gazette_candidate_schema.csv"
SCHEMA_MATRIX = SCHEMAS / "formal_negative_gazette_gate_schema.csv"
DOC = DOCS / "protocolo_c_adjudicacao_negativos_diario_v1nf.md"
REG_FIELDS = ["candidate_id", "source_candidate_id", "official_gazette_gate", "administrative_act_gate", "explicit_negative_statement_gate", "phenomenon_specific_gate", "date_gate", "precise_location_gate", "coordinate_or_geocodable_address_gate", "independent_area_gate", "positive_buffer_exclusion_gate", "patch_extractability_gate", "leakage_precheck_gate", "decision", "can_create_operational_label", "can_train_model"]
MATRIX_FIELDS = ["matrix_id", "formal_negative_count", "review_or_blocked_count", "decision", "remaining_blocker", "can_create_operational_label", "can_train_model"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_REG.exists() and OUT_MATRIX.exists() and not args.force:
        print(json.dumps({"stage": "v1nf", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    rows, matrix = adjudicate_gazette_negatives()
    if args.force or args.emit_evidence:
        write_csv(OUT_REG, rows, REG_FIELDS)
        write_csv(OUT_MATRIX, [matrix], MATRIX_FIELDS)
        write_schema(SCHEMA_REG, REG_FIELDS, "v1nf formal negative gazette candidate")
        write_schema(SCHEMA_MATRIX, MATRIX_FIELDS, "v1nf formal negative gazette gate")
        write_simple_doc(DOC, "Protocolo C - adjudicacao negativos Diario v1nf", [f"Negativos formais: {matrix['formal_negative_count']}", f"Bloqueador: {matrix['remaining_blocker']}", "Candidato so passa com declaracao explicita, fenomeno, data, local preciso, buffer, patch e leakage."])
    print(json.dumps({"stage": "v1nf", **matrix}, indent=2))


if __name__ == "__main__":
    main()
