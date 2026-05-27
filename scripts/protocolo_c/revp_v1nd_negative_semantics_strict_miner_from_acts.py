"""REV-P v1nd - strict negative semantics mining from gazette acts."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import mine_negative_semantics_from_acts, write_simple_doc


OUT_CAND = DATASETS / "gazette_negative_semantics_candidate_registry.csv"
OUT_REJ = DATASETS / "gazette_negative_semantics_rejection_registry.csv"
SCHEMA_CAND = SCHEMAS / "gazette_negative_semantics_candidate_schema.csv"
SCHEMA_REJ = SCHEMAS / "gazette_negative_semantics_rejection_schema.csv"
DOC = DOCS / "protocolo_c_mineracao_semantica_negativa_diario_v1nd.md"
FIELDS = ["candidate_id", "act_id", "issue_id", "issue_date", "page", "phrase_or_context", "explicit_negative_statement_gate", "date_gate", "location_gate", "phenomenon_specific_gate", "decision", "can_create_operational_label", "can_train_model"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_CAND.exists() and OUT_REJ.exists() and not args.force:
        print(json.dumps({"stage": "v1nd", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    cand, rej = mine_negative_semantics_from_acts()
    if args.force or args.emit_evidence:
        write_csv(OUT_CAND, cand, FIELDS)
        write_csv(OUT_REJ, rej, FIELDS)
        write_schema(SCHEMA_CAND, FIELDS, "v1nd gazette negative semantics candidate")
        write_schema(SCHEMA_REJ, FIELDS, "v1nd gazette negative semantics rejection")
        write_simple_doc(DOC, "Protocolo C - semantica negativa estrita v1nd", [f"Candidatos fortes: {len([r for r in cand if r['candidate_id'] != 'GAZETTENEG_V1ND_NONE'])}", "Desinterdicao sem justificativa explicita nao vira negativo.", "Baixo risco nao vira negativo formal."])
    print(json.dumps({"stage": "v1nd", "candidates": len(cand), "rejections": len(rej)}, indent=2))


if __name__ == "__main__":
    main()
