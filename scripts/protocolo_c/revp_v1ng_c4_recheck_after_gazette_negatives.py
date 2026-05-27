"""REV-P v1ng - C4 recheck after gazette negative route."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import c4_after_gazette_negatives, write_simple_doc


OUT_C4 = DATASETS / "c4_recheck_after_gazette_negatives.csv"
OUT_READY = DATASETS / "c4_label_readiness_after_gazette_negatives.csv"
SCHEMA_C4 = SCHEMAS / "c4_recheck_after_gazette_negatives_schema.csv"
SCHEMA_READY = SCHEMAS / "c4_label_readiness_after_gazette_negatives_schema.csv"
DOC = DOCS / "protocolo_c_recheck_c4_diario_v1ng.md"
C4_FIELDS = ["decision_id", "formal_positive_count", "formal_negative_count", "negative_provenance_gate", "patch_extractability_gate", "split_leakage_gate", "decision", "remaining_blocker", "can_create_operational_label", "can_train_model"]
READY_FIELDS = ["readiness_id", "formal_positive_count", "formal_negative_count", "positive_gate", "negative_gate", "split_leakage_gate", "decision", "remaining_blocker", "can_create_operational_label", "can_train_model"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_C4.exists() and OUT_READY.exists() and not args.force:
        print(json.dumps({"stage": "v1ng", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    c4, ready = c4_after_gazette_negatives()
    if args.force or args.emit_evidence:
        write_csv(OUT_C4, [c4], C4_FIELDS)
        write_csv(OUT_READY, [ready], READY_FIELDS)
        write_schema(SCHEMA_C4, C4_FIELDS, "v1ng C4 recheck after gazette negatives")
        write_schema(SCHEMA_READY, READY_FIELDS, "v1ng C4 label readiness after gazette negatives")
        write_simple_doc(DOC, "Protocolo C - C4 apos Diario Oficial v1ng", [f"C4: {c4['decision']}", f"Negativos formais: {c4['formal_negative_count']}", f"Bloqueador: {c4['remaining_blocker']}"])
    print(json.dumps({"stage": "v1ng", **c4}, indent=2))


if __name__ == "__main__":
    main()
