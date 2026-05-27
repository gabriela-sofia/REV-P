"""REV-P v1nc - administrative act segmentation from gazette text."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import segment_gazette_acts, write_simple_doc


OUT_SEG = DATASETS / "administrative_act_segment_registry.csv"
OUT_HIT = DATASETS / "administrative_act_keyword_hit_registry.csv"
SCHEMA_SEG = SCHEMAS / "administrative_act_segment_schema.csv"
SCHEMA_HIT = SCHEMAS / "administrative_act_keyword_hit_schema.csv"
DOC = DOCS / "protocolo_c_segmentacao_atos_diario_v1nc.md"
SEG_FIELDS = ["act_id", "issue_id", "issue_date", "page", "act_type", "context", "matched_terms", "extraction_method", "private_path_removed"]
HIT_FIELDS = ["hit_id", "act_id", "issue_id", "page", "matched_term", "term_class", "private_path_removed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_SEG.exists() and OUT_HIT.exists() and not args.force:
        print(json.dumps({"stage": "v1nc", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    seg, hits = segment_gazette_acts()
    if args.force or args.emit_evidence:
        write_csv(OUT_SEG, seg, SEG_FIELDS)
        write_csv(OUT_HIT, hits, HIT_FIELDS)
        write_schema(SCHEMA_SEG, SEG_FIELDS, "v1nc administrative act segment")
        write_schema(SCHEMA_HIT, HIT_FIELDS, "v1nc administrative act keyword hit")
        write_simple_doc(DOC, "Protocolo C - segmentacao de atos v1nc", [f"Atos/trechos segmentados: {len([r for r in seg if r['act_id'] != 'GAZETTEACT_V1NC_NONE'])}", "Segmentacao e por termo administrativo; nao cria negativo formal."])
    print(json.dumps({"stage": "v1nc", "segments": len(seg), "hits": len(hits)}, indent=2))


if __name__ == "__main__":
    main()
