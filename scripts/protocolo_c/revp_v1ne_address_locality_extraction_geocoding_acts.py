"""REV-P v1ne - address/locality extraction and geocoding gate for gazette candidates."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import geocode_gazette_candidates, write_simple_doc


OUT_GEO = DATASETS / "gazette_negative_address_geocoding_registry.csv"
OUT_MATRIX = DATASETS / "gazette_negative_spatial_specificity_matrix.csv"
SCHEMA_GEO = SCHEMAS / "gazette_negative_address_geocoding_schema.csv"
SCHEMA_MATRIX = SCHEMAS / "gazette_negative_spatial_specificity_schema.csv"
DOC = DOCS / "protocolo_c_geocodificacao_negativos_diario_v1ne.md"
GEO_FIELDS = ["geocode_id", "candidate_id", "address_or_locality_status", "latitude", "longitude", "precise_location_gate", "coordinate_or_geocodable_address_gate", "review_area_only_flag", "nearest_positive_distance_m", "positive_buffer_exclusion_gate", "patch_extractability_gate", "private_path_removed"]
MATRIX_FIELDS = ["matrix_id", "candidate_count", "precise_location_pass_count", "coordinate_or_geocodable_address_pass_count", "patch_extractability_pass_count", "decision", "remaining_blocker"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_GEO.exists() and OUT_MATRIX.exists() and not args.force:
        print(json.dumps({"stage": "v1ne", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    geo, matrix = geocode_gazette_candidates()
    if args.force or args.emit_evidence:
        write_csv(OUT_GEO, geo, GEO_FIELDS)
        write_csv(OUT_MATRIX, [matrix], MATRIX_FIELDS)
        write_schema(SCHEMA_GEO, GEO_FIELDS, "v1ne gazette negative geocoding")
        write_schema(SCHEMA_MATRIX, MATRIX_FIELDS, "v1ne gazette negative spatial specificity")
        write_simple_doc(DOC, "Protocolo C - geocodificacao de candidatos do Diario v1ne", [f"Candidatos geocodificados: {matrix['candidate_count']}", f"Decisao espacial: {matrix['decision']}", "Bairro generico fica REVIEW_AREA_ONLY e nao passa formal."])
    print(json.dumps({"stage": "v1ne", **matrix}, indent=2))


if __name__ == "__main__":
    main()
