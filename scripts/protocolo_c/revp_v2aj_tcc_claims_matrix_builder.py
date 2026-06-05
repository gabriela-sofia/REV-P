#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aj_common import parse_args, run_tcc_claims_matrix_builder
except ModuleNotFoundError:
    from revp_v2aj_common import parse_args, run_tcc_claims_matrix_builder


if __name__ == "__main__":
    run_tcc_claims_matrix_builder(parse_args())
