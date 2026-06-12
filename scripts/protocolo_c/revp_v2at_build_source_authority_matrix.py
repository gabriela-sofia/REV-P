#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_build_source_authority_matrix
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_build_source_authority_matrix
if __name__ == "__main__":
    run_build_source_authority_matrix(parse_args())
