#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_content_mismatch_resolver
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_content_mismatch_resolver


if __name__ == "__main__":
    run_content_mismatch_resolver(parse_args())
