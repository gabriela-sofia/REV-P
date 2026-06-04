#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uw_curitiba_common import parse_args, run_hydromet_anchor_resolver
except ModuleNotFoundError:
    from revp_v1uw_curitiba_common import parse_args, run_hydromet_anchor_resolver


if __name__ == "__main__":
    run_hydromet_anchor_resolver(parse_args())
