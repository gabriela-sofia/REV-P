#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ae_common import parse_args, run_multiregion_readiness_consolidator
except ModuleNotFoundError:
    from revp_v2ae_common import parse_args, run_multiregion_readiness_consolidator


if __name__ == "__main__":
    run_multiregion_readiness_consolidator(parse_args())
