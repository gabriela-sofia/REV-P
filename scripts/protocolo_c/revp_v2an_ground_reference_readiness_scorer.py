#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_ground_reference_readiness_scorer
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_ground_reference_readiness_scorer


if __name__ == "__main__":
    run_ground_reference_readiness_scorer(parse_args())
