#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ad_common import parse_args, run_readiness_consistency_qa
except ModuleNotFoundError:
    from revp_v2ad_common import parse_args, run_readiness_consistency_qa


if __name__ == "__main__":
    run_readiness_consistency_qa(parse_args())
