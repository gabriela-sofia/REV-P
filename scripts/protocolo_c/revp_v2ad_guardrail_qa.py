#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ad_common import parse_args, run_guardrail_qa
except ModuleNotFoundError:
    from revp_v2ad_common import parse_args, run_guardrail_qa


if __name__ == "__main__":
    run_guardrail_qa(parse_args())
