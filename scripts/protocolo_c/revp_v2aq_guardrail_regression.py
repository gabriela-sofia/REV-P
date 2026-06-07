#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_guardrail_regression
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_guardrail_regression


if __name__ == "__main__":
    run_guardrail_regression(parse_args())
