#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2af_common import parse_args, run_guardrail_regression_runner
except ModuleNotFoundError:
    from revp_v2af_common import parse_args, run_guardrail_regression_runner


if __name__ == "__main__":
    run_guardrail_regression_runner(parse_args())
