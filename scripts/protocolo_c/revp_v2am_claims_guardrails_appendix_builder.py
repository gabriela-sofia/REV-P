#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2am_common import parse_args, run_claims_guardrails_appendix_builder
except ModuleNotFoundError:
    from revp_v2am_common import parse_args, run_claims_guardrails_appendix_builder


if __name__ == "__main__":
    run_claims_guardrails_appendix_builder(parse_args())
