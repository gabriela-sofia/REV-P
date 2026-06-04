#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uz_common import parse_args, run_guardrail_audit
except ModuleNotFoundError:
    from revp_v1uz_common import parse_args, run_guardrail_audit


if __name__ == "__main__":
    run_guardrail_audit(parse_args())
