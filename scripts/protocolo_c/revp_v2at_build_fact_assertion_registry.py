#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_build_fact_assertion_registry
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_build_fact_assertion_registry
if __name__ == "__main__":
    run_build_fact_assertion_registry(parse_args())
