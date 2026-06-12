#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_orchestrator
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_orchestrator
if __name__ == "__main__":
    run_orchestrator(parse_args())
