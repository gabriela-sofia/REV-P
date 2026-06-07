#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ap_common import parse_args, run_master_orchestrator
except ModuleNotFoundError:
    from revp_v2ap_common import parse_args, run_master_orchestrator


if __name__ == "__main__":
    run_master_orchestrator(parse_args())
