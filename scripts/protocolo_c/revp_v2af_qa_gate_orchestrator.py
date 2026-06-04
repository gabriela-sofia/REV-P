#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2af_common import parse_args, run_qa_gate_orchestrator
except ModuleNotFoundError:
    from revp_v2af_common import parse_args, run_qa_gate_orchestrator


if __name__ == "__main__":
    run_qa_gate_orchestrator(parse_args())
