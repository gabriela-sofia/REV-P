#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ad_common import parse_args, run_qa_gate_summary_builder
except ModuleNotFoundError:
    from revp_v2ad_common import parse_args, run_qa_gate_summary_builder


if __name__ == "__main__":
    run_qa_gate_summary_builder(parse_args())
