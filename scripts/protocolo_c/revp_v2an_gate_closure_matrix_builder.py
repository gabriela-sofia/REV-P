#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_gate_closure_matrix_builder
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_gate_closure_matrix_builder


if __name__ == "__main__":
    run_gate_closure_matrix_builder(parse_args())
