#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ah_common import parse_args, run_ground_truth_search_stop_gate
except ModuleNotFoundError:
    from revp_v2ah_common import parse_args, run_ground_truth_search_stop_gate


if __name__ == "__main__":
    run_ground_truth_search_stop_gate(parse_args())
