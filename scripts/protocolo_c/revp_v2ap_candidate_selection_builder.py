#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ap_common import parse_args, run_candidate_selection_builder
except ModuleNotFoundError:
    from revp_v2ap_common import parse_args, run_candidate_selection_builder


if __name__ == "__main__":
    run_candidate_selection_builder(parse_args())
