#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ac_common import parse_args, run_next_programming_target_ranker
except ModuleNotFoundError:
    from revp_v2ac_common import parse_args, run_next_programming_target_ranker


if __name__ == "__main__":
    run_next_programming_target_ranker(parse_args())
