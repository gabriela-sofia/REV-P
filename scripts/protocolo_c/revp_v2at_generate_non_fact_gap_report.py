#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_generate_non_fact_gap_report
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_generate_non_fact_gap_report
if __name__ == "__main__":
    run_generate_non_fact_gap_report(parse_args())
