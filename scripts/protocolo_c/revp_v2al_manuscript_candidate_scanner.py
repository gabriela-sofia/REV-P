#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2al_common import parse_args, run_manuscript_candidate_scanner
except ModuleNotFoundError:
    from revp_v2al_common import parse_args, run_manuscript_candidate_scanner


if __name__ == "__main__":
    run_manuscript_candidate_scanner(parse_args())
