#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_patch_date_candidate_consolidator
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_patch_date_candidate_consolidator


if __name__ == "__main__":
    run_patch_date_candidate_consolidator(parse_args())
