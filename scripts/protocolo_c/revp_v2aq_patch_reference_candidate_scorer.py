#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_patch_reference_candidate_scorer
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_patch_reference_candidate_scorer


if __name__ == "__main__":
    run_patch_reference_candidate_scorer(parse_args())
