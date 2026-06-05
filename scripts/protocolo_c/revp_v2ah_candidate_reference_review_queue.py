#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ah_common import parse_args, run_candidate_reference_review_queue
except ModuleNotFoundError:
    from revp_v2ah_common import parse_args, run_candidate_reference_review_queue


if __name__ == "__main__":
    run_candidate_reference_review_queue(parse_args())
