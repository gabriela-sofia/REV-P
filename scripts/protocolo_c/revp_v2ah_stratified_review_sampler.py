#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ah_common import parse_args, run_stratified_review_sampler
except ModuleNotFoundError:
    from revp_v2ah_common import parse_args, run_stratified_review_sampler


if __name__ == "__main__":
    run_stratified_review_sampler(parse_args())
