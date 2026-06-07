#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_candidate_inventory_normalizer
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_candidate_inventory_normalizer


if __name__ == "__main__":
    run_candidate_inventory_normalizer(parse_args())
