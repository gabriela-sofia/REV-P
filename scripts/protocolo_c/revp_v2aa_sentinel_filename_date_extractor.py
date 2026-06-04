#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_sentinel_filename_date_extractor
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_sentinel_filename_date_extractor


if __name__ == "__main__":
    run_sentinel_filename_date_extractor(parse_args())
