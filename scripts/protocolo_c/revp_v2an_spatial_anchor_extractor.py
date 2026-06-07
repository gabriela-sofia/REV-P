#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_spatial_anchor_extractor
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_spatial_anchor_extractor


if __name__ == "__main__":
    run_spatial_anchor_extractor(parse_args())
