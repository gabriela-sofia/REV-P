#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_spatial_anchor_strength_builder
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_spatial_anchor_strength_builder


if __name__ == "__main__":
    run_spatial_anchor_strength_builder(parse_args())
