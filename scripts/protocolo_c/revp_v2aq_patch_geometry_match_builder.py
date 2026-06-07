#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_patch_geometry_match_builder
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_patch_geometry_match_builder


if __name__ == "__main__":
    run_patch_geometry_match_builder(parse_args())
