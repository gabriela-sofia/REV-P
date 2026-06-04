#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ag_common import parse_args, run_explicit_crosswalk_detector
except ModuleNotFoundError:
    from revp_v2ag_common import parse_args, run_explicit_crosswalk_detector


if __name__ == "__main__":
    run_explicit_crosswalk_detector(parse_args())
