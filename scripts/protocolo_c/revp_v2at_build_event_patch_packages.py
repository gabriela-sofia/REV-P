#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_build_event_patch_packages
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_build_event_patch_packages
if __name__ == "__main__":
    run_build_event_patch_packages(parse_args())
