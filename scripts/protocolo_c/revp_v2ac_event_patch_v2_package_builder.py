#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ac_common import parse_args, run_event_patch_v2_package_builder
except ModuleNotFoundError:
    from revp_v2ac_common import parse_args, run_event_patch_v2_package_builder


if __name__ == "__main__":
    run_event_patch_v2_package_builder(parse_args())
