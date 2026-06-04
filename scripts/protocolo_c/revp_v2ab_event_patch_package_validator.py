#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ab_common import parse_args, run_event_patch_package_validator
except ModuleNotFoundError:
    from revp_v2ab_common import parse_args, run_event_patch_package_validator


if __name__ == "__main__":
    run_event_patch_package_validator(parse_args())
