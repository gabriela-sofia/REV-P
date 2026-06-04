#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uz_common import parse_args, run_curitiba_event_patch_hold_updater
except ModuleNotFoundError:
    from revp_v1uz_common import parse_args, run_curitiba_event_patch_hold_updater


if __name__ == "__main__":
    run_curitiba_event_patch_hold_updater(parse_args())
