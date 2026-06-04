#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_event_patch_readiness_updater
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_event_patch_readiness_updater


if __name__ == "__main__":
    run_event_patch_readiness_updater(parse_args())
