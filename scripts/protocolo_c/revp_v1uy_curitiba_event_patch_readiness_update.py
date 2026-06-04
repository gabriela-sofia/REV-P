#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_event_patch_readiness_update
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_event_patch_readiness_update


if __name__ == "__main__":
    run_event_patch_readiness_update(parse_args())
