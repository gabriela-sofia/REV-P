#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uv_curitiba_common import parse_args, run_public_event_discovery
except ModuleNotFoundError:
    from revp_v1uv_curitiba_common import parse_args, run_public_event_discovery


if __name__ == "__main__":
    run_public_event_discovery(parse_args())
