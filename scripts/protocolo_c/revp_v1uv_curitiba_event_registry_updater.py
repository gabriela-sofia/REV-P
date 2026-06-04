#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uv_curitiba_common import parse_args, run_event_registry_updater
except ModuleNotFoundError:
    from revp_v1uv_curitiba_common import parse_args, run_event_registry_updater


if __name__ == "__main__":
    run_event_registry_updater(parse_args())
