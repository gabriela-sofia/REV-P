#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uo_multiregion_common import run_multiregion_event_registry_builder, simple_main
except ModuleNotFoundError:
    from revp_v1uo_multiregion_common import run_multiregion_event_registry_builder, simple_main

if __name__ == "__main__":
    simple_main(run_multiregion_event_registry_builder)
