#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ae_common import parse_args, run_canonical_event_registry_builder
except ModuleNotFoundError:
    from revp_v2ae_common import parse_args, run_canonical_event_registry_builder


if __name__ == "__main__":
    run_canonical_event_registry_builder(parse_args())
