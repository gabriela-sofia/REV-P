#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_deep_probe_priority_builder
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_deep_probe_priority_builder


if __name__ == "__main__":
    run_deep_probe_priority_builder(parse_args())
