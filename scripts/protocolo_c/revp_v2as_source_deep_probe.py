#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_source_deep_probe
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_source_deep_probe


if __name__ == "__main__":
    run_source_deep_probe(parse_args())
