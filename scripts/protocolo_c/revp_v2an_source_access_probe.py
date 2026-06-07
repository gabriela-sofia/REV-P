#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_source_access_probe
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_source_access_probe


if __name__ == "__main__":
    run_source_access_probe(parse_args())
