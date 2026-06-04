#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ag_common import parse_args, run_crosswalk_source_inventory
except ModuleNotFoundError:
    from revp_v2ag_common import parse_args, run_crosswalk_source_inventory


if __name__ == "__main__":
    run_crosswalk_source_inventory(parse_args())
