#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ab_common import parse_args, run_patch_namespace_inventory
except ModuleNotFoundError:
    from revp_v2ab_common import parse_args, run_patch_namespace_inventory


if __name__ == "__main__":
    run_patch_namespace_inventory(parse_args())
