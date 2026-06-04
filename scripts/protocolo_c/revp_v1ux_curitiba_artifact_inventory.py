#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_artifact_inventory
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_artifact_inventory


if __name__ == "__main__":
    run_artifact_inventory(parse_args())
