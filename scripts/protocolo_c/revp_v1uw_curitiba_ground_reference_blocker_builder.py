#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uw_curitiba_common import parse_args, run_ground_reference_blocker_builder
except ModuleNotFoundError:
    from revp_v1uw_curitiba_common import parse_args, run_ground_reference_blocker_builder


if __name__ == "__main__":
    run_ground_reference_blocker_builder(parse_args())
