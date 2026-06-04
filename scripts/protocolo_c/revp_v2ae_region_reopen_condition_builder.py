#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ae_common import parse_args, run_region_reopen_condition_builder
except ModuleNotFoundError:
    from revp_v2ae_common import parse_args, run_region_reopen_condition_builder


if __name__ == "__main__":
    run_region_reopen_condition_builder(parse_args())
