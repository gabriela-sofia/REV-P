#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_coordinate_sanity_validator
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_coordinate_sanity_validator


if __name__ == "__main__":
    run_coordinate_sanity_validator(parse_args())
