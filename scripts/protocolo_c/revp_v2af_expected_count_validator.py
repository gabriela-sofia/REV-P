#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2af_common import parse_args, run_expected_count_validator
except ModuleNotFoundError:
    from revp_v2af_common import parse_args, run_expected_count_validator


if __name__ == "__main__":
    run_expected_count_validator(parse_args())
