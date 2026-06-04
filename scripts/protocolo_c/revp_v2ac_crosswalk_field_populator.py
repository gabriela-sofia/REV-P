#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ac_common import parse_args, run_crosswalk_field_populator
except ModuleNotFoundError:
    from revp_v2ac_common import parse_args, run_crosswalk_field_populator


if __name__ == "__main__":
    run_crosswalk_field_populator(parse_args())
