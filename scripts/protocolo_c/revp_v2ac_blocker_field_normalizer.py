#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ac_common import parse_args, run_blocker_field_normalizer
except ModuleNotFoundError:
    from revp_v2ac_common import parse_args, run_blocker_field_normalizer


if __name__ == "__main__":
    run_blocker_field_normalizer(parse_args())
