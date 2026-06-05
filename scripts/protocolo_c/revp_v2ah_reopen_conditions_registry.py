#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ah_common import parse_args, run_reopen_conditions_registry
except ModuleNotFoundError:
    from revp_v2ah_common import parse_args, run_reopen_conditions_registry


if __name__ == "__main__":
    run_reopen_conditions_registry(parse_args())
