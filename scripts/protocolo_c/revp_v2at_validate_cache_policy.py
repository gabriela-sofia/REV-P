#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_validate_cache_policy
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_validate_cache_policy
if __name__ == "__main__":
    run_validate_cache_policy(parse_args())
