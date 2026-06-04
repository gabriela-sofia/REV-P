#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ae_common import parse_args, run_safe_use_policy_registry_builder
except ModuleNotFoundError:
    from revp_v2ae_common import parse_args, run_safe_use_policy_registry_builder


if __name__ == "__main__":
    run_safe_use_policy_registry_builder(parse_args())
