#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ab_common import parse_args, run_temporal_field_contract_enforcer
except ModuleNotFoundError:
    from revp_v2ab_common import parse_args, run_temporal_field_contract_enforcer


if __name__ == "__main__":
    run_temporal_field_contract_enforcer(parse_args())
