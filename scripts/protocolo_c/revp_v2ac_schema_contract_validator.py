#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ac_common import parse_args, run_schema_contract_validator
except ModuleNotFoundError:
    from revp_v2ac_common import parse_args, run_schema_contract_validator


if __name__ == "__main__":
    run_schema_contract_validator(parse_args())
