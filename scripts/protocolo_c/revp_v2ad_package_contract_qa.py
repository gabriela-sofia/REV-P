#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ad_common import parse_args, run_package_contract_qa
except ModuleNotFoundError:
    from revp_v2ad_common import parse_args, run_package_contract_qa


if __name__ == "__main__":
    run_package_contract_qa(parse_args())
