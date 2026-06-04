#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ae_common import parse_args, run_registry_consistency_qa
except ModuleNotFoundError:
    from revp_v2ae_common import parse_args, run_registry_consistency_qa


if __name__ == "__main__":
    run_registry_consistency_qa(parse_args())
