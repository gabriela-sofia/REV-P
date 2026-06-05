#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ah_common import parse_args, run_safe_tcc_export_builder
except ModuleNotFoundError:
    from revp_v2ah_common import parse_args, run_safe_tcc_export_builder


if __name__ == "__main__":
    run_safe_tcc_export_builder(parse_args())
