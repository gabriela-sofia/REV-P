#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_patch_source_registry_scanner
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_patch_source_registry_scanner


if __name__ == "__main__":
    run_patch_source_registry_scanner(parse_args())
