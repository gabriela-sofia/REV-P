#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ag_common import parse_args, run_patch_identity_key_extractor
except ModuleNotFoundError:
    from revp_v2ag_common import parse_args, run_patch_identity_key_extractor


if __name__ == "__main__":
    run_patch_identity_key_extractor(parse_args())
