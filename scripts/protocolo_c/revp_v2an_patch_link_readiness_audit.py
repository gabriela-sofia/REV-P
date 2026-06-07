#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_patch_link_readiness_audit
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_patch_link_readiness_audit


if __name__ == "__main__":
    run_patch_link_readiness_audit(parse_args())
