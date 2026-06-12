#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_audit_hazard_typing
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_audit_hazard_typing
if __name__ == "__main__":
    run_audit_hazard_typing(parse_args())
