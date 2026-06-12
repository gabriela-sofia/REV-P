#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_audit_license_crs
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_audit_license_crs
if __name__ == "__main__":
    run_audit_license_crs(parse_args())
