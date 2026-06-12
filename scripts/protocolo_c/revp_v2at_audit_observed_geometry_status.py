#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_audit_observed_geometry_status
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_audit_observed_geometry_status
if __name__ == "__main__":
    run_audit_observed_geometry_status(parse_args())
